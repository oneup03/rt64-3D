//
// RT64
//

#include "rt64_present_queue.h"

#include "common/rt64_thread.h"
#include "rhi/rt64_render_hooks.h"

#include "rt64_workload_queue.h"

#ifdef LEIASR_SUPPORTED
#   include "contrib/plume/plume_d3d12.h"
#endif

namespace RT64 {
    // PresentQueue

    PresentQueue::PresentQueue() {
        reset();
    }

    PresentQueue::~PresentQueue() {
        presentThreadRunning = false;
        cursorCondition.notify_all();

        if (presentThread != nullptr) {
            presentThread->join();
            delete presentThread;
        }

        presentIdCondition.notify_all();
    }

    void PresentQueue::reset() {
        threadCursor = 0;
        writeCursor = 0;
        barrierCursor = 0;
        presentId = 0;
    }

    void PresentQueue::advanceToNextPresent() {
        int nextWriteCursor = (writeCursor + 1) % presents.size();

        // Stall the thread until the barrier is lifted if we're trying to write on a present being used by the GPU.
        bool waitForBarrier;
        do {
            const std::scoped_lock lock(cursorMutex);
            waitForBarrier = (nextWriteCursor == barrierCursor);
        } while (waitForBarrier);

        // Modify the cursor and notify anything waiting on the queue.
        {
            const std::scoped_lock lock(cursorMutex);
            writeCursor = nextWriteCursor;
        }

        cursorCondition.notify_all();
    }

    void PresentQueue::repeatLastPresent() {
        {
            const std::scoped_lock lock(cursorMutex);
            threadCursor = previousWriteCursor();
        }

        cursorCondition.notify_all();
    }

    uint32_t PresentQueue::previousWriteCursor() const {
        if (writeCursor > 0) {
            return writeCursor - 1;
        }
        else {
            return uint32_t(presents.size()) - 1;
        }
    }

    void PresentQueue::waitForIdle() {
        std::unique_lock<std::mutex> threadLock(threadMutex);
    }

    void PresentQueue::waitForPresentId(uint64_t waitId) {
        std::unique_lock<std::mutex> presentLock(presentIdMutex);
        presentIdCondition.wait(presentLock, [&]() {
            return (waitId <= presentId) || !presentThreadRunning;
        });
    }

    void PresentQueue::setup(const External &ext) {
        this->ext = ext;

        viRenderer = std::make_unique<VIRenderer>();
        stereoRenderer = std::make_unique<StereoRenderer>();

        presentThreadRunning = true;
        presentThread = new std::thread(&PresentQueue::threadLoop, this);
    }

    void PresentQueue::threadPresent(const Present &present, bool &swapChainValid) {
        FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
        RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
        const bool usingMSAA = (targetManager.multisampling.sampleCount > 1);
        hlslpp::float2 resolutionScale;
        EnhancementConfiguration::Presentation::Mode presentationMode;
        bool removeBlackBorders;
        UserConfiguration::RefreshRate refreshRate;
        UserConfiguration::Filtering filtering;
        uint32_t viOriginalRate;
        uint32_t targetRate;
        {
            std::scoped_lock<std::mutex> configurationLock(ext.sharedResources->configurationMutex);
            resolutionScale = ext.sharedResources->resolutionScale;
            presentationMode = ext.sharedResources->enhancementConfig.presentation.mode;
            removeBlackBorders = ext.sharedResources->enhancementConfig.presentation.removeBlackBorders;
            refreshRate = ext.sharedResources->userConfig.refreshRate;
            filtering = ext.sharedResources->userConfig.filtering;
            viOriginalRate = ext.sharedResources->viOriginalRate;
            targetRate = ext.sharedResources->targetRate;
        }

        RenderTarget *colorTarget = nullptr;
        int32_t framesToPresent = 1;
        bool lockedWorkloadMutex = false;
        InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];

        // TODO: There's a possible race condition interactions that can happen while the workload
        // queue is rendering extra frames and the present event is processed while it's generating
        // interpolated frames. When the framebuffer manager or the render target manager maps are
        // modified while the present queue is retrieving the framebuffer or the target. These can
        // likely be solved by locking the access to the managers during modification.
        
        // Perform any external write operations indicated by the event.
        if (!present.fbOperations.empty()) {
            const std::scoped_lock lock(screenFbChangePoolMutex);
            {
                RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                fbManager.performOperations(ext.presentGraphicsWorker, &screenFbChangePool, nullptr, ext.shaderLibrary, nullptr,
                    present.fbOperations, targetManager, resolutionScale, 0, 0, nullptr);
            }
        }

        // Present the VI specified by the event.
        // Attempt to find the matching framebuffer for the VI based on the origin address.
        // If that fails, we look at the shared storage.
        if (present.screenVI.visible()) {
            Framebuffer *viFb = nullptr;
            if (!viewRDRAM) {
                viFb = fbManager.find(present.screenVI.fbAddress());
            }

            Framebuffer *presentFb = viFb;
            
            // Show the framebuffer the debugger has requested instead.
            if (present.debuggerFramebuffer.view) {
                Framebuffer *candidateFb = fbManager.find(present.debuggerFramebuffer.address);
                if (candidateFb != nullptr) {
                    presentFb = candidateFb;
                }
            }
            
            if ((presentFb != nullptr) && (viFb != nullptr)) {
                for (uint32_t colorAddress : ext.sharedResources->colorImageAddressVector) {
                    Framebuffer *colorFb = fbManager.find(colorAddress);
                    if (colorFb == nullptr) {
                        continue;
                    }

                    // Always default to interpolation being disabled for all modified framebuffers.
                    colorFb->interpolationEnabled = false;
                    
                    // When the skip buffering option is on, we check the video history to find if any of the framebuffers that
                    // were drawn in this frame have been previously used for presentation. This is ignored when the debugger
                    // has forced viewing a particular framebuffer.
                    if (!present.debuggerFramebuffer.view && (presentationMode == EnhancementConfiguration::Presentation::Mode::SkipBuffering)) {
                        for (size_t h = 0; h < viHistory.history.size(); h++) {
                            const VIHistory::Present &entry = viHistory.history[h];
                            if ((colorFb->addressStart == entry.vi.fbAddress()) && (colorFb->width == entry.fbWidth) && (colorFb->siz == entry.vi.fbSiz()) && entry.vi.compatibleWith(present.screenVI)) {
                                presentFb = colorFb;
                                break;
                            }
                        }
                    }

                    // Present early (or games that behave like it) will make it so that the presented image is a color image
                    // that the workload modified. We run a basic check to see if that holds true to indicate it was presented
                    // so interpolation is possible.
                    if (colorFb == presentFb) {
                        presentFb->interpolationEnabled = true;
                        break;
                    }
                }

                if (presentFb->interpolationEnabled) {
                    framesToPresent = frameCounters.count;
                }
                else {
                    lockedWorkloadMutex = true;
                    ext.sharedResources->workloadMutex.lock();
                }

                RenderTargetKey colorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, Framebuffer::Type::Color);
                colorTarget = &targetManager.get(colorTargetKey, true);
                if (!colorTarget->isEmpty()) {
                    // If a depth framebuffer is about to be shown, convert it to color.
                    if (presentFb->isLastWriteDifferent(Framebuffer::Type::Color)) {
                        RenderTargetKey otherColorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, presentFb->lastWriteType);
                        RenderTarget &otherColorTarget = targetManager.get(otherColorTargetKey, true);
                        if (!otherColorTarget.isEmpty()) {
                            const FixedRect &r = presentFb->lastWriteRect;
                            RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                            colorTarget->copyFromTarget(ext.presentGraphicsWorker, &otherColorTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                        }
                    }
                }
                else {
                    colorTarget = nullptr;
                }

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, viFb->width);
                }
            }
            else {
                uint32_t fbAddress = present.screenVI.fbAddress();

                // Use a scratch framebuffer to upload the RAM to the render target.
                hlslpp::uint2 fbSize = present.screenVI.fbSize();
                scratchFb.addressStart = fbAddress;
                scratchFb.width = fbSize.x;
                scratchFb.height = fbSize.y;
                scratchFb.siz = present.screenVI.fbSiz();

                lockedWorkloadMutex = true;
                ext.sharedResources->workloadMutex.lock();

                RenderTargetKey colorTargetKey(fbAddress, scratchFb.width, scratchFb.siz, Framebuffer::Type::Color);
                colorTarget = &targetManager.get(colorTargetKey, true);
                colorTarget->resize(ext.presentGraphicsWorker, scratchFb.width, scratchFb.height);
                colorTarget->resolutionScale = { 1.0f, 1.0f };
                colorTarget->downsampleMultiplier = 1;

                scratchFb.nativeTarget.resetBufferHistory();

                {
                    RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                    colorTarget->clearColorTarget(ext.presentGraphicsWorker);
                    FramebufferChange *colorFbChange = scratchFb.readChangeFromBytes(ext.presentGraphicsWorker, scratchFbChangePool, Framebuffer::Type::Color,
                        G_IM_FMT_RGBA, present.storage.data(), 0, scratchFb.height, ext.shaderLibrary);

                    if (colorFbChange != nullptr) {
                        colorTarget->copyFromChanges(ext.presentGraphicsWorker, *colorFbChange, scratchFb.width, scratchFb.height, 0, ext.shaderLibrary);
                    }
                }

                scratchFbChangePool.reset();

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, fbSize.x);
                }
            }
        }

        // Create the framebuffers if necessary.
        if (swapChainFramebuffers.empty()) {
            uint32_t textureCount = ext.swapChain->getTextureCount();
            swapChainFramebuffers.resize(textureCount);
            for (uint32_t i = 0; i < textureCount; i++) {
                const RenderTexture *swapChainTexture = ext.swapChain->getTexture(i);
                swapChainFramebuffers[i] = ext.device->createFramebuffer(RenderFramebufferDesc(&swapChainTexture, 1));
            }
        }
        
        for (int32_t i = 0; i < framesToPresent; i++) {
            uint32_t frameCountersNextPresented = 0;
            if ((framesToPresent > 1) && (usingMSAA || (i > 0))) {
                // Stall until the interpolated color target is available.
                const uint32_t targetIndex = usingMSAA ? i : (i - 1);
                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                    return (frameCounters.available > targetIndex) || ((frameCounters.available == targetIndex) && frameCounters.skipped);
                });

                // Do not present any more frames after this one after reaching the last available frame if the workload was skipped.
                if ((frameCounters.available == targetIndex) && frameCounters.skipped) {
                    framesToPresent = std::min(int(frameCounters.available), i + 1);
                    frameCountersNextPresented = frameCounters.count;
                }
                else {
                    frameCountersNextPresented = frameCounters.presented + 1;
                }

                if (i < framesToPresent) {
                    uint32_t targetIndex = usingMSAA ? i : (i - 1);
                    colorTarget = ext.sharedResources->interpolatedColorTargets[targetIndex].get();
                }
                else {
                    colorTarget = nullptr;
                }
            }
            else if (framesToPresent == 1) {
                frameCountersNextPresented = frameCounters.count;
            }

            uint32_t swapChainIndex = 0;
            const bool presentFrame = (i < framesToPresent) && swapChainValid;
            if (presentFrame) {
                swapChainValid = ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex);
            }

            if (presentFrame && swapChainValid) {
                // Draw the framebuffer with the VI renderer.
                RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
                RenderFramebuffer *swapChainFramebuffer = swapChainFramebuffers[swapChainIndex].get();
                RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                commandList->begin();
                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
                
                VIRenderer::RenderParams renderParams;
                if (colorTarget != nullptr) {
                    renderParams.device = ext.device;
                    renderParams.commandList = commandList;
                    renderParams.swapChain = ext.swapChain;
                    renderParams.shaderLibrary = ext.shaderLibrary;
                    renderParams.textureFormat = colorTarget->format;
                    renderParams.resolutionScale = colorTarget->resolutionScale;
                    renderParams.downsamplingScale = 1;
                    renderParams.filtering = filtering;
                    renderParams.vi = &present.screenVI;
                    renderParams.removeBlackBorders = removeBlackBorders;

                    const bool useDownsampling = (colorTarget->downsampleMultiplier > 1);
                    if (useDownsampling) {
                        colorTarget->downsampleTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->downsampledTexture.get();
                        renderParams.textureWidth = colorTarget->width / colorTarget->downsampleMultiplier;
                        renderParams.textureHeight = colorTarget->height / colorTarget->downsampleMultiplier;
                        renderParams.downsamplingScale = colorTarget->downsampleMultiplier;
                    }
                    else {
                        colorTarget->resolveTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->getResolvedTexture();
                        renderParams.textureWidth = colorTarget->width;
                        renderParams.textureHeight = colorTarget->height;
                    }
                }
                
                commandList->setFramebuffer(swapChainFramebuffer);
                commandList->clearColor();

                // Tracks whether the LeiaSR weaver consumed the SbS intermediate
                // and wrote the final image to the swap chain itself. Hoisted out
                // of the inner stereo branch because the UI-overlay block below
                // needs to know (UI was already composed into the weaved image,
                // so don't run the post-weave SbS overlay again).
                bool composedThroughWeaver = false;

                if (renderParams.texture != nullptr) {
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(renderParams.texture, RenderTextureLayout::SHADER_READ));

                    // Branch on stereoMode. The mono path is unchanged. For stereo, we
                    // pair the natural color target (Left eye) with stereoRightColorTarget
                    // (Right eye, populated by the workload thread's second pass) and let
                    // StereoCompose pack them into the requested layout. If the right-eye
                    // target hasn't been populated yet (first stereo frame, or any frame
                    // where the workload couldn't run the second pass), we fall back to
                    // binding the left texture to both slots.
                    const auto stereoMode = ext.sharedResources->userConfig.stereoMode;
                    if (stereoMode != UserConfiguration::StereoMode::Off) {
                        RenderTexture *rightTexture = renderParams.texture;
                        uint32_t rightTextureWidth = renderParams.textureWidth;
                        uint32_t rightTextureHeight = renderParams.textureHeight;
                        // Pick the right-eye target that matches whatever left-eye color
                        // target this present iteration is using: interpolatedTargets[i]
                        // pairs with stereoRightInterpolatedTargets[i]; the natural
                        // target pairs with stereoRightColorTarget.
                        RenderTarget *rightTarget = nullptr;
                        if ((framesToPresent > 1) && (usingMSAA || (i > 0))) {
                            const uint32_t targetIndex = usingMSAA ? i : (i - 1);
                            auto &rightSlots = ext.sharedResources->stereoRightInterpolatedTargets;
                            if (targetIndex < rightSlots.size()) {
                                rightTarget = rightSlots[targetIndex].get();
                            }
                        }
                        else {
                            rightTarget = ext.sharedResources->stereoRightColorTarget.get();
                        }
                        if (rightTarget != nullptr) {
                            const bool useRightDownsampling = (rightTarget->downsampleMultiplier > 1);
                            if (useRightDownsampling) {
                                rightTarget->downsampleTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                                rightTexture = rightTarget->downsampledTexture.get();
                                rightTextureWidth = rightTarget->width / rightTarget->downsampleMultiplier;
                                rightTextureHeight = rightTarget->height / rightTarget->downsampleMultiplier;
                            }
                            else {
                                rightTarget->resolveTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                                rightTexture = rightTarget->getResolvedTexture();
                                rightTextureWidth = rightTarget->width;
                                rightTextureHeight = rightTarget->height;
                            }
                            commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(rightTexture, RenderTextureLayout::SHADER_READ));
                        }

                        StereoRenderer::RenderParams stereoParams;
                        stereoParams.device = renderParams.device;
                        stereoParams.commandList = renderParams.commandList;
                        stereoParams.leftTexture = renderParams.texture;
                        stereoParams.rightTexture = rightTexture;
                        stereoParams.swapChain = renderParams.swapChain;
                        stereoParams.shaderLibrary = renderParams.shaderLibrary;
                        stereoParams.textureFormat = renderParams.textureFormat;
                        stereoParams.resolutionScale = renderParams.resolutionScale;
                        stereoParams.downsamplingScale = renderParams.downsamplingScale;
                        stereoParams.textureWidth = renderParams.textureWidth;
                        stereoParams.textureHeight = renderParams.textureHeight;
                        stereoParams.stereoMode = stereoMode;
                        stereoParams.vi = renderParams.vi;
                        stereoParams.removeBlackBorders = renderParams.removeBlackBorders;
                        // Always fill the full target for stereo: VIRenderer's
                        // letterbox math would otherwise pillar/letterbox the
                        // SbS/TaB image inside the swap chain, which breaks
                        // full-SbS AR glasses (they need the whole frame),
                        // 3D-TV row interlacing (rows must align with screen
                        // scanlines), and the LeiaSR compose intermediate
                        // (the weaver wants a packed-stereo input texture).
                        // Run the game in fullscreen + auto resolution to get
                        // the swap chain at the desktop's native size.
                        stereoParams.fillFullTarget = true;

                        // Forward the workload's aspectRatioScale so the
                        // compose shader can crop each eye slot to the
                        // content area of the eye texture. resolutionScale
                        // is { multiplier * aspectRatioScale, multiplier },
                        // so x/y recovers the original aspectRatioScale.
                        // Defaults to 1 if resolution scale isn't available
                        // yet (early frames), which makes the shader treat
                        // the eye texture as fully filled.
                        const auto resScale = ext.sharedResources->resolutionScale;
                        const float resScaleX = static_cast<float>(resScale.x);
                        const float resScaleY = static_cast<float>(resScale.y);
                        stereoParams.aspectRatioScale = (resScaleY > 1e-6f) ? (resScaleX / resScaleY) : 1.0f;

#                   ifdef LEIASR_SUPPORTED
                        // LeiaSR weaving requires D3D12 and the SR Platform
                        // service. When all of that is in place: compose SbS
                        // into a desktop-resolution intermediate, then hand
                        // that texture to the lenticular weaver which writes
                        // to the swap chain instead of us. If anything's
                        // missing we silently fall through to the regular
                        // SbS-to-swap-chain compose below.
                        if ((stereoMode == UserConfiguration::StereoMode::LeiaSR) &&
                            (ext.createdGraphicsAPI == UserConfiguration::GraphicsAPI::D3D12) &&
                            (ext.appWindow != nullptr) && (ext.appWindow->windowHandle != nullptr)) {
                            // Attempt one-shot SDK init. Failure (SR Platform
                            // service missing, etc.) sets leiaSRInitAttempted
                            // so we don't keep retrying every frame.
                            if (!leiaSRWeaver.isAvailable() && !leiaSRInitAttempted) {
                                auto *d3dDevice = static_cast<plume::D3D12Device *>(ext.device);
                                leiaSRWeaver.initialize(d3dDevice ? d3dDevice->d3d : nullptr, ext.appWindow->windowHandle);
                                leiaSRInitAttempted = true;
                            }

                            if (leiaSRWeaver.isAvailable()) {
                                const uint32_t perEyeW   = ext.swapChain->getWidth();
                                const uint32_t perEyeH   = ext.swapChain->getHeight();
                                const uint32_t sbsW      = perEyeW * 2u;
                                const uint32_t sbsH      = perEyeH;
                                if ((leiaSRComposeTexture == nullptr) || (leiaSRComposeWidth != sbsW) || (leiaSRComposeHeight != sbsH)) {
                                    const RenderFormat composeFormat = RenderFormat::R8G8B8A8_UNORM;
                                    RenderClearValue composeClear = RenderClearValue::Color(RenderColor(0.0f, 0.0f, 0.0f, 1.0f), composeFormat);
                                    leiaSRComposeTexture = ext.device->createTexture(RenderTextureDesc::ColorTarget(sbsW, sbsH, composeFormat, RenderMultisampling(), &composeClear));
                                    const RenderTexture *composeAttachment = leiaSRComposeTexture.get();
                                    leiaSRComposeFramebuffer = ext.device->createFramebuffer(RenderFramebufferDesc(&composeAttachment, 1));
                                    leiaSRComposeWidth  = sbsW;
                                    leiaSRComposeHeight = sbsH;
                                }

                                // 1. Compose SbS-packed stereo into the 2*perEyeW × perEyeH intermediate.
                                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(leiaSRComposeTexture.get(), RenderTextureLayout::COLOR_WRITE));
                                commandList->setFramebuffer(leiaSRComposeFramebuffer.get());
                                commandList->clearColor();

                                StereoRenderer::RenderParams composeParams = stereoParams;
                                composeParams.stereoMode    = UserConfiguration::StereoMode::SideBySide;
                                composeParams.targetWidth   = sbsW;
                                composeParams.targetHeight  = sbsH;
                                composeParams.fillFullTarget = true;
                                stereoRenderer->render(composeParams);

                                // 2. Composite the Configuration GUI into the SAME SbS intermediate
                                //    BEFORE the weave so the weaver sees game + UI as a single SbS image.
                                RenderHookDraw *leiaDrawHook = GetRenderHookDraw();
                                if (leiaDrawHook != nullptr) {
                                    const uint32_t uiWidth  = perEyeW;
                                    const uint32_t uiHeight = perEyeH;
                                    if ((stereoUITexture == nullptr) || (stereoUITextureWidth != uiWidth) || (stereoUITextureHeight != uiHeight)) {
                                        RenderClearValue uiClear = RenderClearValue::Color(RenderColor(0.0f, 0.0f, 0.0f, 0.0f), RenderFormat::B8G8R8A8_UNORM);
                                        stereoUITexture = ext.device->createTexture(RenderTextureDesc::ColorTarget(uiWidth, uiHeight, RenderFormat::B8G8R8A8_UNORM, RenderMultisampling(), &uiClear));
                                        const RenderTexture *uiColorAttachment = stereoUITexture.get();
                                        stereoUIFramebuffer = ext.device->createFramebuffer(RenderFramebufferDesc(&uiColorAttachment, 1));
                                        stereoUITextureWidth = uiWidth;
                                        stereoUITextureHeight = uiHeight;
                                    }

                                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(stereoUITexture.get(), RenderTextureLayout::COLOR_WRITE));
                                    commandList->setFramebuffer(stereoUIFramebuffer.get());
                                    commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                                    leiaDrawHook(commandList, stereoUIFramebuffer.get());

                                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(stereoUITexture.get(), RenderTextureLayout::SHADER_READ));
                                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(leiaSRComposeTexture.get(), RenderTextureLayout::COLOR_WRITE));
                                    commandList->setFramebuffer(leiaSRComposeFramebuffer.get());

                                    StereoRenderer::RenderParams uiOverlayParams = composeParams;
                                    uiOverlayParams.leftTexture = stereoUITexture.get();
                                    uiOverlayParams.rightTexture = stereoUITexture.get();
                                    uiOverlayParams.textureFormat = RenderFormat::B8G8R8A8_UNORM;
                                    uiOverlayParams.resolutionScale = { 1.0f, 1.0f };
                                    uiOverlayParams.downsamplingScale = 1;
                                    uiOverlayParams.textureWidth = uiWidth;
                                    uiOverlayParams.textureHeight = uiHeight;
                                    uiOverlayParams.aspectRatioScale = 1.0f;
                                    uiOverlayParams.isUIOverlay = true;
                                    stereoRenderer->render(uiOverlayParams);
                                }

                                // 3. Transition intermediate to SHADER_READ and rebind swap chain.
                                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(leiaSRComposeTexture.get(), RenderTextureLayout::SHADER_READ));
                                commandList->setFramebuffer(swapChainFramebuffer);

                                // 3.5. Reset the rasterizer viewport/scissor to swap-chain dims so
                                //      the weaver renders into the swap chain instead of inheriting
                                //      the previous sbsW × sbsH compose viewport.
                                commandList->setViewports(RenderViewport(0.0f, 0.0f, float(ext.swapChain->getWidth()), float(ext.swapChain->getHeight())));
                                commandList->setScissors(RenderRect(0, 0, int32_t(ext.swapChain->getWidth()), int32_t(ext.swapChain->getHeight())));

                                // 4. Hand off to the LeiaSR weaver.
                                auto *d3dCommandList = static_cast<plume::D3D12CommandList *>(commandList);
                                auto *d3dInputTexture = static_cast<plume::D3D12Texture *>(leiaSRComposeTexture.get());
                                auto *d3dSwapChain = static_cast<plume::D3D12SwapChain *>(ext.swapChain);
                                if (d3dCommandList && d3dInputTexture && d3dSwapChain) {
                                    D3D12_VIEWPORT weaveViewport = {};
                                    weaveViewport.Width = static_cast<float>(ext.swapChain->getWidth());
                                    weaveViewport.Height = static_cast<float>(ext.swapChain->getHeight());
                                    weaveViewport.MinDepth = 0.0f;
                                    weaveViewport.MaxDepth = 1.0f;
                                    D3D12_RECT weaveScissor = {};
                                    weaveScissor.right = static_cast<LONG>(ext.swapChain->getWidth());
                                    weaveScissor.bottom = static_cast<LONG>(ext.swapChain->getHeight());
                                    leiaSRWeaver.weave(d3dInputTexture->d3d,
                                        d3dSwapChain->nativeFormat,
                                        d3dCommandList->d3d, weaveViewport, weaveScissor);
                                    composedThroughWeaver = true;
                                }
                            }
                        }
#                   endif // LEIASR_SUPPORTED

                        if (!composedThroughWeaver) {
                            stereoRenderer->render(stereoParams);
                        }
                    }
                    else {
                        viRenderer->render(renderParams);
                    }
                }

                RenderHookDraw *drawHook = GetRenderHookDraw();
                if (drawHook != nullptr) {
                    const auto stereoModeHook = ext.sharedResources->userConfig.stereoMode;
#               ifdef LEIASR_SUPPORTED
                    // When the LeiaSR weaver handled the game compose, the UI
                    // was already alpha-blended into the SbS intermediate and
                    // weaved together. Drawing it again here would render
                    // SbS-layout UI on top of the weaved image.
                    const bool skipUIOverlay = composedThroughWeaver;
#               else
                    constexpr bool skipUIOverlay = false;
#               endif
                    if (skipUIOverlay) {
                        // UI already composed via the weaver above.
                    }
                    else if (stereoModeHook != UserConfiguration::StereoMode::Off) {
                        // Render the Configuration GUI into an off-screen texture
                        // then alpha-blend it over the already-composed stereo
                        // image, mirrored per eye. The off-screen target is sized
                        // to the swap chain so the UI renders at its native scale.
                        const uint32_t uiWidth = ext.swapChain->getWidth();
                        const uint32_t uiHeight = ext.swapChain->getHeight();
                        if ((stereoUITexture == nullptr) || (stereoUITextureWidth != uiWidth) || (stereoUITextureHeight != uiHeight)) {
                            RenderClearValue uiClear = RenderClearValue::Color(RenderColor(0.0f, 0.0f, 0.0f, 0.0f), RenderFormat::B8G8R8A8_UNORM);
                            stereoUITexture = ext.device->createTexture(RenderTextureDesc::ColorTarget(uiWidth, uiHeight, RenderFormat::B8G8R8A8_UNORM, RenderMultisampling(), &uiClear));
                            const RenderTexture *uiColorAttachment = stereoUITexture.get();
                            stereoUIFramebuffer = ext.device->createFramebuffer(RenderFramebufferDesc(&uiColorAttachment, 1));
                            stereoUITextureWidth = uiWidth;
                            stereoUITextureHeight = uiHeight;
                        }

                        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(stereoUITexture.get(), RenderTextureLayout::COLOR_WRITE));
                        commandList->setFramebuffer(stereoUIFramebuffer.get());
                        commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));

                        drawHook(commandList, stereoUIFramebuffer.get());

                        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(stereoUITexture.get(), RenderTextureLayout::SHADER_READ));
                        commandList->setFramebuffer(swapChainFramebuffer);

                        StereoRenderer::RenderParams overlayParams;
                        overlayParams.device = ext.device;
                        overlayParams.commandList = commandList;
                        overlayParams.leftTexture = stereoUITexture.get();
                        overlayParams.rightTexture = stereoUITexture.get();
                        overlayParams.swapChain = ext.swapChain;
                        overlayParams.shaderLibrary = ext.shaderLibrary;
                        overlayParams.textureFormat = RenderFormat::B8G8R8A8_UNORM;
                        overlayParams.resolutionScale = { 1.0f, 1.0f };
                        overlayParams.downsamplingScale = 1;
                        overlayParams.textureWidth = uiWidth;
                        overlayParams.textureHeight = uiHeight;
                        overlayParams.stereoMode = stereoModeHook;
                        overlayParams.vi = &present.screenVI;
                        overlayParams.removeBlackBorders = removeBlackBorders;
                        overlayParams.isUIOverlay = true;
                        stereoRenderer->render(overlayParams);
                    }
                    else {
                        drawHook(commandList, swapChainFramebuffer);
                    }
                }

                {
                    const std::scoped_lock lock(inspectorMutex);
                    if (inspector != nullptr) {
                        inspector->draw(commandList);
                    }
                    
                    commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::PRESENT));
                    commandList->end();
                    const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                    RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
                    RenderCommandSemaphore *signalSemaphore = drawSemaphores[swapChainIndex].get();
                    ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, &signalSemaphore, 1, ext.presentGraphicsWorker->commandFence.get());
                    ext.presentGraphicsWorker->wait();
                }
            }

            if (lockedWorkloadMutex) {
                ext.sharedResources->workloadMutex.unlock();
                lockedWorkloadMutex = false;
            }
            
            if (frameCountersNextPresented > 0) {
                {
                    std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                    frameCounters.presented = frameCountersNextPresented;
                }

                ext.sharedResources->interpolatedCondition.notify_all();
            }

            // As soon as we're done with the first render target, we notify the workload queue it can proceed.
            if (i == 0) {
                notifyPresentId(present);
            }

            if (presentFrame && swapChainValid) {
                // Wait until the approximate time the next present should be at the current intended rate.
                if ((presentTimestamp != Timestamp()) && (targetRate > 0) && (targetRate > viOriginalRate)) {
                    Timer::preciseSleepUntil(presentTimestamp + std::chrono::nanoseconds(1'000'000'000 / targetRate));
                }

                if (presentWaitEnabled) {
                    ext.swapChain->wait();
                }

                RenderCommandSemaphore *waitSemaphore = drawSemaphores[swapChainIndex].get();
                presentTimestamp = Timer::current();
                swapChainValid = ext.swapChain->present(swapChainIndex, &waitSemaphore, 1);
                presentProfiler.logAndRestart();
            }
        }
    }

    void PresentQueue::skipInterpolation() {
        {
            std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
            InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
            frameCounters.presented = frameCounters.count;
        }

        ext.sharedResources->interpolatedCondition.notify_all();
    }

    void PresentQueue::notifyPresentId(const Present &present) {
        {
            std::scoped_lock<std::mutex> cursorLock(presentIdMutex);
            presentId = present.presentId;
        }

        presentIdCondition.notify_all();
    }
    
    void PresentQueue::threadAdvanceBarrier() {
        std::scoped_lock<std::mutex> cursorLock(cursorMutex);
        barrierCursor = (barrierCursor + 1) % presents.size();
    }

    void PresentQueue::threadLoop() {
        Thread::setCurrentThreadName("RT64 Present");

        // Create the semaphore the acquire method will use.
        acquiredSemaphore = ext.device->createCommandSemaphore();

        // Create as many semaphores to signal as textures there are.
        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
        }

        // Since the swap chain might not need a resize right away, detect present wait.
        presentWaitEnabled = ext.device->getCapabilities().presentWait;

        int processCursor = -1;
        bool skipPresent = false;
        uint32_t displayTimingRate = UINT32_MAX;
        const bool displayTiming = ext.device->getCapabilities().displayTiming;
        bool swapChainValid = !ext.swapChain->needsResize();
        while (presentThreadRunning) {
            {
                std::unique_lock<std::mutex> cursorLock(cursorMutex);
                cursorCondition.wait(cursorLock, [&]() {
                    return (writeCursor != threadCursor) || !presentThreadRunning;
                });

                if (presentThreadRunning) {
                    processCursor = threadCursor;
                    threadCursor = (threadCursor + 1) % presents.size();
                    skipPresent = (writeCursor != threadCursor);
                }
            }

            if (processCursor >= 0) {
                std::unique_lock<std::mutex> threadLock(threadMutex);
                const bool needsResize = ext.swapChain->needsResize() || !swapChainValid;
                if (needsResize) {
                    ext.presentGraphicsWorker->commandList->begin();
                    ext.presentGraphicsWorker->commandList->end();
                    ext.presentGraphicsWorker->execute();
                    ext.presentGraphicsWorker->wait();
                    swapChainValid = ext.swapChain->resize();
                    swapChainFramebuffers.clear();

                    if (swapChainValid) {
                        ext.sharedResources->setSwapChainSize(ext.swapChain->getWidth(), ext.swapChain->getHeight());
                        
                        // Texture count could've changed after resize, so new semaphores are needed.
                        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
                            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
                        }
                    }
                }

                if (needsResize || ext.appWindow->detectWindowMoved()) {
                    ext.appWindow->detectRefreshRate();
                    ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), displayTimingRate));
                }

                if (displayTiming) {
                    uint32_t newDisplayTimingRate = ext.swapChain->getRefreshRate();
                    if (newDisplayTimingRate == 0) {
                        newDisplayTimingRate = UINT32_MAX;
                    }

                    if (newDisplayTimingRate != displayTimingRate) {
                        ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), newDisplayTimingRate));
                        displayTimingRate = newDisplayTimingRate;
                    }
                }

                skipPresent = skipPresent || ext.swapChain->isEmpty();

                Present &present = presents[processCursor];
                ext.workloadQueue->waitForWorkloadId(present.workloadId);

                if (!presentThreadRunning) {
                    continue;
                }

                if (skipPresent) {
                    skipInterpolation();
                    notifyPresentId(present);
                }
                else {
                    threadPresent(present, swapChainValid);
                }

                if (!present.paused) {
                    if (!present.fbOperations.empty()) {
                        const std::scoped_lock lock(screenFbChangePoolMutex);
                        screenFbChangePool.release(present.fbOperations.front().writeChanges.id);
                        present.fbOperations.clear();
                    }

                    threadAdvanceBarrier();
                }

                processCursor = -1;
            }
        }

        // Transition the active swap chain render target out of the present state to avoid live references to the resource.
        uint32_t swapChainIndex = 0;
        if (!ext.swapChain->isEmpty() && ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex)) {
            RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
            ext.presentGraphicsWorker->commandList->begin();
            ext.presentGraphicsWorker->commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
            ext.presentGraphicsWorker->commandList->end();

            const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
            RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
            ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, nullptr, 0, ext.presentGraphicsWorker->commandFence.get());
            ext.presentGraphicsWorker->wait();
        }
    }
};
