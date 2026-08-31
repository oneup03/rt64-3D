//
// RT64
//

#include "rt64_workload_queue.h"

#include "common/rt64_thread.h"

#include "rt64_present_queue.h"
#include "render/rt64_stereo_depth_sampler.h"

#include <cmath>
#include <cstdio>
#include <atomic>
#include <cstdarg>
#include <cstring>
#include <cstdlib>

#define ENABLE_HIGH_RESOLUTION_RENDERER 1

namespace RT64 {
    // Temporary stereo depth bring-up instrumentation. DK64_STEREO_DEPTH_LOG=1
    // enables the readback and writes the sampled centre depth to
    // stereo_depth.log, alongside both candidate device-depth conventions
    // evaluated against DK64's near/far (10 / 1500). dynamic3d 4.1 warns not to
    // assume which convention a renderer uses, so this reports both and lets a
    // known-distance observation pick the right one. Remove once calibrated.
    // Depth readback for the depth-aware stereo features. On by default now
    // that the crash is fixed; set DK64_STEREO_DEPTH_SAMPLE=0 to turn it off.
    //
    // The crash was not what it looked like. The copy is issued from inside the
    // framebuffer loop, which looked like recording a texture copy during an
    // active render pass, but plume's copyTextureRegion ends the pass itself.
    // The real cause was that plume's VULKAN backend only implemented
    // buffer -> image copies: every other combination fell through to a generic
    // branch that dereferences the destination texture, which is null when the
    // destination is a buffer. D3D12 was fine throughout. The missing
    // image -> buffer path is now implemented in plume_vulkan.cpp.
    //
    // An earlier opt-IN variable also hid the fact that this code had never
    // executed at all - an empty log looked identical to a disabled one - hence
    // the explicit "sampling active" heartbeat in logStereoDepthSample.
    static bool stereoDepthSamplingEnabled() {
        static const bool enabled = [] {
            const char *v = std::getenv("DK64_STEREO_DEPTH_SAMPLE");
            return (v == nullptr) || ((v[0] != 0) && (v[0] != '0'));
        }();
        return enabled;
    }

    // Writes to both the log file and stderr. The console copy matters because
    // the file lands in whatever the process's working directory happens to be,
    // which is not always where you expect.
    static void stereoDepthLogLine(const char *fmt, ...) {
        static uint32_t linesWritten = 0;
        if (linesWritten >= 4000) {
            return;
        }
        linesWritten++;

        char buf[256];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);

        fprintf(stderr, "[stereo-depth] %s\n", buf);
        FILE *f = fopen("stereo_depth.log", "a");
        if (f != nullptr) {
            fprintf(f, "%s\n", buf);
            fclose(f);
        }
    }

    // Convergence the depth loop wants, in tenths of a slider unit, or 0 when it
    // has nothing to say and the user's manual value should stand. Written on
    // the render thread where the depth is sampled and read a few lines earlier
    // in the same loop on the following frame, hence the atomic.
    static std::atomic<uint32_t> stereoAutoConvergenceTenths{0};

    // Centre-of-screen view depth in game units, as float bits, or 0 when there
    // is nothing valid. Feeds the depth-aware crosshair.
    static std::atomic<uint32_t> stereoCenterViewZBits{0};

    static void stereoStoreCenterViewZ(float viewZ) {
        uint32_t bits = 0;
        if (viewZ > 0.0f) {
            std::memcpy(&bits, &viewZ, sizeof(bits));
        }
        stereoCenterViewZBits.store(bits, std::memory_order_relaxed);
    }

    static float stereoLoadCenterViewZ() {
        const uint32_t bits = stereoCenterViewZBits.load(std::memory_order_relaxed);
        if (bits == 0) {
            return -1.0f;
        }
        float viewZ = 0.0f;
        std::memcpy(&viewZ, &bits, sizeof(viewZ));
        return viewZ;
    }

    // The aim depth driving the crosshair. Logged on meaningful change so that
    // "the reticle sits too deep" can be checked against a number rather than
    // guessed at - aiming at something a known distance away should report a
    // depth in the same ballpark as the near depth the convergence loop sees.
    static void logStereoAimDepth(float aimZ) {
        static float lastLogged = -1.0f;
        if (aimZ <= 0.0f) {
            return;
        }
        if ((lastLogged > 0.0f) && (std::fabs(aimZ - lastLogged) < (lastLogged * 0.10f))) {
            return;
        }
        lastLogged = aimZ;
        stereoDepthLogLine("aim: z=%.1f", aimZ);
    }

    static void logStereoAutoConvergence(float nearestZ, uint32_t manualTenths, uint32_t appliedTenths) {
        static uint32_t lastLogged = UINT32_MAX;
        // Only on a real change, so a steady scene stays quiet.
        if (appliedTenths == lastLogged) {
            return;
        }
        lastLogged = appliedTenths;
        stereoDepthLogLine("autoconv: nearestZ=%.1f manual=%.1f applied=%.1f",
            nearestZ, manualTenths * 0.1f, appliedTenths * 0.1f);
    }

    // Reports the first few depth targets offered for sampling and whether the
    // size gate accepted them. Without this, "no output at all" is ambiguous
    // between the gate rejecting everything and the readback returning nothing.
    static void logStereoDepthCandidate(uint32_t depthW, uint32_t depthH,
                                        uint32_t colorW, uint32_t colorH, bool accepted) {
        static uint32_t reported = 0;
        if (reported >= 8) {
            return;
        }
        reported++;
        stereoDepthLogLine("(candidate depth %ux%u vs colour %ux%u -> %s)",
            depthW, depthH, colorW, colorH, accepted ? "sampled" : "skipped");
    }

    static void logStereoDepthSample(float deviceDepth) {
        // A periodic summary rather than a per-frame heartbeat. The first
        // version logged on every invalid frame, and because valid and invalid
        // samples alternated it consumed the whole line budget in seconds and
        // silenced the log - which read as "no depth data" while sampling was
        // in fact working. A summary keeps "sampling is running" distinguishable
        // from "sampling is disabled" without drowning the useful lines.
        static uint32_t sampled = 0;
        static uint32_t validCount = 0;
        static float lastValid = -1.0f;

        sampled++;
        if (deviceDepth > 0.0f) {
            validCount++;
            lastValid = deviceDepth;
        }

        if ((sampled % 600) == 0) {
            stereoDepthLogLine("(status: %u sampled, %u valid, last=%.6f)", sampled, validCount, lastValid);
        }

        if (deviceDepth <= 0.0f) {
            return;
        }

        // Only log on a meaningful change, so walking around produces a
        // readable trace instead of 60 identical lines a second.
        static float lastLogged = -1.0f;
        if ((lastLogged > 0.0f) && (std::fabs(deviceDepth - lastLogged) < (lastLogged * 0.02f))) {
            return;
        }
        lastLogged = deviceDepth;

        // Linear view-space distance, assuming DK64's near/far (10 / 1500 read
        // from global_asm .data) and a standard [0,1] depth range.
        //
        // An earlier version of this printed two columns, labelled as the "GL"
        // and "D3D" depth conventions, on the theory that comparing them against
        // a known distance would reveal which one the backend uses. They are the
        // same function:
        //     2nf / ((f+n) - (2d-1)(f-n))
        // has denominator 2(f - d(f-n)), so it reduces to nf / (f - d(f-n)).
        // The two columns agreed in every sample because they could not disagree.
        // Distinguishing a standard from a reversed depth range needs a
        // known-distance observation, not a second algebraic form.
        constexpr float nearZ = 10.0f;
        constexpr float farZ = 1500.0f;
        const float denom = farZ - (deviceDepth * (farZ - nearZ));
        const float viewZ = (std::fabs(denom) > 1e-6f) ? ((nearZ * farZ) / denom) : -1.0f;

        stereoDepthLogLine("device=%.6f  viewZ=%.1f", deviceDepth, viewZ);
    }

    // WorkloadQueue

    WorkloadQueue::WorkloadQueue() {
        reset();
    }

    WorkloadQueue::~WorkloadQueue() {
        threadsRunning = false;
        cursorCondition.notify_all();
        idleCondition.notify_all();

        if (renderThread != nullptr) {
            renderThread->join();
            delete renderThread;
        }

        if (idleThread != nullptr) {
            idleThread->join();
            delete idleThread;
        }

        workloadIdCondition.notify_all();
    }

    void WorkloadQueue::reset() {
        for (Workload &w : workloads) {
            w.reset();
        }

        threadCursor = 0;
        writeCursor = 0;
        barrierCursor = int(workloads.size()) - 1;
        workloadId = 0;
        lastPresentId = 0;
    }

    void WorkloadQueue::advanceToNextWorkload() {
        int nextWriteCursor = (writeCursor + 1) % workloads.size();

        // Stall the thread until the barrier is lifted if we're trying to write on a workload being used by the GPU.
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

    void WorkloadQueue::repeatLastWorkload() {
        {
            const std::scoped_lock lock(cursorMutex);
            threadCursor = previousWriteCursor();
        }

        cursorCondition.notify_all();
    }

    uint32_t WorkloadQueue::previousWriteCursor() const {
        if (writeCursor > 0) {
            return writeCursor - 1;
        }
        else {
            return uint32_t(workloads.size()) - 1;
        }
    }

    void WorkloadQueue::waitForIdle() {
        std::unique_lock<std::mutex> threadLock(threadMutex);
    }

    void WorkloadQueue::waitForWorkloadId(uint64_t waitId) {
        std::unique_lock<std::mutex> workloadLock(workloadIdMutex);
        workloadIdCondition.wait(workloadLock, [&]() {
            return (waitId <= workloadId) || !threadsRunning;
        });
    }

    void WorkloadQueue::setup(const External &ext) {
        this->ext = ext;

        rspProcessor = std::make_unique<RSPProcessor>(ext.device);
        vertexProcessor = std::make_unique<VertexProcessor>(ext.device);
        framebufferRenderer = std::make_unique<FramebufferRenderer>(ext.workloadGraphicsWorker, true, ext.createdGraphicsAPI, ext.shaderLibrary);
        renderFramebufferManager = std::make_unique<RenderFramebufferManager>(ext.device);
        queryPool = ext.device->createQueryPool(2);

        projectionProcessor.setup(ext.workloadGraphicsWorker);
        transformProcessor.setup(ext.workloadGraphicsWorker);
        tileProcessor.setup(ext.workloadGraphicsWorker);
        lookAtProcessor.setup(ext.workloadGraphicsWorker);

        threadsRunning = true;
        renderThread = new std::thread(&WorkloadQueue::renderThreadLoop, this);
        idleThread = new std::thread(&WorkloadQueue::idleThreadLoop, this);
    }

    void WorkloadQueue::updateMultisampling() {
        renderFramebufferManager->destroyAll();
        dummyDepthTarget.reset();
        framebufferRenderer->updateMultisampling();
    }

    void WorkloadQueue::threadConfigurationUpdate(hlslpp::uint2 viFbSize, WorkloadConfiguration &workloadConfig) {
        const std::scoped_lock lock(ext.sharedResources->configurationMutex);
        const bool sizeChanged = ext.sharedResources->swapChainSizeChanged;
        ext.sharedResources->swapChainSizeChanged = false;
        
        // Retrieve the reference height to be used for determining the resolution scale. Impose a minimum in case
        // the game is using too small of a portion of the VI.
        const uint32_t MinimumReferenceHeight = 60;
        const uint32_t referenceHeight = (viFbSize[1] > 0) ? std::max(viFbSize[1], MinimumReferenceHeight) : 240;

        // Compute the aspect ratio to be used for the frame.
        workloadConfig.aspectRatioSource = (viFbSize[1] > 0) ? float(viFbSize[0]) / float(viFbSize[1]) : (4.0f / 3.0f);

        const auto ratioMode = ext.sharedResources->userConfig.aspectRatio;
        switch (ratioMode) {
        case UserConfiguration::AspectRatio::Expand:
            if ((ext.sharedResources->swapChainWidth > 0) && (ext.sharedResources->swapChainHeight > 0)) {
                const float derivedRatioTarget = float(ext.sharedResources->swapChainWidth) / float(ext.sharedResources->swapChainHeight);
                workloadConfig.aspectRatioTarget = std::max(derivedRatioTarget, workloadConfig.aspectRatioSource);
            }
            else {
                workloadConfig.aspectRatioTarget = workloadConfig.aspectRatioSource;
            }

            break;
        case UserConfiguration::AspectRatio::Manual:
            workloadConfig.aspectRatioTarget = float(ext.sharedResources->userConfig.aspectTarget);
            break;
        case UserConfiguration::AspectRatio::Original:
        default:
            workloadConfig.aspectRatioTarget = workloadConfig.aspectRatioSource;
            break;
        }

        // Compute the extended GBI aspect ratio percentage to be used for the frame.
        const auto extRatioMode = ext.sharedResources->userConfig.extAspectRatio;
        switch (extRatioMode) {
        case UserConfiguration::AspectRatio::Expand:
            workloadConfig.extAspectPercentage = 1.0f;
            break;
        case UserConfiguration::AspectRatio::Manual:
            if ((ext.sharedResources->swapChainWidth > 0) && (ext.sharedResources->swapChainHeight > 0)) {
                const float reducedExtTarget = float(ext.sharedResources->userConfig.extAspectTarget) - workloadConfig.aspectRatioSource;
                const float reducedDisplayTarget = workloadConfig.aspectRatioTarget - workloadConfig.aspectRatioSource;
                if ((reducedExtTarget > 0.0f) && (reducedDisplayTarget > 0.0f)) {
                    workloadConfig.extAspectPercentage = std::clamp((reducedExtTarget / reducedDisplayTarget), 0.0f, 1.0f);
                }
                else {
                    workloadConfig.extAspectPercentage = 0.0f;
                }
            }
            else {
                workloadConfig.extAspectPercentage = 0.0f;
            }

            break;
        case UserConfiguration::AspectRatio::Original:
        default:
            workloadConfig.extAspectPercentage = 0.0f;
            break;
        }

        // Compute the resolution scaling to be used for the frame.
        float resolutionMultiplier;
        const auto resolutionMode = ext.sharedResources->userConfig.resolution;
        switch (resolutionMode) {
        case UserConfiguration::Resolution::WindowIntegerScale:
            if (ext.sharedResources->swapChainHeight > 0) {
                resolutionMultiplier = std::max(float((ext.sharedResources->swapChainHeight + referenceHeight - 1) / referenceHeight), 1.0f);
            }
            else {
                resolutionMultiplier = 1.0f;
            }

            break;
        case UserConfiguration::Resolution::Manual:
            resolutionMultiplier = float(ext.sharedResources->userConfig.resolutionMultiplier);
            break;
        case UserConfiguration::Resolution::Original:
        default:
            resolutionMultiplier = 1.0f;
            break;
        }

        uint32_t msaaSampleCount = ext.sharedResources->userConfig.msaaSampleCount();

        // Build the resolution scale vector from the configuration.
        workloadConfig.aspectRatioScale = workloadConfig.aspectRatioTarget / workloadConfig.aspectRatioSource;
        workloadConfig.resolutionScale = { resolutionMultiplier * workloadConfig.aspectRatioScale, resolutionMultiplier };
        workloadConfig.downsampleMultiplier = ext.sharedResources->userConfig.downsampleMultiplier;
        ext.sharedResources->resolutionScale = workloadConfig.resolutionScale;

        // Find the target refresh rate from the configuration.
        const auto refreshRate = ext.sharedResources->userConfig.refreshRate;
        switch (refreshRate) {
        case UserConfiguration::RefreshRate::Display:
            workloadConfig.targetRate = ext.sharedResources->swapChainRate;
            break;
        case UserConfiguration::RefreshRate::Manual:
            workloadConfig.targetRate = ext.sharedResources->userConfig.refreshRateTarget;

            // Limit the target rate to the rate detected by the swap chain.
            if ((ext.sharedResources->swapChainRate > 0) && (workloadConfig.targetRate > ext.sharedResources->swapChainRate)) {
                workloadConfig.targetRate = ext.sharedResources->swapChainRate;
            }

            break;
        case UserConfiguration::RefreshRate::Original:
        default:
            workloadConfig.targetRate = 0;
            break;
        }

        // Store the rate that was chosen for the configuration.
        ext.sharedResources->targetRate = workloadConfig.targetRate;

#   if RT_ENABLED
        workloadConfig.raytracingEnabled = rtEnabled;

        if (workloadConfig.raytracingEnabled && (ext.sharedResources->rtConfigChanged || ext.sharedResources->fbConfigChanged || sizeChanged)) {
            // Only load the RT pipeline if the device supports it.
             if (ext.device->getCapabilities().raytracing && !ext.rtShaderCache->isSetup()) {
                 ext.rtShaderCache->setup();
            }

            framebufferRenderer->setRaytracingConfig(ext.sharedResources->rtConfig, ext.sharedResources->fbConfigChanged || sizeChanged);
            ext.sharedResources->rtConfigChanged = false;
        }
#   endif
        
        workloadConfig.postBlendNoise = ext.sharedResources->emulatorConfig.dither.postBlendNoise;
        workloadConfig.postBlendNoiseNegative = ext.sharedResources->emulatorConfig.dither.postBlendNoiseNegative;
        
        if (ext.sharedResources->fbConfigChanged || sizeChanged) {
            {
                // Wait until the other queue has stopped using the interpolated color targets.
                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                InterpolatedFrameCounters &curFrameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
                ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                    return curFrameCounters.presented >= curFrameCounters.available;
                });
            }

            std::scoped_lock<std::mutex> managerLock(ext.sharedResources->managerMutex);
            FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
            RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
            renderFramebufferManager->destroyAll();
            targetManager.destroyAll();
            fbManager.destroyAllTileCopies();
            ext.sharedResources->fbConfigChanged = false;
            ext.sharedResources->interpolatedColorTargets.clear();
        }

        if (ext.sharedResources->userConfigChanged) {
            idleMutex.lock();
            idleActive = ext.sharedResources->userConfig.idleWorkActive;
            idleMutex.unlock();
            idleCondition.notify_all();
        }
    }

    void WorkloadQueue::threadConfigurationValidate() {
        const std::scoped_lock lock(ext.sharedResources->configurationMutex);
        if (ext.sharedResources->userConfigChanged) {
            ext.sharedResources->newConfigValidated = true;
            ext.sharedResources->userConfigChanged = false;
        }
    }
    
    void WorkloadQueue::threadRenderFrame(GameFrame &curFrame, const GameFrame &prevFrame, const WorkloadConfiguration &workloadConfig,
        const DebuggerRenderer &debuggerRenderer, const DebuggerCamera &debuggerCamera, float curFrameWeight, float prevFrameWeight,
        float deltaTimeMs, RenderTargetKey overrideTargetKey, int32_t overrideTargetFbPairIndex, RenderTarget *overrideTarget,
        uint32_t overrideTargetModifier, bool uploadVelocity, bool uploadExtras, bool interpolateTiles, bool interpolateLookAts,
        StereoEye stereoEye)
    {
#   if ENABLE_HIGH_RESOLUTION_RENDERER
        std::scoped_lock<std::mutex> managerLock(ext.sharedResources->workloadMutex);
        FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
        RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
        const bool usingMSAA = (targetManager.multisampling.sampleCount > 1);

        rendererCPUProfiler.start();

        const bool aspectRatioAdjustment = (abs(workloadConfig.aspectRatioScale - 1.0f) > 1e-6f);
        const auto stereoMode = ext.sharedResources->userConfig.stereoMode;
        const bool stereoActive = (stereoMode != UserConfiguration::StereoMode::Off);
        const bool processProjections = aspectRatioAdjustment || prevFrame.matched || curFrame.isDebuggerCameraEnabled(*this) || stereoActive;
        bool uploadProjections = false;
        if (processProjections) {
            ProjectionProcessor::ProcessParams projParams;
            projParams.worker = ext.workloadGraphicsWorker;
            projParams.workloadQueue = this;
            projParams.curFrame = &curFrame;
            projParams.prevFrame = &prevFrame;
            projParams.curFrameWeight = curFrameWeight;
            projParams.prevFrameWeight = prevFrameWeight;
            projParams.aspectRatioScale = workloadConfig.aspectRatioScale;
            projParams.stereoMode = stereoMode;
            projParams.stereoSeparation = ext.sharedResources->userConfig.stereoSeparation;
            projParams.stereoConvergence = ext.sharedResources->userConfig.stereoConvergence;

            // Depth-driven auto-convergence overrides the manual value when it
            // has something to say. It only ever pulls convergence IN - the
            // manual slider stays the ceiling - so this cannot push the screen
            // plane further out than the user asked for.
            if (ext.sharedResources->userConfig.stereoAutoConvergence != 0) {
                const uint32_t autoTenths = stereoAutoConvergenceTenths.load(std::memory_order_relaxed);
                if (autoTenths > 0) {
                    projParams.stereoConvergence = std::min(autoTenths, projParams.stereoConvergence);
                }
            }
            projParams.stereoHudDepth = ext.sharedResources->userConfig.stereoHudDepth;
            // Caller selects which eye this pass renders. The two-pass driver in
            // renderThreadLoop runs this function twice with Left then Right when
            // stereoMode != Off and snapshots each pass's output target.
            projParams.stereoEye = stereoActive ? stereoEye : StereoEye::None;
            projectionProcessor.process(projParams);
            projectionProcessor.upload(projParams);
            uploadProjections = true;
        }

        const bool processTransforms = prevFrame.matched;
        bool uploadTransforms = false;
        if (processTransforms) {
            TransformProcessor::ProcessParams transformParams;
            transformParams.worker = ext.workloadGraphicsWorker;
            transformParams.workloadQueue = this;
            transformParams.curFrame = &curFrame;
            transformParams.prevFrame = &prevFrame;
            transformParams.curFrameWeight = curFrameWeight;
            transformParams.prevFrameWeight = prevFrameWeight;
            transformProcessor.process(transformParams);
            transformProcessor.upload(transformParams);
            uploadTransforms = true;
        }

        bool uploadTiles = false;
        if (interpolateTiles) {
            TileProcessor::ProcessParams tileParams;
            tileParams.worker = ext.workloadGraphicsWorker;
            tileParams.workloadQueue = this;
            tileParams.curFrame = &curFrame;
            tileParams.prevFrame = &prevFrame;
            tileParams.curFrameWeight = curFrameWeight;
            tileParams.prevFrameWeight = prevFrameWeight;
            tileProcessor.process(tileParams);
            tileProcessor.upload(tileParams);
            uploadTiles = true;
        }

        bool uploadLookAts = false;
        if (interpolateLookAts) {
            LookAtProcessor::ProcessParams lookAtParams;
            lookAtParams.worker = ext.workloadGraphicsWorker;
            lookAtParams.workloadQueue = this;
            lookAtParams.curFrame = &curFrame;
            lookAtParams.prevFrame = &prevFrame;
            lookAtParams.curFrameWeight = curFrameWeight;
            lookAtParams.prevFrameWeight = prevFrameWeight;
            lookAtProcessor.process(lookAtParams);
            lookAtProcessor.upload(lookAtParams);
            uploadLookAts = true;
        }

        // Reset the max height tracking for all active framebuffers.
        fbManager.resetTracking();

        if ((overrideTarget != nullptr) && !usingMSAA) {
            targetManager.setOverride(overrideTargetKey, overrideTarget);
        }

        for (uint32_t w = 0; w < curFrame.workloads.size(); w++) {
            Workload &workload = workloads[curFrame.workloads[w]];

            // There's no guarantee the RSP was processed if framebuffers were not rendered.
            const bool processRSP = true;
            if (processRSP) {
                workload.resetRSPOutputBuffers();

                RSPProcessor::ProcessParams rspParams;
                rspParams.worker = ext.workloadGraphicsWorker;
                rspParams.drawData = &workload.drawData;
                rspParams.drawBuffers = &workload.drawBuffers;
                rspParams.outputBuffers = &workload.outputBuffers;
                rspParams.prevFrameWeight = prevFrameWeight;
                rspParams.curFrameWeight = curFrameWeight;
                rspProcessor->process(rspParams);
            }

            const bool processWorldVertices = prevFrame.matched;
            if (processWorldVertices) {
                workload.resetWorldOutputBuffers();

                VertexProcessor::ProcessParams vertexParams;
                vertexParams.worker = ext.workloadGraphicsWorker;
                vertexParams.drawData = &workload.drawData;
                vertexParams.drawBuffers = &workload.drawBuffers;
                vertexParams.outputBuffers = &workload.outputBuffers;
                vertexParams.curFrameWeight = curFrameWeight;
                vertexParams.prevFrameWeight = prevFrameWeight;
                vertexProcessor->process(vertexParams);
            }

            hlslpp::float2 fixedResScale;
            Framebuffer *colorFb;
            Framebuffer *depthFb;
            uint32_t nativeColorWidth;
            uint32_t nativeColorHeight;
            uint32_t targetWidth;
            uint32_t targetHeight;
            uint32_t targetMisalignX;
            uint32_t rtWidth;
            uint32_t rtHeight;
            RenderTarget *colorTarget;
            RenderTarget *depthTarget;
            RenderFramebufferKey fbKey;
            auto getTargetsFromPair = [&](uint32_t f) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
                const auto &colorImg = fbPair.colorImage;
                const auto &depthImg = fbPair.depthImage;
                fixedResScale = workloadConfig.resolutionScale;
                if (!fbPair.drawColorRect.isEmpty()) {
                    colorFb = nullptr;
                    depthFb = nullptr;
                    nativeColorWidth = colorImg.width;
                    nativeColorHeight = fbPair.drawColorRect.bottom(true);

                    // When the target is much bigger than the reference height, we reduce the resolution scaling (but clamped to 1.0).
                    const uint32_t heightThreshold = (workload.viFbSize[1] > 0) ? ((workload.viFbSize[1] * 3) / 2) : 360;
                    uint32_t downsampleMultiplier = workloadConfig.downsampleMultiplier;
                    if ((nativeColorHeight >= heightThreshold) && (fixedResScale[1] >= 2.0f)) {
                        fixedResScale = hlslpp::max(fixedResScale / 2.0f, hlslpp::float2(1.0f, 1.0f));
                        downsampleMultiplier = std::max(downsampleMultiplier / 2U, 1U);
                    }

                    if (fbPair.depthRead || fbPair.depthWrite || fbPair.fastPaths.clearDepthOnly) {
                        uint32_t depthAddress = fbPair.fastPaths.clearDepthOnly ? colorImg.address : depthImg.address;
                        depthFb = &fbManager.get(depthAddress, G_IM_SIZ_16b, nativeColorWidth, nativeColorHeight);
                        depthFb->everUsedAsDepth = true;
                    }
                    else {
                        depthFb = nullptr;
                    }

                    // Ensure dimensions are the same for the color and depth targets based on their previous sizes.
                    fbKey = RenderFramebufferKey();

                    if (!fbPair.fastPaths.clearDepthOnly) {
                        colorFb = &fbManager.get(colorImg.address, colorImg.siz, nativeColorWidth, nativeColorHeight);
                    }

                    if (colorFb != nullptr) {
                        fbKey.colorTargetKey = RenderTargetKey(colorFb->addressStart, colorFb->width, colorFb->siz, Framebuffer::Type::Color);
                        colorTarget = &targetManager.get(fbKey.colorTargetKey);
                    }
                    else {
                        colorTarget = nullptr;
                    }

                    // Apply the modifier key if we retrieved the override target.
                    if ((colorTarget != nullptr) && (colorTarget == overrideTarget)) {
                        fbKey.modifierKey = overrideTargetModifier;
                    }

                    fixedResScale = RenderTarget::computeFixedResolutionScale(colorImg.width, fixedResScale);
                    RenderTarget::computeScaledSize(nativeColorWidth, nativeColorHeight, fixedResScale, targetWidth, targetHeight, targetMisalignX);

                    rtWidth = targetWidth;
                    rtHeight = targetHeight;

                    // The desired size should not be less than the existing size of the color and depth targets.
                    RenderTarget *chosenRt = nullptr;
                    if (depthFb != nullptr) {
                        fbKey.depthTargetKey = RenderTargetKey(depthFb->addressStart, depthFb->width, depthFb->siz, Framebuffer::Type::Depth);
                        depthTarget = &targetManager.get(fbKey.depthTargetKey);
                        depthTarget->resolutionScale = fixedResScale;
                        rtWidth = std::max(rtWidth, depthTarget->width);
                        rtHeight = std::max(rtHeight, depthTarget->height);
                        chosenRt = depthTarget;
                    }
                    else {
                        depthTarget = nullptr;
                    }

                    if (colorTarget != nullptr) {
                        rtWidth = std::max(rtWidth, colorTarget->width);
                        rtHeight = std::max(rtHeight, colorTarget->height);
                        chosenRt = colorTarget;
                    }

                    assert(chosenRt != nullptr);
                    chosenRt->resolutionScale = fixedResScale;
                    chosenRt->downsampleMultiplier = downsampleMultiplier;
                    chosenRt->misalignX = targetMisalignX;
                    chosenRt->invMisalignX = (targetMisalignX > 0) ? (std::lround(fixedResScale.y) - targetMisalignX) : 0;

                    assert((colorTarget != nullptr) || (depthTarget != nullptr));
                    return true;
                }
                else {
                    return false;
                }
            };

            thread_local std::unordered_set<RenderTarget *> resizedTargets;
            thread_local std::vector<std::pair<RenderTarget *, RenderTarget *>> colorDepthPairs;
            resizedTargets.clear();
            colorDepthPairs.clear();

            const uint32_t fbPairCount = (debuggerRenderer.framebufferIndex >= 0) ? (debuggerRenderer.framebufferIndex + 1) : workload.fbPairCount;
            for (uint32_t f = 0; f < fbPairCount; f++) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
#           if RT_ENABLED
                for (uint32_t p = 0; p < fbPair.projectionCount; p++) {
                    const Projection &proj = fbPair.projections[p];
                    const bool perspProj = (proj.type == Projection::Type::Perspective);
                    const bool rtProj = (perspProj && workloadConfig.raytracingEnabled && fbPair.depthWrite); // TODO: Move this condition out of here, ideally by moving the shader submission elsewhere.
                    if (!rtProj) {
                        continue;
                    }

                    // Submit RT shaders if it's an RT proj.
                    for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                        const GameCall &call = proj.gameCalls[d];
                        ext.rtShaderCache->submit(call.shaderDesc);
                    }
                }
#           endif

                // Resize the render targets for this framebuffer pair if necessary.
                if (getTargetsFromPair(f)) {
                    // Resize the native target buffers.
                    if (colorFb != nullptr) {
                        colorFb->nativeTarget.resetBufferHistory();
                    }

                    if (depthFb != nullptr) {
                        depthFb->nativeTarget.resetBufferHistory();
                    }

                    if ((colorTarget != nullptr) && colorTarget->resize(ext.workloadGraphicsWorker, rtWidth, rtHeight)) {
                        resizedTargets.emplace(colorTarget);
                        colorFb->readHeight = 0;
                    }

                    // Set up the dummy target used for rendering the depth if no depth framebuffer is active.
                    if (depthFb == nullptr) {
                        if (dummyDepthTarget == nullptr) {
                            dummyDepthTarget = std::make_unique<RenderTarget>(0, Framebuffer::Type::Depth, targetManager.multisampling, targetManager.usesHDR);
                            dummyDepthTarget->setupDepth(ext.workloadGraphicsWorker, rtWidth, rtHeight);
                        }

                        if ((dummyDepthTarget != nullptr) && dummyDepthTarget->resize(ext.workloadGraphicsWorker, rtWidth, rtHeight)) {
                            resizedTargets.emplace(dummyDepthTarget.get());
                        }

                        if (colorTarget != nullptr) {
                            colorDepthPairs.emplace_back(colorTarget, dummyDepthTarget.get());
                        }
                    }
                    else if (depthTarget != nullptr) {
                        if (colorTarget != nullptr) {
                            colorDepthPairs.emplace_back(colorTarget, depthTarget);
                        }

                        if (depthTarget->resize(ext.workloadGraphicsWorker, rtWidth, rtHeight)) {
                            resizedTargets.emplace(depthTarget);
                            depthFb->readHeight = 0;
                        }
                    }
                }

                fbManager.setupOperations(ext.workloadGraphicsWorker, fbPair.startFbOperations, fixedResScale, targetManager, &resizedTargets);
                fbManager.setupOperations(ext.workloadGraphicsWorker, fbPair.endFbOperations, fixedResScale, targetManager, &resizedTargets);
            }

            // Make sure all depth targets are at least bigger than their corresponding color targets.
            for (auto colorDepthPair : colorDepthPairs) {
                if (colorDepthPair.second->resize(ext.workloadGraphicsWorker, colorDepthPair.first->width, colorDepthPair.first->height)) {
                    resizedTargets.emplace(colorDepthPair.second);
                }
            }

            for (RenderTarget *renderTarget : resizedTargets) {
                renderFramebufferManager->destroyAllWithRenderTarget(renderTarget);
            }

            uint32_t gameCallCursor = 0;
            const uint32_t gameCallCountMax = (debuggerRenderer.globalDrawCallIndex >= 0) ? (debuggerRenderer.globalDrawCallIndex + 1) : workload.gameCallCount;
            thread_local std::vector<BufferUploader *> bufferUploaders;
            bufferUploaders.clear();

            // Indicate to the texture cache the textures must not be deleted.
            ext.textureCache->incrementLock();

            // Reset the texture cache vectors for the framebuffer renderer.
            framebufferRenderer->updateTextureCache(ext.textureCache);

            for (uint32_t f = 0; f < fbPairCount; f++) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
                fbManager.performDiscards(fbPair.startFbDiscards);
            }
            
            // Add all framebuffer pairs to the framebuffer renderer and setup the operations.
            scratchFbChangePool.reset();
            fbManager.resetOperations();
            framebufferRenderer->resetFramebuffers(ext.workloadGraphicsWorker, ubershadersVisible, workload.extended.ditherNoiseStrength, targetManager.multisampling);

#       if RT_ENABLED
            if (workloadConfig.raytracingEnabled) {
                framebufferRenderer->resetRaytracing(ext.rtShaderCache, ext.blueNoiseTexture);
            }
#       endif

            for (uint32_t f = 0; f < fbPairCount; f++) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
                if (getTargetsFromPair(f)) {
                    RenderFramebufferStorage &fbStorage = renderFramebufferManager->get(fbKey, colorTarget, (depthTarget != nullptr) ? depthTarget : dummyDepthTarget.get());
                    FramebufferRenderer::DrawParams drawParams;
                    drawParams.worker = ext.workloadGraphicsWorker;
                    drawParams.fbStorage = &fbStorage;
                    drawParams.curWorkload = &workload;
                    drawParams.fbPairIndex = f;
                    drawParams.fbWidth = nativeColorWidth;
                    drawParams.fbHeight = nativeColorHeight;
                    drawParams.targetWidth = targetWidth;
                    drawParams.targetHeight = targetHeight;
                    drawParams.rasterShaderCache = ext.rasterShaderCache;
                    drawParams.resolutionScale = fixedResScale;
                    drawParams.aspectRatioSource = workloadConfig.aspectRatioSource;
                    drawParams.aspectRatioTarget = workloadConfig.aspectRatioTarget;
                    drawParams.extAspectPercentage = workloadConfig.extAspectPercentage;
                    drawParams.horizontalMisalignment = (colorTarget != nullptr) ? float(colorTarget->misalignX) : float(depthTarget->misalignX);
                    drawParams.presetScene = curFrame.presetScene;
                    drawParams.rtEnabled = workloadConfig.raytracingEnabled;
                    drawParams.submissionFrame = workload.submissionFrame;
                    drawParams.deltaTimeMs = deltaTimeMs;
                    drawParams.ubershadersOnly = ubershadersOnly;
                    drawParams.postBlendNoise = workloadConfig.postBlendNoise;
                    drawParams.postBlendNoiseNegative = workloadConfig.postBlendNoiseNegative;
                    drawParams.maxGameCall = std::min(gameCallCountMax - gameCallCursor, fbPair.gameCallCount);
                    // Compute the per-eye NDC.x offset for texture rectangles
                    // so HUD text / dialog text / score icons / item prints
                    // shift consistently with the rest of the HUD when stereo
                    // is on. Sign and magnitude match applyStereoHudShift's
                    // orthographic path so screen-rect UI lands at the same
                    // visible depth as ortho-projected UI.
                    drawParams.stereoRectOffsetX = 0.0f;
                    if ((stereoMode != UserConfiguration::StereoMode::Off) &&
                        (stereoEye != StereoEye::None)) {
                        const auto hudDepth = ext.sharedResources->userConfig.stereoHudDepth;
                        if (hudDepth != 50) {
                            const float centered = (static_cast<float>(hudDepth) - 50.0f) / 50.0f;
                            constexpr float maxHudOffset = 0.04f;
                            const float hudOffset = -centered * maxHudOffset;
                            constexpr float perspectiveToOrthoScale = 2.75f;
                            const float eyeSign = (stereoEye == StereoEye::Left) ? +1.0f : -1.0f;
                            // Negation mirrors applyStereoHudShift's orthographic
                            // branch so rectangles shift in the same direction
                            // as everything else.
                            drawParams.stereoRectOffsetX = -eyeSign * hudOffset * perspectiveToOrthoScale;
                        }

                        // Depth-aware crosshair (dynamic3d 5.1). The reticle is
                        // orthographic, so it normally rides the HUD depth shift
                        // applied to that projection. Placing it at the aimed
                        // depth means applying the disparity for that depth and
                        // cancelling the HUD shift it would otherwise have got.
                        const float centerZ = stereoLoadCenterViewZ();
                        const auto &ccfg = ext.sharedResources->userConfig;
                        {
                            const float separation = float(ccfg.stereoSeparation) * (0.10f / 50.0f);
                            uint32_t convTenths = ccfg.stereoConvergence;
                            if (ccfg.stereoAutoConvergence != 0) {
                                const uint32_t autoTenths = stereoAutoConvergenceTenths.load(std::memory_order_relaxed);
                                if (autoTenths > 0) {
                                    convTenths = std::min(autoTenths, convTenths);
                                }
                            }
                            const float convergence = float(convTenths) * 2.0f;
                            const float eyeSign = (stereoEye == StereoEye::Left) ? +1.0f : -1.0f;

                            // dynamic3d 1.2 in NDC: zero at the convergence
                            // distance, tending to the full separation offset as
                            // the aim point recedes.
                            //
                            // With no usable depth the reticle recedes to
                            // INFINITY rather than falling back to the screen
                            // plane. Aiming at open sky should put it far away,
                            // and a reticle that snaps forward to the glass
                            // whenever the sample drops out is far more jarring
                            // than one that sits deep. This matches what the
                            // Perfect Dark port does when no aim target resolves.
                            const float depthRatio = (centerZ > 0.0f) ? (convergence / centerZ) : 0.0f;
                            const float aimNdc = -eyeSign * separation * (1.0f - depthRatio);

                            // What the orthographic HUD shift already applies, so
                            // it can be removed rather than compounded.
                            float hudNdc = 0.0f;
                            const auto hudDepth = ccfg.stereoHudDepth;
                            if (hudDepth != 50) {
                                const float centered = (static_cast<float>(hudDepth) - 50.0f) / 50.0f;
                                constexpr float maxHudOffset = 0.04f;
                                constexpr float perspectiveToOrthoScale = 2.75f;
                                hudNdc = -eyeSign * (-centered * maxHudOffset) * perspectiveToOrthoScale;
                            }

                            drawParams.stereoCrosshairOffsetX = aimNdc - hudNdc;
                            // The geometric test cannot tell the reticle from the
                            // title screen's centred logo, and a depth gate does
                            // not separate them either - that screen runs a 3D
                            // demo behind it, so it has a world pass and a valid
                            // depth like any other.
                            //
                            // What does separate them is game state: the crosshair
                            // only exists in Adventure play, and the game already
                            // reports menu, demo and cutscene scenes for the
                            // convergence loop. Reusing that flag costs nothing
                            // and is a statement about the game rather than a
                            // guess about geometry.
                            drawParams.stereoCrosshairValid = (stereoMode != UserConfiguration::StereoMode::Off) &&
                                (stereoEye != StereoEye::None) && (separation > 0.0f) &&
                                (ccfg.stereoSceneLowConvergence == 0);
                        }
                    }
                    framebufferRenderer->addFramebuffer(drawParams);
                }
                
                gameCallCursor += fbPair.gameCallCount;
            }

            // Create all GPU tile mappings and upload them.
            if (!workload.drawData.gpuTiles.empty()) {
                std::pair<size_t, size_t> gpuTileRange;
                gpuTileRange.first = 0;
                gpuTileRange.second = workload.drawData.gpuTiles.size();
                framebufferRenderer->createGPUTiles(workload.drawData.callTiles.data(), uint32_t(workload.drawData.gpuTiles.size()),
                    workload.drawData.gpuTiles.data(), &fbManager, ext.textureCache, workload.submissionFrame);

                // Upload the GPU tiles.
                ext.workloadTilesUploader->submit(ext.workloadGraphicsWorker, {
                    { workload.drawData.gpuTiles.data(), gpuTileRange, sizeof(interop::GPUTile), RenderBufferFlag::STORAGE, { }, &workload.drawBuffers.gpuTilesBuffer}
                });

                bufferUploaders.emplace_back(ext.workloadTilesUploader);
            }

            if (uploadVelocity) {
                bufferUploaders.emplace_back(ext.workloadVelocityUploader);
                uploadVelocity = false;
            }

            if (uploadExtras) {
                bufferUploaders.emplace_back(ext.workloadExtrasUploader);
                uploadExtras = false;
            }

            if (uploadProjections) {
                bufferUploaders.emplace_back(projectionProcessor.bufferUploader.get());
                uploadProjections = false;
            }

            if (uploadTransforms) {
                bufferUploaders.emplace_back(transformProcessor.bufferUploader.get());
                uploadTransforms = false;
            }

            if (uploadTiles) {
                bufferUploaders.emplace_back(tileProcessor.bufferUploader.get());
                uploadTiles = false;
            }

            if (uploadLookAts) {
                bufferUploaders.emplace_back(lookAtProcessor.bufferUploader.get());
                uploadLookAts = false;
            }

#       if RT_ENABLED
            if (workloadConfig.raytracingEnabled) {
                ext.rtShaderCache->setNextState();
            }
#       endif

            workerMutex.lock();
            ext.workloadGraphicsWorker->commandList->begin();
            ext.workloadGraphicsWorker->commandList->resetQueryPool(queryPool.get(), 0, 2);
            ext.workloadGraphicsWorker->commandList->writeTimestamp(queryPool.get(), 0);
            framebufferRenderer->endFramebuffers(ext.workloadGraphicsWorker, &workload.drawBuffers, &workload.outputBuffers, workloadConfig.raytracingEnabled);
            framebufferRenderer->recordSetup(ext.workloadGraphicsWorker, bufferUploaders, processRSP ? rspProcessor.get() : nullptr, processWorldVertices ? vertexProcessor.get() : nullptr, &workload.outputBuffers, workloadConfig.raytracingEnabled);
            
            // Record all framebuffer pairs.
            uint32_t framebufferIndex = 0;
            for (uint32_t f = 0; f < fbPairCount; f++) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
                bool validTargets = getTargetsFromPair(f);
                fbManager.recordOperations(ext.workloadGraphicsWorker, &workload.fbChangePool, &workload.fbStorage, ext.shaderLibrary, ext.textureCache,
                    fbPair.startFbOperations, targetManager, fixedResScale, f, workload.submissionFrame);

                if (validTargets) {
                    const auto &colorImg = fbPair.colorImage;
                    const auto &depthImg = fbPair.depthImage;
                    bool colorFormatUpdated = false;
                    if (colorFb != nullptr) {
                        if (colorImg.formatChanged) {
                            colorFb->discardLastWrite();
                        }
                        else if (colorFb->isLastWriteDifferent(Framebuffer::Type::Color)) {
                            RenderTargetKey otherColorTargetKey(colorFb->addressStart, colorFb->width, colorFb->siz, colorFb->lastWriteType);
                            RenderTarget &otherColorTarget = targetManager.get(otherColorTargetKey);
                            if (!otherColorTarget.isEmpty()) {
                                const FixedRect &r = colorFb->lastWriteRect;
                                colorTarget->copyFromTarget(ext.workloadGraphicsWorker, &otherColorTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                                colorFb->discardLastWrite();
                                colorFormatUpdated = true;
                            }
                        }

                        if (colorImg.formatChanged) {
                            colorTarget->clearColorTarget(ext.workloadGraphicsWorker);
                            colorFb->readHeight = 0;
                        }

                        if (colorFb->height > colorFb->readHeight) {
                            uint32_t readRowCount = colorFb->height - colorFb->readHeight;
                            FramebufferChange *colorFbChange = colorFb->readChangeFromStorage(ext.workloadGraphicsWorker, workload.fbStorage, scratchFbChangePool,
                                Framebuffer::Type::Color, colorImg.fmt, f, colorFb->readHeight, readRowCount, ext.shaderLibrary);

                            if (colorFbChange != nullptr) {
                                colorTarget->copyFromChanges(ext.workloadGraphicsWorker, *colorFbChange, colorFb->width, readRowCount, colorFb->readHeight, ext.shaderLibrary);
                            }

                            colorFb->readHeight = colorFb->height;
                        }
                    }

                    bool depthFormatUpdated = false;
                    bool depthFbChanged = false;
                    bool depthFbTypeChanged = false;
                    if (depthFb != nullptr) {
                        depthFbTypeChanged = (depthFb->lastWriteType == Framebuffer::Type::Color);

                        bool imgFormatChanged = (colorFb == nullptr) ? colorImg.formatChanged : depthImg.formatChanged;
                        if (imgFormatChanged) {
                            depthFb->discardLastWrite();
                        }
                        else if (depthFb->isLastWriteDifferent(Framebuffer::Type::Depth)) {
                            RenderTargetKey otherDepthTargetKey(depthFb->addressStart, depthFb->width, depthFb->siz, depthFb->lastWriteType);
                            RenderTarget &otherDepthTarget = targetManager.get(otherDepthTargetKey);
                            if (!otherDepthTarget.isEmpty()) {
                                const FixedRect &r = depthFb->lastWriteRect;
                                depthTarget->copyFromTarget(ext.workloadGraphicsWorker, &otherDepthTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                                depthFb->discardLastWrite();
                                depthFormatUpdated = true;
                            }
                        }

                        if (imgFormatChanged) {
                            depthTarget->clearDepthTarget(ext.workloadGraphicsWorker);
                            depthFb->readHeight = 0;
                        }

                        if (depthFb->height > depthFb->readHeight) {
                            uint32_t readRowCount = depthFb->height - depthFb->readHeight;
                            FramebufferChange *depthFbChange = depthFb->readChangeFromStorage(ext.workloadGraphicsWorker, workload.fbStorage, scratchFbChangePool, Framebuffer::Type::Depth,
                                G_IM_FMT_DEPTH, f, depthFb->readHeight, readRowCount, ext.shaderLibrary);

                            if (depthFbChange != nullptr) {
                                depthTarget->copyFromChanges(ext.workloadGraphicsWorker, *depthFbChange, depthFb->width, readRowCount, depthFb->readHeight, ext.shaderLibrary);
                                depthFbChanged = true;
                            }

                            depthFb->readHeight = depthFb->height;
                        }
                    }
                    
                    framebufferRenderer->recordFramebuffer(ext.workloadGraphicsWorker, framebufferIndex++);

                    // Transition the render targets in case the present queue will show them so it doesn't have to perform transitions.
                    if (colorTarget != nullptr && depthTarget != nullptr) {
                        RenderTextureBarrier textureBarriers[] = {
                            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::SHADER_READ),
                            RenderTextureBarrier(depthTarget->texture.get(), RenderTextureLayout::SHADER_READ)
                        };

                        ext.workloadGraphicsWorker->commandList->barriers(RenderBarrierStage::GRAPHICS, textureBarriers, uint32_t(std::size(textureBarriers)));
                    }
                    else {
                        RenderTarget *chosenTarget = (colorTarget != nullptr) ? colorTarget : depthTarget;
                        ext.workloadGraphicsWorker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(chosenTarget->texture.get(), RenderTextureLayout::SHADER_READ));
                    }

                    // Depth sampling for the depth-aware stereo features
                    // (dynamic3d 4.1 / 5.1). Render-thread only, so no locking.
                    //
                    // This loop runs once per framebuffer pair and a frame has
                    // several. Only the world pass carries usable centre depth;
                    // the smaller auxiliary passes are empty there, so sampling
                    // every pair produced a stream that was invalid most of the
                    // time. Require the depth target to match the colour target's
                    // size, and let the sampler take only the first such pass per
                    // frame - reading back only when it actually sampled.
                    if (stereoDepthSamplingEnabled()) {
                        // Auxiliary passes are much smaller than the world pass;
                        // half the colour target's dimensions separates them
                        // without needing an exact match.
                        //
                        // Only the left eye is sampled. In stereo the world is
                        // rendered twice and both passes produce a full-size
                        // depth target, so taking both interleaved two series
                        // that differ by the eye disparity - enough to make
                        // anything driven from this jitter every frame.
                        const bool sampledEye = (stereoEye != StereoEye::Right);
                        const bool mainPass = sampledEye && (depthTarget != nullptr) &&
                            ((colorTarget == nullptr) ||
                             ((depthTarget->width * 2 >= colorTarget->width) &&
                              (depthTarget->height * 2 >= colorTarget->height)));
                        // Logged even when there is no depth target, so "no
                        // output" cannot be confused with "never reached".
                        logStereoDepthCandidate((depthTarget != nullptr) ? depthTarget->width : 0,
                            (depthTarget != nullptr) ? depthTarget->height : 0,
                            (colorTarget != nullptr) ? colorTarget->width : 0,
                            (colorTarget != nullptr) ? colorTarget->height : 0, mainPass);
                        if (mainPass) {
                            static StereoDepthSampler stereoDepthSampler;
                            static StereoAutoConvergence stereoAutoConvergence;
                            // Aim point in depth-target texels, taken from the
                            // frame's scissor rather than from the target's own
                            // dimensions. The target can be padded and carries a
                            // horizontal misalignment of its own, so its midpoint
                            // is not where the reticle is - centring on it put the
                            // sample window off to the right of the crosshair,
                            // reading only its middle-to-right-edge.
                            const FixedRect &aimRect = fbPair.scissorRect;
                            const int32_t aimCenterX = int32_t(
                                ((aimRect.left(false) + (aimRect.width(false, true) / 2)) * fixedResScale[0])) - depthTarget->misalignX;
                            const int32_t aimCenterY = int32_t(
                                ((aimRect.top(false) + (aimRect.height(false, true) / 2)) * fixedResScale[1]));
                            if (stereoDepthSampler.submit(ext.workloadGraphicsWorker, depthTarget, aimCenterX, aimCenterY)) {
                                const StereoDepthSampler::Sample sample = stereoDepthSampler.fetch();
                                logStereoDepthSample(sample.valid ? sample.medianDeviceDepth : -1.0f);

                                // Invert the sampled depth with the projection
                                // the frame was actually rendered with, rather
                                // than assuming DK64's nominal near/far.
                                float projM22 = 0.0f;
                                float projM32 = 0.0f;
                                const auto &cfg = ext.sharedResources->userConfig;
                                if (sample.valid && (cfg.stereoAutoConvergence != 0) &&
                                    stereoGetWorldDepthTerms(projM22, projM32)) {
                                    const float aimZ = stereoDeviceDepthToViewZ(sample.medianDeviceDepth, projM22, projM32);
                                    stereoStoreCenterViewZ(aimZ);
                                    logStereoAimDepth(aimZ);
                                    const float nearestZ = stereoDeviceDepthToViewZ(sample.nearestDeviceDepth, projM22, projM32);
                                    // Ceiling is the user's UNSCALED slider. Using
                                    // the effective value made the ceiling flicker
                                    // as the game's scene classification toggled,
                                    // and the loop chased it instead of the scene.
                                    stereoAutoConvergenceTenths.store(
                                        stereoAutoConvergence.update(nearestZ, cfg.stereoConvergenceManual, cfg.stereoSeparation,
                                            cfg.stereoComfortTarget, cfg.stereoSceneLowConvergence != 0),
                                        std::memory_order_relaxed);
                                    logStereoAutoConvergence(nearestZ, cfg.stereoConvergenceManual,
                                        stereoAutoConvergenceTenths.load(std::memory_order_relaxed));
                                }
                                else if (cfg.stereoAutoConvergence == 0) {
                                    // Snap back rather than easing out, so turning
                                    // the feature off is immediate.
                                    stereoAutoConvergence.reset();
                                    stereoAutoConvergenceTenths.store(0, std::memory_order_relaxed);
                                }
                            }
                        }
                    }

                    // Do the resolve if using MSAA while target override is active and we're on the correct framebuffer pair index.
                    if (usingMSAA && (overrideTarget != nullptr) && ((uint32_t)overrideTargetFbPairIndex == f)) {
                        overrideTarget->resize(ext.workloadGraphicsWorker, colorTarget->width, colorTarget->height);
                        overrideTarget->resolveFromTarget(ext.workloadGraphicsWorker, colorTarget, ext.shaderLibrary);
                    }

                    const uint64_t writeTimestamp = fbManager.nextWriteTimestamp();
                    FixedRect depthFbRect;
                    if (colorFb != nullptr) {
                        colorFb->lastWriteRect.merge(fbPair.drawColorRect.scaled(fixedResScale.x, fixedResScale.y));
                        colorFb->lastWriteType = Framebuffer::Type::Color;
                        colorFb->lastWriteFmt = colorImg.fmt;
                        colorFb->lastWriteTimestamp = writeTimestamp;
                        depthFbRect = fbPair.drawDepthRect;
                    }
                    else {
                        depthFbRect = fbPair.drawColorRect;
                    }
                    
                    const bool depthWrite = ((colorFb == nullptr) || depthFbChanged || depthFbTypeChanged || fbPair.depthWrite) && (depthFb != nullptr);
                    if (depthWrite && !depthFbRect.isNull()) {
                        depthFb->lastWriteRect.merge(depthFbRect.scaled(fixedResScale.x, fixedResScale.y));
                        depthFb->lastWriteType = Framebuffer::Type::Depth;
                        depthFb->lastWriteFmt = G_IM_FMT_DEPTH;
                        depthFb->lastWriteTimestamp = writeTimestamp;
                    }
                }
                
                fbManager.recordOperations(ext.workloadGraphicsWorker, &workload.fbChangePool, &workload.fbStorage, ext.shaderLibrary, ext.textureCache,
                    fbPair.endFbOperations, targetManager, fixedResScale, f, workload.submissionFrame);
            }

            ext.workloadGraphicsWorker->commandList->writeTimestamp(queryPool.get(), 1);
            ext.workloadGraphicsWorker->commandList->end();
            framebufferRenderer->waitForUploaders();
            ext.workloadGraphicsWorker->execute();
            ext.workloadGraphicsWorker->wait();
            workerMutex.unlock();

            // Update the GPU profiler with the results from the timestamps of the frame.
            queryPool->queryResults();
            const uint64_t *frameTimestamps = queryPool->getResults();
            rendererGPUProfiler.log(double(frameTimestamps[1] - frameTimestamps[0]) / 1000000.0);

            // Indicate to the texture cache it's safe to delete the textures if no locks are active.
            ext.textureCache->decrementLock();
        }

        if ((overrideTarget != nullptr) && !usingMSAA) {
            targetManager.removeOverride(overrideTargetKey);
        }

        framebufferRenderer->advanceFrame(workloadConfig.raytracingEnabled);
        rendererCPUProfiler.end();
        rendererCPUProfiler.log();
        rendererCPUProfiler.reset();
#   endif
    }

    void WorkloadQueue::threadAdvanceBarrier() {
        std::scoped_lock<std::mutex> cursorLock(cursorMutex);
        barrierCursor = (barrierCursor + 1) % workloads.size();
    }

    void WorkloadQueue::threadAdvanceWorkloadId(uint64_t newWorkloadId) {
        {
            std::scoped_lock<std::mutex> cursorLock(workloadIdMutex);
            workloadId = newWorkloadId;
        }

        workloadIdCondition.notify_all();
    }

    void WorkloadQueue::renderThreadLoop() {
        Thread::setCurrentThreadName("RT64 Workload");

        WorkloadConfiguration workloadConfig;
        int64_t logicalTicks = 0;
        int64_t displayTicks = 0;
        uint32_t originalRateForTicks = 0;
        uint32_t displayRateForTicks = 0;
        int processCursor = -1;
        bool frameReduction = false;
        while (threadsRunning) {
            {
                std::unique_lock<std::mutex> cursorLock(cursorMutex);
                cursorCondition.wait(cursorLock, [&]() {
                    return (writeCursor != threadCursor) || !threadsRunning;
                });

                if (threadsRunning) {
                    processCursor = threadCursor;
                    threadCursor = (threadCursor + 1) % workloads.size();
                }
            }

            if (processCursor >= 0) {
                std::unique_lock<std::mutex> threadLock(threadMutex);
                Workload &workload = workloads[processCursor];
                ext.presentQueue->waitForPresentId(workload.presentId);

                if (!threadsRunning) {
                    continue;
                }

                ElapsedTimer workloadTimer;
                workloadProfiler.start();
                threadConfigurationUpdate(workload.viFbSize, workloadConfig);

                // FIXME: This is a very hacky way to find out if we need to advance the frame if the workload was paused for the first time.
                if (!workload.paused || (!gameFrames[curFrameIndex].workloads.empty() && (gameFrames[curFrameIndex].workloads[0] != (uint32_t)processCursor))) {
                    prevFrameIndex = curFrameIndex;
                    curFrameIndex = (curFrameIndex + 1) % gameFrames.size();
                }
                
                // TODO: The frame detection needs to be more elaborate than just matching one workload to one frame.
                GameFrame &curFrame = gameFrames[curFrameIndex];
                const GameFrame &prevFrame = gameFrames[prevFrameIndex];
                uint32_t workloadIndex = processCursor;
                curFrame.set(*this, &workloadIndex, 1);

                // Detect the color image to interpolate for this workload.
                RenderTargetKey interpolationTargetKey;
                int32_t interpolationTargetFbPairIndex = -1;
                {
                    std::scoped_lock<std::mutex> managerLock(ext.sharedResources->managerMutex);
                    FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
                    std::vector<uint32_t> &colorVector = ext.sharedResources->colorImageAddressVector;
                    std::unordered_set<uint32_t> &colorSet = ext.sharedResources->colorImageAddressSet;
                    colorVector.clear();
                    colorSet.clear();
                    for (int32_t f = workload.fbPairCount - 1; f >= 0; f--) {
                        const FramebufferPair &fbPair = workload.fbPairs[f];
                        bool interpolationCandidate = fbPair.earlyPresentCandidate();
                        if (fbPair.drawColorRect.isEmpty()) {
                            continue;
                        }

                        const auto &colorImg = fbPair.colorImage;
                        if (colorSet.find(colorImg.address) != colorSet.end()) {
                            continue;
                        }
                        else {
                            if (interpolationCandidate) {
                                colorVector.push_back(colorImg.address);
                            }

                            colorSet.insert(colorImg.address);
                        }

                        if (!interpolationCandidate || !interpolationTargetKey.isEmpty()) {
                            continue;
                        }

                        Framebuffer *interpolationFb = fbManager.find(colorImg.address);
                        if ((interpolationFb != nullptr) && interpolationFb->interpolationEnabled) {
                            interpolationTargetKey.fbType = Framebuffer::Type::Color;
                            interpolationTargetKey.address = fbPair.colorImage.address;
                            interpolationTargetKey.siz = fbPair.colorImage.siz;
                            interpolationTargetKey.width = fbPair.colorImage.width;
                            interpolationTargetFbPairIndex = f;
                        }
                    }
                }

                float prevFrameWeight = 0.0f;
                float curFrameWeight = 1.0f;
                float deltaTimeMs = 1.0f / 30.0f;
                const bool requiresFrameMatching = (workloadConfig.targetRate > 0) || workloadConfig.raytracingEnabled;
                bool generateInterpolatedFrames = false;
                bool velocityUploaderUsed = false;
                bool tileInterpolationUsed = false;
                bool lookAtInterpolationUsed = false;
                if (requiresFrameMatching) {
                    matchingProfiler.reset();
                    matchingProfiler.start();
                    curFrame.match(ext.workloadGraphicsWorker, *this, prevFrame, ext.workloadVelocityUploader, velocityUploaderUsed, tileInterpolationUsed, lookAtInterpolationUsed);
                    matchingProfiler.end();
                    matchingProfiler.log();

                    const bool displayRateAboveOriginal = (workload.viOriginalRate > 0) && (workloadConfig.targetRate > workload.viOriginalRate);
                    generateInterpolatedFrames = !workload.paused && displayRateAboveOriginal && !interpolationTargetKey.isEmpty();

                    const bool resetTicks = !generateInterpolatedFrames || (originalRateForTicks != workload.viOriginalRate) || (displayRateForTicks != workloadConfig.targetRate) || !displayRateAboveOriginal;
                    if (resetTicks) {
                        logicalTicks = 0;
                        displayTicks = 0;
                        originalRateForTicks = workload.viOriginalRate;
                        displayRateForTicks = workloadConfig.targetRate;
                    }
                }

                // Estimate amount of frames to render based on how many display frames it'd take to reach the next logical frame.
                uint32_t displayFrames = 1;
                if (generateInterpolatedFrames) {
                    logicalTicks += workloadConfig.targetRate;
                    displayFrames = uint32_t((logicalTicks - displayTicks) / workload.viOriginalRate);
                    deltaTimeMs = 1.0f / float(workloadConfig.targetRate);

                    if ((displayFrames > 1) && frameReduction) {
                        displayTicks += workload.viOriginalRate;
                        displayFrames--;
                        frameReduction = false;
                    }

                    assert((logicalTicks > displayTicks) && "Logical ticks must always remain bigger than the display ticks.");
                    assert(((logicalTicks - displayTicks) <= (workloadConfig.targetRate + workload.viOriginalRate)) && "The gap between logical ticks and display ticks can't be bigger than the target rate.");
                    assert((displayFrames > 0) && "At least one display frame must be generated.");
                }
                else if (workload.viOriginalRate > 0) {
                    deltaTimeMs = 1.0f / float(workload.viOriginalRate);
                }

                ext.sharedResources->viOriginalRate = workload.viOriginalRate;
                
                // Get the current and previous set of frame counters. The other set can be in use by the present queue. Skip if no new present event has arrived before this workload event.
                InterpolatedFrameCounters &prevFrameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
                const bool useDifferentCounters = (lastPresentId != workload.presentId);
                if (useDifferentCounters) {
                    ext.sharedResources->interpolatedFramesIndex = ext.sharedResources->interpolatedFramesIndex ^ 1;
                    lastPresentId = workload.presentId;
                }
                // If the same set of counters is used, we wait until the presentation of its targets is finished so the targets are available to use. Waiting is ignored
                // if the frame counter has never presented anything yet, as it'll only be a valid value if the previous present event actually did something.
                else if (generateInterpolatedFrames) {
                    std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                    ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                        return (prevFrameCounters.presented == 0) || (prevFrameCounters.presented >= prevFrameCounters.available);
                    });
                }

                InterpolatedFrameCounters &curFrameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
                curFrameCounters.skipped = false;
                curFrameCounters.presented = 0;
                curFrameCounters.available = 0;
                curFrameCounters.count = displayFrames;

                // Create as many render targets as required to store the interpolated targets.
                auto &interpolatedTargets = ext.sharedResources->interpolatedColorTargets;
                const bool usingMSAA = (ext.sharedResources->renderTargetManager.multisampling.sampleCount > 1);
                const bool usesHDR = ext.sharedResources->renderTargetManager.usesHDR;
                uint32_t requiredFrames = (usingMSAA && generateInterpolatedFrames) ? displayFrames : (displayFrames - 1);
                if ((requiredFrames > 0) && (interpolatedTargets.size() < requiredFrames)) {
                    uint32_t previousSize = uint32_t(interpolatedTargets.size());
                    interpolatedTargets.resize(requiredFrames);
                    for (uint32_t i = previousSize; i < requiredFrames; i++) {
                        interpolatedTargets[i] = std::make_unique<RenderTarget>(interpolationTargetKey.address, Framebuffer::Type::Color, RenderMultisampling(), usesHDR);
                    }
                }
                
                const int64_t originalTimeMicro = (workload.viOriginalRate > 0) ? (1000000 / workload.viOriginalRate) : 0;
                const int64_t setupTimeMicro = workloadTimer.elapsedMicroseconds();
                const int64_t adjustedTimeWindowMicro = originalTimeMicro - setupTimeMicro;
                const int64_t maxTimePerFrameMicro = adjustedTimeWindowMicro / displayFrames;
                bool skippedFrames = false;
                bool skipWorkloadNow = false;
                uint32_t targetIndex = 0;
                uint32_t framesRendered = 0;
                int64_t renderTimeTotalMicro = 0;
                for (uint32_t frame = 0; (frame < displayFrames) && !skipWorkloadNow; frame++) {
                    // Evaluate if this frame should be skipped. Measure the current time and compare it to what frame is estimated should be have been rendered by now.
                    if ((frame > 0) && (originalTimeMicro > 0)) {
                        const int64_t currentTimeMicro = workloadTimer.elapsedMicroseconds() - setupTimeMicro;
                        const int64_t expectedTimeMicro = frame * maxTimePerFrameMicro;
                        const int64_t measuredFrameMicro = renderTimeTotalMicro / framesRendered;
                        if ((currentTimeMicro > expectedTimeMicro) || ((currentTimeMicro + measuredFrameMicro) > adjustedTimeWindowMicro)) {
                            displayTicks += workload.viOriginalRate;
                            skippedFrames = true;
                            continue;
                        }
                    }

                    RenderTarget *overrideTarget = nullptr;
                    uint32_t overrideModifier = 0;
                    if (generateInterpolatedFrames) {
                        prevFrameWeight = std::clamp((workloadConfig.targetRate + displayTicks - logicalTicks) / float(workloadConfig.targetRate), 0.0f, 1.0f);
                        displayTicks += workload.viOriginalRate;
                        curFrameWeight = std::clamp((workloadConfig.targetRate + displayTicks - logicalTicks) / float(workloadConfig.targetRate), 0.0f, 1.0f);

                        // Override the render target.
                        if (usingMSAA || (frame > 0)) {
                            overrideTarget = interpolatedTargets[targetIndex].get();
                            overrideModifier = (targetIndex + 1);

                            if (useDifferentCounters && (prevFrameCounters.available > 0) && (targetIndex < prevFrameCounters.available)) {
                                // Wait until the target has finished presenting if the alternate frame counter (used by the present queue) is making use of this target.
                                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                                ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                                    frameReduction = frameReduction || (prevFrameCounters.presented <= targetIndex);
                                    return prevFrameCounters.presented > targetIndex;
                                });
                            }

                            targetIndex++;
                        }
                    }
                    else if (workload.paused) {
                        curFrameWeight = workload.debuggerRenderer.interpolationWeight;
                        prevFrameWeight = 1.0f - curFrameWeight;
                    }
                    else {
                        prevFrameWeight = 0.0f;
                        curFrameWeight = 1.0f;
                    }

                    const bool uploadExtras = (frame == 0) && workloadConfig.raytracingEnabled;
                    if (uploadExtras) {
                        BufferUploader::Upload extrasUpload = { workload.drawData.extraParams.data(), { 0, workload.drawData.extraParams.size() }, sizeof(interop::ExtraParams), RenderBufferFlag::STORAGE, {}, &workload.drawBuffers.extraParamsBuffer };
                        ext.workloadExtrasUploader->submit(ext.workloadGraphicsWorker, { extrasUpload });
                    }

                    int64_t renderTimeMicro = workloadTimer.elapsedMicroseconds();
                    threadRenderFrame(curFrame, prevFrame, workloadConfig, workload.debuggerRenderer, workload.debuggerCamera, curFrameWeight, prevFrameWeight, deltaTimeMs,
                        interpolationTargetKey, interpolationTargetFbPairIndex, overrideTarget, overrideModifier, velocityUploaderUsed, uploadExtras, tileInterpolationUsed, lookAtInterpolationUsed,
                        StereoEye::Left);

                    // Stereoscopic 3D second pass: re-render this interpolated frame with
                    // the right-eye projection. Paired one-to-one with the left-eye target
                    // so both eyes update at the same rate. Gated on !usingMSAA because
                    // RT64's override-target redirection is itself gated that way.
                    const auto stereoModeWl = ext.sharedResources->userConfig.stereoMode;
                    const bool stereoActiveWl = (stereoModeWl != UserConfiguration::StereoMode::Off);
                    if (stereoActiveWl && !usingMSAA && !workload.paused && !interpolationTargetKey.isEmpty()) {
                        RenderTarget *rightOverride = nullptr;
                        uint32_t rightModifier = 0;
                        if (overrideTarget != nullptr) {
                            // Left eye used interpolatedTargets[targetIndex - 1]; we
                            // pair it with stereoRightInterpolatedTargets at the same
                            // slot so the present queue can match them index-for-index.
                            const uint32_t slot = targetIndex - 1;
                            auto &slots = ext.sharedResources->stereoRightInterpolatedTargets;
                            if (slots.size() <= slot) {
                                slots.resize(slot + 1);
                            }
                            if (slots[slot] == nullptr) {
                                slots[slot] = std::make_unique<RenderTarget>(interpolationTargetKey.address, Framebuffer::Type::Color, RenderMultisampling(), usesHDR);
                            }
                            rightOverride = slots[slot].get();
                            rightModifier = 0x8000u + (slot + 1u);
                        }
                        else {
                            // Left eye used the natural target (the first frame in the
                            // non-MSAA interpolated path, or any frame when interpolation
                            // is off). Right eye goes into the single stereoRightColorTarget.
                            auto &single = ext.sharedResources->stereoRightColorTarget;
                            if (single == nullptr) {
                                single = std::make_unique<RenderTarget>(interpolationTargetKey.address, Framebuffer::Type::Color, RenderMultisampling(), usesHDR);
                            }
                            rightOverride = single.get();
                            rightModifier = 0x8000u;
                        }
                        threadRenderFrame(curFrame, prevFrame, workloadConfig, workload.debuggerRenderer, workload.debuggerCamera, curFrameWeight, prevFrameWeight, deltaTimeMs,
                            interpolationTargetKey, interpolationTargetFbPairIndex, rightOverride, rightModifier, velocityUploaderUsed, false /* extras already uploaded by left pass */, tileInterpolationUsed, lookAtInterpolationUsed,
                            StereoEye::Right);
                    }

                    // Add total time the frame took to render.
                    renderTimeTotalMicro += workloadTimer.elapsedMicroseconds() - renderTimeMicro;

                    // After one frame is rendered, we indicate the workload has been processed so the present thread can start presenting frames as soon as it can.
                    if (frame == 0) {
                        threadAdvanceWorkloadId(workload.workloadId);
                    }

                    // For every additional frame, we increase the frames available and notify the present queue.
                    if (generateInterpolatedFrames && (usingMSAA || (frame > 0))) {
                        {
                            std::scoped_lock<std::mutex> cursorLock(cursorMutex);
                            skipWorkloadNow = ((frame + 1) < displayFrames) && (writeCursor != threadCursor);
                        }

                        {
                            std::scoped_lock<std::mutex> managerLock(ext.sharedResources->interpolatedMutex);
                            curFrameCounters.skipped = skipWorkloadNow;
                            curFrameCounters.available++;
                        }

                        // Add the amount of display ticks that correspond to the remaining frames.
                        if (skipWorkloadNow) {
                            displayTicks += workload.viOriginalRate * (displayFrames - (frame + 1));
                        }

                        ext.sharedResources->interpolatedCondition.notify_all();
                    }

                    framesRendered++;
                }

                // (Right-eye stereo pass is now run inside the displayFrames loop above,
                // paired one-to-one with each left-eye render so both eyes match rate.)

                // Set the skipped parameter on the frame counter if the workload wasn't skipped but some of its frames were.
                if (skippedFrames && !skipWorkloadNow) {
                    {
                        std::scoped_lock<std::mutex> managerLock(ext.sharedResources->interpolatedMutex);
                        curFrameCounters.skipped = true;
                    }

                    ext.sharedResources->interpolatedCondition.notify_all();
                }

                threadConfigurationValidate();

                if (!workload.paused) {
                    threadAdvanceBarrier();
                }

                processCursor = -1;
                workloadProfiler.end();
                workloadProfiler.log();
                workloadProfiler.reset();
            }
        }
    }
    
    void WorkloadQueue::idleThreadLoop() {
        // Beware traveler as you enter the zone of dirty driver hacks. Given N64 games are not exactly a demanding thing to render
        // nowadays for modern GPUs and due to how the plugin's cooperative multiqueue system works, it's sometimes just not possible
        // to keep the GPU busy at all times. It is often the case that the GPU might've already rendered all the frames it needed to
        // generate before the screen update event from the emulator even arrives on time.
        // 
        // Under this situation, some drivers are a bit too trigger-happy to downclock the GPU and lower the power consumption, eventually
        // resulting in very low power states that cause unwanted frametime spikes that can no longer reach the target framerate. This
        // results in visible judder during gameplay.
        //
        // This thread will take care of sending some GPU work that does nothing useful while the GPU is not actually busy generating
        // new frames. The waiting interval is close to the minimum resolution the OS provides and big enough to not cause any significant
        // delays or unwanted power consumption: it's just enough to keep the driver from downclocking to a power state level that is
        // usually intended for 2D work or video playback.
        //
        // This workaround is not required if the driver is configured to be at the "Max Performance" power state.

        Thread::setCurrentThreadName("RT64 Idle");

        const ShaderRecord &idle = ext.shaderLibrary->idle;
        RenderCommandList *commandList = ext.workloadGraphicsWorker->commandList.get();
        while (threadsRunning) {
            {
                std::unique_lock<std::mutex> idleLock(idleMutex);
                idleCondition.wait(idleLock, [&]() {
                    return idleActive || !threadsRunning;
                });
            }

            if (threadsRunning) {
                if (workerMutex.try_lock()) {
                    commandList->begin();
                    commandList->setPipeline(idle.pipeline.get());
                    commandList->setComputePipelineLayout(idle.pipelineLayout.get());
                    commandList->dispatch(1, 1, 1);
                    commandList->end();
                    ext.workloadGraphicsWorker->execute();
                    ext.workloadGraphicsWorker->wait();
                    workerMutex.unlock();
                }
                
                Thread::sleepMilliseconds(1);
            }
        }
    }
};