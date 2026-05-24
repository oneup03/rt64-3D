//
// RT64
//

#include "rt64_stereo_renderer.h"

#include "shared/rt64_hlsl.h"
#include "shared/rt64_stereo_compose.h"

#include "rt64_vi_renderer.h"

namespace RT64 {
    StereoRenderer::StereoRenderer() { }

    StereoRenderer::~StereoRenderer() { }

    void StereoRenderer::render(const RenderParams &p) {
        const ShaderRecord *shader = p.isUIOverlay
            ? &p.shaderLibrary->stereoComposeUIOverlay
            : &p.shaderLibrary->stereoCompose;
        const RenderSampler *sampler = p.shaderLibrary->samplerLibrary.linear.borderBorder.get();

        // Use a separate descriptor set per pipeline so the world and UI passes
        // can keep their texture bindings independent within a single frame.
        std::unique_ptr<StereoComposeDescriptorSet> &activeSet = p.isUIOverlay ? uiOverlayDescriptorSet : descriptorSet;
        const RenderSampler *&activeSampler = p.isUIOverlay ? uiOverlayDescriptorSetSampler : descriptorSetSampler;
        if ((activeSet == nullptr) || (activeSampler != sampler)) {
            activeSet = std::make_unique<StereoComposeDescriptorSet>(sampler, p.device);
            activeSampler = sampler;
        }

        // For the UI overlay we bind the same texture to both eye slots so the
        // shader's SbS/TaB/Interlaced logic naturally mirrors the UI across
        // halves/rows.
        RenderTexture *rightForBind = p.isUIOverlay ? p.leftTexture : p.rightTexture;
        activeSet->setTexture(activeSet->gLeftEye, p.leftTexture, RenderTextureLayout::SHADER_READ);
        activeSet->setTexture(activeSet->gRightEye, rightForBind, RenderTextureLayout::SHADER_READ);

        // World stereo composition uses VIRenderer's letterbox/pillarbox math so
        // the game content sits in the same region as the mono path. The UI
        // overlay, on the other hand, must cover the entire swap chain — the
        // Configuration GUI is authored for the full window, not the 4:3 VI
        // viewport. The fillFullTarget path (used by LeiaSR) also wants the
        // full extent because the weaver expects a packed stereo image
        // covering the entire input texture, with no letterbox.
        RenderViewport viewport;
        RenderRect scissor;
        if (p.isUIOverlay || p.fillFullTarget) {
            const float targetW = p.targetWidth ? float(p.targetWidth) : float(p.swapChain->getWidth());
            const float targetH = p.targetHeight ? float(p.targetHeight) : float(p.swapChain->getHeight());
            viewport = RenderViewport(0.0f, 0.0f, targetW, targetH);
            scissor = RenderRect(0, 0, lround(targetW), lround(targetH));
        }
        else {
            VIRenderer::getViewportAndScissor(p.swapChain, *p.vi, p.resolutionScale, p.downsamplingScale, p.removeBlackBorders, viewport, scissor);
        }
        p.commandList->setViewports(viewport);
        p.commandList->setScissors(scissor);

        // Zero-init so the padding word (_pad0) doesn't carry uninitialized
        // bytes into the GPU buffer.
        interop::StereoComposeCB pushConstants = {};
        pushConstants.contentOrigin = { 0.0f, 0.0f };
        if (p.isUIOverlay) {
            // Force the shader's videoResolution/textureResolution ratio to 1
            // so the no-op rescale at the bottom of PSMain leaves the
            // normalized UV unchanged, then SampleEye's overlay-mode branch
            // samples the full UI texture across each eye half.
            pushConstants.videoResolution = { 1.0f, 1.0f };
            pushConstants.textureResolution = { 1.0f, 1.0f };
            pushConstants.gamma = 1.0f;
        } else if (p.fillFullTarget) {
            // Default fillFullTarget: sample the full eye texture across each
            // SbS / TaB / Interlaced / LeiaSR-intermediate slot. The texture's
            // content already covers the canvas (Expand mode widens the FoV
            // to fill it), so each half gets the whole rendered view
            // stretched/squished to fit — the "fills the half" behavior the
            // user picked as the default.
            const float texW = static_cast<float>(p.textureWidth ? p.textureWidth : 1);
            const float texH = static_cast<float>(p.textureHeight ? p.textureHeight : 1);
            pushConstants.videoResolution = { texW, texH };
            pushConstants.textureResolution = { texW, texH };
            pushConstants.contentOrigin = { 0.0f, 0.0f };
            pushConstants.gamma = p.vi ? p.vi->gamma() : 1.0f;

            // When the rendered view is wider than 16:9 per eye (typical on
            // ultrawide / 32:9 desktops, where full-SbS AR glasses split the
            // image into two 16:9 halves), crop each eye to its centered 16:9
            // slice so the glasses see correctly-proportioned content instead
            // of a horizontally squashed view. On 16:9 displays texAspect is
            // ~16:9 and this is a no-op.
            const float texAspect = (texH > 0.0f) ? (texW / texH) : 1.0f;
            constexpr float kTargetEyeAspect = 16.0f / 9.0f;
            if (texAspect > kTargetEyeAspect + 1e-4f) {
                const float contentFractionX = kTargetEyeAspect / texAspect;
                pushConstants.videoResolution = { texW * contentFractionX, texH };
                pushConstants.contentOrigin = { (1.0f - contentFractionX) * 0.5f, 0.0f };
            }
        } else {
            pushConstants.videoResolution = (hlslpp::float2(p.vi->fbSize()) * p.resolutionScale) / float(p.downsamplingScale);
            pushConstants.textureResolution = { float(p.textureWidth), float(p.textureHeight) };
            pushConstants.gamma = p.vi->gamma();
        }
        // LeiaSR's intermediate is SbS-packed, so the shader uses the
        // SideBySide split for it too. Translate to the layout the shader
        // (PSMain case 1) actually knows about rather than passing an enum
        // value that would fall through to the default branch.
        auto shaderStereoMode = p.stereoMode;
        if (shaderStereoMode == UserConfiguration::StereoMode::LeiaSR) {
            shaderStereoMode = UserConfiguration::StereoMode::SideBySide;
        }
        pushConstants.stereoMode = static_cast<uint32_t>(shaderStereoMode);
        pushConstants.useUIOverlayMode = p.isUIOverlay ? 1u : 0u;

        p.commandList->setPipeline(shader->pipeline.get());
        p.commandList->setGraphicsPipelineLayout(shader->pipelineLayout.get());
        p.commandList->setGraphicsDescriptorSet(activeSet->get(), 0);
        p.commandList->setGraphicsPushConstants(0, &pushConstants);
        p.commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        p.commandList->drawInstanced(3, 1, 0, 0);
    }
};
