//
// RT64
//

#pragma once

#include "common/rt64_user_configuration.h"
#include "hle/rt64_vi.h"

#include "rt64_descriptor_sets.h"
#include "rt64_shader_library.h"

namespace RT64 {
    // Mirrors VIRenderer but binds two eye textures and the stereoCompose
    // pipeline so the final present packs the image into Side-by-Side,
    // Top-and-Bottom, or Row-Interlaced output. Reuses VIRenderer's viewport
    // and scissor math so windowed/letterbox/pillarbox behaviour matches the
    // mono path.
    struct StereoRenderer {
        std::unique_ptr<StereoComposeDescriptorSet> descriptorSet;
        // Separate descriptor set for the UI overlay pipeline so we don't
        // disturb the world descriptor's texture bindings between the two
        // back-to-back draws in a single frame.
        std::unique_ptr<StereoComposeDescriptorSet> uiOverlayDescriptorSet;
        const RenderSampler *descriptorSetSampler = nullptr;
        const RenderSampler *uiOverlayDescriptorSetSampler = nullptr;

        struct RenderParams {
            RenderDevice *device = nullptr;
            RenderCommandList *commandList = nullptr;
            RenderTexture *leftTexture = nullptr;
            RenderTexture *rightTexture = nullptr;
            const RenderSwapChain *swapChain = nullptr;
            const ShaderLibrary *shaderLibrary = nullptr;
            RenderFormat textureFormat = RenderFormat::UNKNOWN;
            hlslpp::float2 resolutionScale;
            uint32_t downsamplingScale = 0;
            uint32_t textureWidth = 0;
            uint32_t textureHeight = 0;
            // Dimensions of the right eye's texture. The two eyes are separate
            // render targets that are resized independently, so these are not
            // necessarily the same as the left's. Left as zero, they fall back to
            // the left eye's dimensions, which is correct for callers that bind
            // one texture to both slots (the UI overlay pass).
            uint32_t rightTextureWidth = 0;
            uint32_t rightTextureHeight = 0;
            UserConfiguration::StereoMode stereoMode = UserConfiguration::StereoMode::Off;
            const VI *vi = nullptr;
            bool removeBlackBorders = false;
            // When true, use the alpha-blended pipeline and bind leftTexture to
            // both eye slots. Used for stamping UI content over the already
            // composed stereo world image.
            bool isUIOverlay = false;
            // When non-zero, use these dimensions instead of the swap chain
            // for the viewport/scissor. Set by the LeiaSR path so the compose
            // step renders to a desktop-sized off-screen intermediate that's
            // then fed to the lenticular weaver. Zero falls back to the
            // existing swap-chain-derived viewport math.
            uint32_t targetWidth = 0;
            uint32_t targetHeight = 0;
            // When true, fill the full target with the SbS/TaB/Interlaced
            // image (no VIRenderer letterbox/pillarbox). Used by the LeiaSR
            // path because the weaver expects a packed stereo image covering
            // the entire input texture, not a letterboxed sub-region.
            bool fillFullTarget = false;
            // Aspect-ratio scale that was used to size the eye texture
            // (workloadConfig.aspectRatioTarget / aspectRatioSource). When
            // greater than 1 (e.g. Expand mode on a wide swap chain), the
            // projection processor narrows the projection by 1/aspectRatioScale,
            // so the rendered content fills only the center 1/aspectRatioScale
            // of the texture width with pillarbox on each side. We use this to
            // compute contentOrigin in fillFullTarget mode so each eye slot
            // samples the actual content area instead of carrying the
            // pillarbox through to the SbS output. Defaults to 1 (no
            // adjustment) which preserves the previous behavior.
            float aspectRatioScale = 1.0f;
        };

        StereoRenderer();
        ~StereoRenderer();
        void render(const RenderParams &p);
    };
};
