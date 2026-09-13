//
// RT64
//

#pragma once

#include "common/rt64_user_configuration.h"
#include "hle/rt64_game_frame.h"

#include "rt64_buffer_uploader.h"

namespace RT64 {
    // The world projection's depth terms, published each frame by the
    // projection processor so the depth sampler can turn a device depth into a
    // view-space distance using the projection actually in use rather than
    // assuming DK64's nominal near/far. Returns false until a world projection
    // has been seen.
    //
    // m22/m32 are the projection's depth row (m[2][2], m[3][2]). vpScaleZ /
    // vpTranslateZ are the RSP VIEWPORT's depth scale and translate, which sit
    // BETWEEN ndc.z and what the depth buffer actually holds - dynamic3d 4.1
    // warns that this is rarely the clean half it looks like, and on the N64 it
    // is a fixed-point value slightly under it.
    void stereoPublishWorldDepthTerms(float m22, float m32, float vpScaleZ, float vpTranslateZ);
    bool stereoGetWorldDepthTerms(float &m22, float &m32, float &vpScaleZ, float &vpTranslateZ);

    // Where the first-person reticle actually IS, in native screen
    // coordinates, read out of the draw that FramebufferRenderer matched rather
    // than assumed to be the middle of the viewport.
    //
    // The aim point used to be the scissor centre. DK64 draws its reticle
    // through the HUD's own coordinate mapping, which is not obliged to put it
    // there, and the geometric match only requires it within a fifth of the
    // viewport - so a constant offset of a hundred-odd texels passes unnoticed
    // and biases every depth sample by it. That reads as the reticle committing
    // late onto a target and clinging to it after sweeping off, which is exactly
    // the symptom a fixed lateral offset produces.
    //
    // Accumulated as a bounding box over the frame's matched quads (DK64's
    // reticle is nine of them) and promoted once per pass, so readers get a
    // completed frame's box rather than a half-built one.
    void stereoAccumulateReticleBounds(float minX, float maxX, float minY, float maxY);
    void stereoPromoteReticleBounds();
    bool stereoGetReticleCenter(float &x, float &y);

    enum class StereoEye {
        None,
        Left,
        Right
    };

    // Depth-aware crosshair (dynamic3d 5.1). The per-eye NDC x offset a reticle
    // needs to land at aimViewZ: zero at the convergence distance, tending to
    // the full background disparity as the aim point recedes, negative (pop-out)
    // nearer than convergence. A non-positive aimViewZ means "no usable depth"
    // and resolves to INFINITY rather than to the screen plane.
    //
    // Shared with the depth sampler's per-eye aim correction so the two cannot
    // drift apart - dynamic3d 1.4 is explicit that deriving the same
    // displacement twice is how sign conventions diverge.
    float stereoAimRectOffsetX(StereoEye eye, float aimViewZ, uint32_t separationSlider,
                               uint32_t convergenceHundredths);

    struct ProjectionProcessor {
        std::unique_ptr<BufferUploader> bufferUploader;
        std::vector<BufferUploader::Upload> uploads;

        struct ProcessParams {
            RenderWorker *worker = nullptr;
            WorkloadQueue *workloadQueue = nullptr;
            GameFrame *curFrame = nullptr;
            const GameFrame *prevFrame = nullptr;
            float curFrameWeight = 1.0f;
            float prevFrameWeight = 0.0f;
            float aspectRatioScale = 1.0f;
            // Stereoscopic 3D parameters. When stereoMode == Off (default), no per-eye
            // adjustment is performed and behavior matches the original mono pipeline.
            UserConfiguration::StereoMode stereoMode = UserConfiguration::StereoMode::Off;
            StereoEye stereoEye = StereoEye::None;
            // Slider values 0..100 from the host application; converted to world-space
            // separation/convergence inside processScene.
            uint32_t stereoSeparation = 0;
            // Hundredths of a slider unit; 500 = 5.0 on the slider.
            uint32_t stereoConvergence = 500;
            // HUD depth slider 0..100 (50 = screen plane). Below 50 pushes HUD
            // behind the screen; above 50 makes it pop out.
            uint32_t stereoHudDepth = 50;
        };

        ProjectionProcessor();
        ~ProjectionProcessor();
        void setup(RenderWorker *worker);
        void process(const ProcessParams &p);
        void processScene(const ProcessParams &p, const GameScene &scene, size_t sceneIndex);
        void upload(const ProcessParams &p);
    };
};