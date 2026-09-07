//
// RT64
//

#pragma once

#include "common/rt64_user_configuration.h"
#include "hle/rt64_game_frame.h"

#include "rt64_buffer_uploader.h"

namespace RT64 {
    // The world projection's depth terms (m[2][2], m[3][2]), published each
    // frame by the projection processor so the depth sampler can turn a device
    // depth into a view-space distance using the projection actually in use
    // rather than assuming Dinosaur Planet's nominal near/far — which camSetFOV
    // and the gFarPlane lerp both move at runtime. Returns false until a world
    // projection has been seen.
    // The two projection depth elements plus the VIEWPORT depth mapping the RSP
    // applied on top of them. All four are needed to invert a sampled depth: the
    // buffer holds the viewport-mapped value, not ndc.z.
    void stereoPublishWorldDepthTerms(float m22, float m32, float vpScaleZ, float vpTranslateZ);
    bool stereoGetWorldDepthTerms(float &m22, float &m32, float &vpScaleZ, float &vpTranslateZ);

    // View-space depth under the aiming reticle, for the depth-aware crosshair
    // (dynamic3d 5.1). Published by the depth sampler on the render thread and
    // read back on the same thread a few lines later; the atomic is for the
    // same reason the depth terms above use one, not for cross-thread sharing.
    // <= 0 means "nothing valid", which the crosshair treats as infinity.
    // Where the game says it is drawing the aiming reticle, in its own 320x240
    // screen space. Negative x means none is on screen this frame.
    //
    // Per-frame state, so deliberately NOT part of UserConfiguration even
    // though the game publishes it: the queues read a COPY of that struct which
    // is only refreshed when a setting changes, so a value republished every
    // frame would never arrive. Same reasoning as the auto-convergence values.
    void stereoSetAimScreenPoint(int32_t x, int32_t y);
    bool stereoGetAimScreenPoint(int32_t &x, int32_t &y);

    void stereoStoreAimViewZ(float z);
    float stereoLoadAimViewZ();

    enum class StereoEye {
        None,
        Left,
        Right
    };

    // Per-eye NDC.x offset for screen-space texture rectangles, matching the
    // orthographic branch of the projection processor's HUD shift.
    //
    // Dinosaur Planet draws its HUD as rectangles rather than through an
    // orthographic projection, so the rectangle path in FramebufferRenderer needs
    // the same offset the projection path computes. This lives here, and is the
    // single definition of that math, so the two paths cannot drift apart when
    // the HUD tuning constants are retuned.
    //
    // Returns 0 when there is nothing to shift (mono, or either slider is
    // neutral — the shift scales with separation, so a separation of 0 leaves
    // the HUD flat along with the rest of the image).
    // Per-eye NDC shift that places the aiming reticle at the depth being
    // aimed at (dynamic3d 5.1), in the same units stereoHudRectOffsetX returns.
    // Replaces the HUD shift for that one rect rather than compounding with it.
    float stereoAimRectOffsetX(StereoEye eye, float aimViewZ, uint32_t separationSlider,
                               uint32_t convergenceTenths);

    float stereoHudRectOffsetX(StereoEye eye, uint32_t hudDepthSlider, uint32_t separationSlider);

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
            // Slider values from the host application, converted to clip-space
            // separation / world-space convergence inside processScene.
            //
            // Separation is 0..100 and maps onto 0..0.10 of screen width — it IS
            // the projection shear, so the number is the background disparity the
            // viewer sees (dynamic3d 1.3).
            //
            // Convergence is in TENTHS of its 0.1..100 slider, so 1..1000. The
            // extra digit exists for the depth-driven convergence loop, whose
            // solve is continuous; at whole slider units its output would step in
            // 10-game-unit jumps.
            uint32_t stereoSeparation = 0;
            uint32_t stereoConvergence = 200;
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