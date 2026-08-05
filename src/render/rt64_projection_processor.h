//
// RT64
//

#pragma once

#include "common/rt64_user_configuration.h"
#include "hle/rt64_game_frame.h"

#include "rt64_buffer_uploader.h"

namespace RT64 {
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
    // Returns 0 when there is nothing to shift (mono, or the slider is neutral).
    float stereoHudRectOffsetX(StereoEye eye, uint32_t hudDepthSlider);

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
            uint32_t stereoConvergence = 50;
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