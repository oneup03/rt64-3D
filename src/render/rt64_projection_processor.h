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

    // The main world projection's depth terms (m[2][2], m[3][2]) and its RSP
    // viewport's depth scale and translate, published each frame for the depth
    // sampler, which needs them to turn stored depth into view distance.
    void stereoPublishWorldDepthTerms(float m22, float m32, float vpScaleZ, float vpTranslateZ);
    bool stereoGetWorldDepthTerms(float &m22, float &m32, float &vpScaleZ, float &vpTranslateZ);

    // The projection processor's unit conversions, for the depth-driven
    // convergence loop: the separation slider as a fraction of screen width,
    // and convergence tenths as game units.
    float stereoSeparationFraction(uint32_t separationSlider);
    float stereoConvergenceWorldUnits(uint32_t convergenceTenths);

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
            // Convergence the depth-driven loop wants, in game units, or 0 when
            // it has nothing to say (auto convergence off, or no depth yet). A
            // float rather than another slider value: the loop solves a
            // continuous convergence, and whole steps would read as the depth
            // visibly stepping. Never larger than stereoConvergence's distance:
            // the slider stays the ceiling.
            float stereoConvergenceAuto = 0.0f;
            // HUD depth slider 0..100 (50 = screen plane). Below 50 pushes HUD
            // behind the screen; above 50 makes it pop out.
            uint32_t stereoHudDepth = 50;
        };

        // Per process() call: the frame's main world projection, the full-width
        // world perspective with the most draw calls.
        const Projection *primaryWorld = nullptr;

        ProjectionProcessor();
        ~ProjectionProcessor();
        void setup(RenderWorker *worker);
        void process(const ProcessParams &p);
        void processScene(const ProcessParams &p, const GameScene &scene, size_t sceneIndex);
        void upload(const ProcessParams &p);
    };
};