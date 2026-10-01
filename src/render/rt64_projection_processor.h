//
// RT64
//

#pragma once

#include "common/rt64_user_configuration.h"
#include "hle/rt64_game_frame.h"

#include "rt64_buffer_uploader.h"

namespace RT64 {
    // The world projection's depth terms (m[2][2], m[3][2]) and its RSP
    // viewport's depth scale/translate, published each frame by the projection
    // processor so the depth sampler can turn a device depth into a view-space
    // distance using the transform actually in use rather than assuming a
    // nominal near/far or a clean 0.5 viewport. vpScaleZ is 0 when the world
    // projection had no viewport. Returns false until one has been seen.
    void stereoPublishWorldDepthTerms(float m22, float m32, float vpScaleZ, float vpTranslateZ);
    bool stereoGetWorldDepthTerms(float &m22, float &m32, float &vpScaleZ, float &vpTranslateZ);

    // Clip-space stereo constants (dynamic3d 1.3), shared with the
    // texture-rectangle path in rt64_workload_queue.cpp. That path has to
    // reproduce the orthographic HUD shift and the at-infinity shift for 2D
    // draws that never pass through a projection matrix, so the tuned numbers
    // live here rather than being written out twice and drifting apart.
    //
    //   StereoSeparationPerSlider    - 0..50 slider -> 0..0.10 of screen width
    //                                  of background disparity.
    //   StereoHudReferenceSeparation - the separation the HUD offsets below
    //                                  were tuned at (the shipped default), so
    //                                  HUD depth scales with the depth knob and
    //                                  goes flat when separation is 0.
    //   StereoHudMaxOffset           - NDC offset at the ends of the HUD slider.
    //   StereoHudOrthoScale          - matches an orthographic HUD element's
    //                                  visible depth to a perspective one at
    //                                  the same slider value.
    //   StereoBehindNdcLimit /
    //   StereoPopOutNdcLimit         - dynamic3d 6.1: pop-out is not
    //                                  divergence, so the two directions get
    //                                  different clamps.
    static constexpr float StereoSeparationPerSlider = 0.10f / 50.0f;
    static constexpr float StereoHudReferenceSeparation = 10.0f * StereoSeparationPerSlider;
    static constexpr float StereoHudMaxOffset = 0.04f;
    static constexpr float StereoHudOrthoScale = 2.75f;
    static constexpr float StereoBehindNdcLimit = 0.10f;   // ~5% of eye width
    static constexpr float StereoPopOutNdcLimit = 0.30f;   // ~15% of eye width

    // The clip-space separation for a slider value: the per-eye NDC x offset of
    // content at infinity, and half the total background disparity as a
    // fraction of screen width.
    float stereoSeparationFraction(uint32_t separationSlider);

    // The per-eye NDC x offset an ORTHOGRAPHIC HUD element gets for a given HUD
    // depth, before the eye sign is applied. Sign convention matches
    // applyStereoHudShift's orthographic branch: the caller subtracts
    // eyeSign * this from the element's NDC x.
    float stereoHudOrthoNdcOffset(uint32_t hudDepthSlider, uint32_t separationSlider);

    enum class StereoEye {
        None,
        Left,
        Right
    };

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
            // Slider values from the host application.
            //   stereoSeparation: 0..50, the CLIP-SPACE separation in units of
            //     0.002 of screen width, so the full range is 0..0.10 of screen
            //     width of background disparity (dynamic3d 1.3).
            //   stereoConvergence: 1..200, the zero-parallax distance directly
            //     in game units - no scaling, it is a distance rather than a
            //     disparity and needs no calibration against the projection.
            uint32_t stereoSeparation = 0;
            uint32_t stereoConvergence = 20;
            // Convergence the depth-driven loop wants, in game units, or 0 when
            // it has nothing to say. A float rather than another slider value:
            // the loop solves a continuous convergence, and at Goemon's scale
            // (the loop pulls to single-digit distances) whole-unit steps are
            // 10-20% jumps that read as the depth visibly stepping. Never larger
            // than stereoConvergence - the slider stays the ceiling.
            float stereoConvergenceAuto = 0.0f;
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