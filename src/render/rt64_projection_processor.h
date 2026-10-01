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
    //   StereoBehindNdcLimit /
    //   StereoPopOutNdcLimit         - dynamic3d 6.1: pop-out is not
    //                                  divergence, so the two directions get
    //                                  different clamps.
    static constexpr float StereoSeparationPerSlider = 0.10f / 50.0f;
    static constexpr float StereoBehindNdcLimit = 0.10f;   // ~5% of eye width
    static constexpr float StereoPopOutNdcLimit = 0.30f;   // ~15% of eye width

    // Largest interpolation rotation applied to a camera-tracking sky, in
    // radians (see DrawExtendedFlags::skyboxTracksCamera): a full game frame of
    // turning at ~6.9 degrees per frame, against a measured 2.9 degree median
    // and 6.6 maximum. The patch draws such a sky wide enough past each edge to
    // cover this rotation at the live FoV (plus the stereo shift), so it
    // reveals real sky rather than an edge and needs no cover zoom, unlike a
    // static backdrop.
    static constexpr float SkyInterpolationAngleLimit = 0.12f;

    // The clip-space separation for a slider value: the per-eye NDC x offset of
    // content at infinity, and half the total background disparity as a
    // fraction of screen width.
    float stereoSeparationFraction(uint32_t separationSlider);

    // The per-eye NDC x offset every HUD element gets for a given HUD depth,
    // before the eye sign is applied: -(hud - 50) / 50 * separation, so 0 is
    // the depth of infinity, 50 the screen plane and 100 the mirror of
    // infinity in front of it. Sign convention matches
    // applyStereoHudShift's orthographic branch: the caller subtracts
    // eyeSign * this from the element's NDC x.
    float stereoHudNdcOffset(uint32_t hudDepthSlider, uint32_t separationSlider);

    // Rounds a per-eye NDC shift to a whole number of NATIVE pixels - multiples
    // of gridPixels output pixels (the resolution scale), given half the
    // viewport's width in output pixels.
    //
    // Goemon draws its text as point-sampled texture rectangles and
    // orthographic quads, so where a glyph lands against the pixel grid decides
    // which source texel each output pixel takes: two eyes shifted by equal and
    // opposite fractions of a pixel sample different texels, and the font comes
    // out a slightly different weight in each eye, which cannot be fused. The
    // Dino3D branch fixed that by rounding to whole OUTPUT pixels.
    //
    // That is not enough here. With 2D upscaling off, RasterPS emulates native
    // resolution by snapping texture coordinates to native-pixel blocks anchored
    // at absolute screen position (lowResUV -= fmod(screenPos, resolutionScale)
    // * dUV). A shift that is a whole number of output pixels but not of native
    // ones moves a glyph against those blocks, so each eye crosses them at a
    // different point - and at a rect's edge one eye gets a partial block that
    // samples the texel past the glyph, seen as a 1-pixel white line down the
    // edge of the text in that eye only. Whole native pixels keep both eyes on
    // the same block phase, at the cost of the HUD moving in native-pixel steps.
    float stereoSnapNdcToPixel(float ndcOffset, float halfViewportWidth, float gridPixels);

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
            // The workload's base resolution scale, from which each framebuffer
            // pair's output pixel width is derived exactly as the framebuffer
            // renderer derives it - so HUD shifts baked into projections can be
            // rounded to the same pixel grid as the ones it applies to rects.
            hlslpp::float2 resolutionScale = { 1.0f, 1.0f };
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

        // Per process() call: the frame's main world projection, the angle by which the INTERPOLATED view lags the CURRENT
        // camera's heading (the bearing of the current forward direction in the
        // interpolated view, positive to the right; zero on a non-interpolated
        // frame), and the world projection's horizontal scale m[0][0]. The
        // workload queue hands both to the framebuffer renderer, which rotates
        // camera-tracking skies by the angle.
        const Projection *primaryWorld = nullptr;
        float skyInterpolationAngle = 0.0f;
        float skyProjScaleX = 0.0f;

        ProjectionProcessor();
        ~ProjectionProcessor();
        void setup(RenderWorker *worker);
        void process(const ProcessParams &p);
        void processScene(const ProcessParams &p, const GameScene &scene, size_t sceneIndex);
        void upload(const ProcessParams &p);
    };
};