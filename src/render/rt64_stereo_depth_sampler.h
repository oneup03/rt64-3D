//
// RT64
//

#pragma once

#include <array>
#include <memory>

#include "rt64_render_target.h"

namespace RT64 {
    // Reads back a small patch of the depth buffer at screen centre, for the
    // depth-aware stereo features (dynamic3d 4.1 / 5.1).
    //
    // The copy is queued on the render thread and read back several frames
    // later out of a ring of readback buffers, rather than stalling the GPU the
    // way RT64's framebuffer-change readback does (end/execute/wait/begin).
    // A per-frame full pipeline flush would cost more than this feature is
    // worth, and a few frames of latency is invisible on a value that is
    // temporally smoothed anyway.
    //
    // Depth arrives in DEVICE units (the depth target is D32_FLOAT). Converting
    // that to a view-space distance needs the projection's own near/far
    // handedness, which dynamic3d 4.1 explicitly warns not to assume - see
    // stereoDeviceDepthToViewZ.
    struct StereoDepthSampler {
        // Square patch of depth texels. Several of these are copied per frame
        // rather than one big region: dynamic3d 4.1 wants the NEAR statistic
        // taken over a wide region of interest, but a contiguous copy covering
        // most of a 4K depth target would move tens of megabytes a frame. A
        // sparse grid of small patches spans the same area for a fraction of it.
        //
        // A single centre patch is not enough on its own - it was measured
        // reporting 330-450 units in third-person play, because the middle of
        // the screen is mid-distance scenery while the player and the near
        // foreground sit elsewhere in frame.
        static constexpr uint32_t PatchSize = 32;
        // Grid spanning the middle ~90% of the frame in each axis. 45 patches
        // of 32x32 is ~46k depth texels per frame - still a small fraction of a
        // 4K depth target, and dense enough that a near object anywhere in frame
        // lands in several of them rather than being missed between samples.
        // Frame edges excluded from the near statistic, as percentages. The
        // bottom margin is much larger than the top: see the placement code in
        // the .cpp for why the box has to be asymmetric rather than merely
        // wide. Raising RoiMarginBottomPercent excludes more low-frame clutter
        // at the cost of ignoring genuinely near floor when looking down.
        static constexpr uint32_t RoiMarginLeftPercent = 5;
        static constexpr uint32_t RoiMarginRightPercent = 5;
        static constexpr uint32_t RoiMarginTopPercent = 5;
        static constexpr uint32_t RoiMarginBottomPercent = 25;
        static constexpr uint32_t PatchCols = 9;
        static constexpr uint32_t PatchRows = 5;
        static constexpr uint32_t PatchCount = PatchCols * PatchRows;
        // Aim candidates: a horizontal STRIP of tight patches, spanning every
        // buffer position the aim point could resolve to.
        //
        // Not a wide pooled window - that is the lateral-bias bug, and these are
        // never pooled. Each patch is judged on its own and exactly one is
        // chosen, by the self-consistency test in WorkloadQueue.
        //
        // The strip replaces placing ONE patch from the previous frame's depth.
        // That was circular - the position depends on the depth being measured -
        // and at a depth edge the circularity has TWO solutions, so the loop
        // alternated between them: measured jitter, and a near object that would
        // not commit until it was dead centre. Sampling the whole candidate
        // range at once removes the frame-to-frame feedback completely.
        // 13 across the span puts the patches about 24 texels apart at 4K and
        // the default separation. The spacing sets how precisely the strip can
        // resolve "under the crosshair": too coarse and a surface that IS at the
        // aim point has no patch close enough to its predicted position to be
        // recognised, which was measured dropping a near object for the last
        // stretch of its sweep.
        static constexpr uint32_t AimPatchCount = 13;
        static constexpr uint32_t AimPatchIndex = PatchCount;
        static constexpr uint32_t TotalPatchCount = PatchCount + AimPatchCount;
        // Footprint rows must be 256-byte aligned for the buffer copy, and the
        // depth format is 4 bytes per texel, so the footprint is padded out to
        // 64 texels per row and only the first PatchSize of each are read.
        static constexpr uint32_t FootprintRowTexels = 64;
        // Frames of slack between queueing a copy and reading it. Enough that
        // the GPU has long since finished without needing an explicit fence.
        static constexpr uint32_t SlotCount = 4;

        struct Slot {
            std::unique_ptr<RenderBuffer> buffer;
            bool queued = false;
            // Offsets, in depth-target texels from the undisplaced aim point,
            // that this slot's aim patches were taken at. Recorded per slot
            // because the readback is several frames behind and the span is
            // derived from the separation setting, which the user can move in
            // the meantime.
            int32_t aimOffsets[AimPatchCount] = {};
        };

        std::array<Slot, SlotCount> slots;
        uint32_t frameIndex = 0;

        // Queue a copy of this frame's centre depth patch. Safe to call with a
        // null or multisampled target; it simply does nothing.
        //
        // Returns true only when a copy was actually queued, so the caller can
        // avoid reading back on a frame it did not sample. The caller runs once
        // per framebuffer PAIR and a frame has several - the world pass plus
        // smaller auxiliary ones whose centre depth is empty - so it is the
        // caller's job to offer only the main pass.
        // contentOriginX/contentWidth bound the part of the target the viewer
        // actually SEES, in texels - see stereoEyeVisibleSpanX. Pass 0 and the
        // full width when nothing is cropped. The patch grid is laid out inside
        // that span rather than across the whole target: on an ultrawide the two
        // are very different, and sampling the discarded margins lets geometry
        // the player cannot see drive convergence.
        //
        // aimOffsetMin/Max bound the strip, in texels either side of aimCenterX.
        // aimCenterX/Y are the UNDISPLACED aim point in DEPTH TARGET texels. They are passed
        // in rather than derived from the target's own dimensions because the
        // two are not the same thing: a render target can be padded and can carry
        // a horizontal misalignment relative to the visible viewport, so
        // targetWidth/2 is not where the player is looking. Centring on it put
        // the aim window off to one side of the reticle.
        bool submit(RenderWorker *worker, RenderTarget *depthTarget,
                    uint32_t contentOriginX, uint32_t contentWidth,
                    int32_t aimCenterX, int32_t aimCenterY,
                    int32_t aimOffsetMin, int32_t aimOffsetMax);

        struct AimCandidate {
            bool valid = false;
            float deviceDepth = -1.0f;
            // Where this patch sat, in texels from the undisplaced aim point.
            int32_t offsetTexels = 0;
            // Covered texels, for the bring-up log.
            uint32_t sampleCount = 0;
        };

        struct Sample {
            bool valid = false;
            // One entry per strip patch, each a MEDIAN of its own window - not
            // the low percentile used for the near statistic above. The
            // crosshair wants the depth of whatever FILLS a window, and a low
            // percentile biases toward whichever edge of it happens to overlap
            // something nearer.
            //
            // Validity is per candidate and independent of `valid` above: the
            // grid and the strip can succeed and fail separately, and the
            // crosshair runs whether or not auto-convergence does.
            AimCandidate aim[AimPatchCount];
            // Robust NEAR depth: a low percentile rather than the minimum, so a
            // single stray texel from a particle or a sliver of geometry cannot
            // slam convergence to the floor (dynamic3d 4.1).
            float nearestDeviceDepth = -1.0f;
        };

        // Read back the oldest completed copy. Both statistics come from one
        // mapping of the buffer.
        Sample fetch();
    };

    // Depth-driven auto-convergence (dynamic3d 4.2 - 4.5).
    //
    // Keeps the nearest significant on-screen object's pop-out under a comfort
    // budget by pulling convergence in, without the player having to chase the
    // slider. The manual value is always a CEILING: this only ever moves the
    // screen plane closer, never further, so the slider still does what the user
    // expects.
    //
    // Under the clip-space parameterization this is a single closed form with no
    // co-scaling of separation: convergence no longer changes background
    // disparity, so there is nothing to hold constant while pulling in.
    struct StereoAutoConvergence {
        // Longer than dynamic3d's suggested 5. The two-level near statistic is
        // deliberately sensitive to small objects, so it legitimately jumps as
        // one enters or leaves a patch; a wider temporal median rejects those
        // transients without slowing the response to real approach.
        static constexpr uint32_t HistorySize = 9;

        float history[HistorySize] = {};
        uint32_t historyCount = 0;
        uint32_t historyCursor = 0;
        float zEma = -1.0f;
        float invConvSmoothed = -1.0f;
        // Consecutive frames the depth has sat far away from the EMA. A camera
        // cut stays changed; a small object flickering in and out of a patch
        // does not, so the snap only fires once this persists.
        uint32_t cutCandidateFrames = 0;

        // nearestViewZ: robust near-depth for this frame, in game units.
        // manualConvergenceHundredths / separationSlider: the user's settings.
        // Returns the convergence to apply, in the same hundredths unit, which is
        // the manual value when there is nothing to protect against.
        // comfortTarget: permitted pop-out in thousandths of screen width.
        //   Signed - negative puts the screen plane in front of the nearest
        //   object, so the whole scene sits behind the screen.
        // sceneWantsLowConvergence: the game reports a close-framed scene, which
        //   tightens the budget further.
        uint32_t update(float nearestViewZ, uint32_t manualConvergenceHundredths, uint32_t separationSlider,
                        int32_t comfortTarget, bool sceneWantsLowConvergence);

        // Snap back to the manual value rather than easing out, which would read
        // as unexplained drift after the user turns the feature off.
        void reset();
    };

    // Convert a device depth value to a positive view-space distance, given the
    // two depth-related elements of the projection the frame was rendered with
    // (m[2][2] and m[3][2]). Returns <= 0 when the sample is invalid (cleared
    // depth, degenerate projection). Taking scalars rather than the matrix keeps
    // this header free of the interop types.
    // vpScaleZ / vpTranslateZ are the RSP viewport's depth scale and translate.
    // Pass a non-positive vpScaleZ when the frame carried no viewport, and the
    // inversion falls back to assuming ndc.z spans [-1,1] over a [0,1] buffer.
    float stereoDeviceDepthToViewZ(float deviceDepth, float projM22, float projM32,
                                   float vpScaleZ, float vpTranslateZ);
}
