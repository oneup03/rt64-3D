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
        // Square patch of depth texels sampled at the centre of the target.
        static constexpr uint32_t PatchSize = 32;
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
        };

        std::array<Slot, SlotCount> slots;
        uint32_t frameIndex = 0;

        // Queue a copy of this frame's centre depth patch. Safe to call with a
        // null or multisampled target; it simply does nothing.
        void submit(RenderWorker *worker, RenderTarget *depthTarget);

        // Read back the oldest completed copy and return the spatial median of
        // the patch in device depth units, or a negative value if there is
        // nothing valid to read yet.
        float fetchMedianDeviceDepth();
    };

    // Convert a device depth value to a positive view-space distance, given the
    // two depth-related elements of the projection the frame was rendered with
    // (m[2][2] and m[3][2]). Returns <= 0 when the sample is invalid (cleared
    // depth, degenerate projection). Taking scalars rather than the matrix keeps
    // this header free of the interop types.
    float stereoDeviceDepthToViewZ(float deviceDepth, float projM22, float projM32);
}
