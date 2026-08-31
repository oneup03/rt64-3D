//
// RT64
//

#include "rt64_stereo_depth_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace RT64 {
    static constexpr uint32_t DepthTexelSize = 4; // D32_FLOAT
    static constexpr uint32_t PatchBufferSize =
        StereoDepthSampler::FootprintRowTexels * StereoDepthSampler::PatchSize * DepthTexelSize;

    void StereoDepthSampler::submit(RenderWorker *worker, RenderTarget *depthTarget) {
        if ((worker == nullptr) || (depthTarget == nullptr)) {
            return;
        }

        // Multisampled depth can't be copied to a buffer directly, and stereo
        // is mutually exclusive with MSAA anyway.
        if (depthTarget->multisampling.sampleCount > 1) {
            return;
        }

        RenderTexture *texture = depthTarget->texture.get();
        if (texture == nullptr) {
            return;
        }

        const uint32_t targetWidth = depthTarget->width;
        const uint32_t targetHeight = depthTarget->height;
        if ((targetWidth < PatchSize) || (targetHeight < PatchSize)) {
            return;
        }

        Slot &slot = slots[frameIndex % SlotCount];
        if (slot.buffer == nullptr) {
            slot.buffer = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(PatchBufferSize));
            if (slot.buffer == nullptr) {
                return;
            }

            // Zero it so a read that lands before the GPU has ever written this
            // slot yields an obviously-invalid value rather than heap garbage.
            void *mapped = slot.buffer->map();
            if (mapped != nullptr) {
                std::memset(mapped, 0, PatchBufferSize);
                slot.buffer->unmap();
            }
        }

        const uint32_t left = (targetWidth - PatchSize) / 2;
        const uint32_t top = (targetHeight - PatchSize) / 2;

        // The footprint format must match the depth resource's own format;
        // describing a D32_FLOAT depth surface as R32_FLOAT makes the copy
        // invalid and it silently produces nothing.
        RenderTextureCopyLocation dstLocation = RenderTextureCopyLocation::PlacedFootprint(
            slot.buffer.get(), RenderFormat::D32_FLOAT, FootprintRowTexels, PatchSize, 1, FootprintRowTexels, 0);
        RenderTextureCopyLocation srcLocation = RenderTextureCopyLocation::Subresource(texture, 0);
        const RenderBox srcBox(int32_t(left), int32_t(top), int32_t(left + PatchSize), int32_t(top + PatchSize));

        worker->commandList->barriers(RenderBarrierStage::COPY,
            RenderTextureBarrier(texture, RenderTextureLayout::COPY_SOURCE));
        worker->commandList->copyTextureRegion(dstLocation, srcLocation, 0, 0, 0, &srcBox);

        // Hand the target back in the layout the rest of the frame expects.
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS,
            RenderTextureBarrier(texture, RenderTextureLayout::SHADER_READ));

        slot.queued = true;
        frameIndex++;
    }

    float StereoDepthSampler::fetchMedianDeviceDepth() {
        if (frameIndex < SlotCount) {
            return -1.0f;
        }

        // The oldest slot in the ring: written SlotCount frames ago.
        Slot &slot = slots[frameIndex % SlotCount];
        if (!slot.queued || (slot.buffer == nullptr)) {
            return -1.0f;
        }

        const RenderRange readRange(0, PatchBufferSize);
        const float *data = reinterpret_cast<const float *>(slot.buffer->map(0, &readRange));
        if (data == nullptr) {
            return -1.0f;
        }

        // dynamic3d 4.1 - a spatial median, never a min. One stray texel from a
        // particle or a sliver of geometry at the patch edge must not be able to
        // yank the result.
        std::vector<float> samples;
        samples.reserve(PatchSize * PatchSize);
        for (uint32_t y = 0; y < PatchSize; y++) {
            const float *row = data + (size_t(y) * FootprintRowTexels);
            for (uint32_t x = 0; x < PatchSize; x++) {
                const float d = row[x];
                // Discard cleared/far texels and anything outside a sane device
                // range; those are sky or an untouched buffer, not a surface the
                // player is looking at.
                if ((d > 0.0f) && (d < 1.0f)) {
                    samples.push_back(d);
                }
            }
        }

        slot.buffer->unmap();

        if (samples.empty()) {
            return -1.0f;
        }

        const size_t mid = samples.size() / 2;
        std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
        return samples[mid];
    }

    float stereoDeviceDepthToViewZ(float deviceDepth, float projM22, float projM32) {
        if ((deviceDepth <= 0.0f) || (deviceDepth >= 1.0f)) {
            return -1.0f;
        }

        // For a standard perspective projection, clip.z = view.z * m[2][2] + m[3][2]
        // and clip.w = -view.z, so ndc.z = -(m[2][2] + m[3][2] / view.z) and
        //     view.z = m[3][2] / (-ndc.z - m[2][2]).
        // The sign conventions differ between the GL-style matrix the N64 builds
        // and the [0,1] device range the backends use, so this is derived from
        // the live matrix rather than from hardcoded near/far values.
        const float denom = -deviceDepth - projM22;
        if (std::fabs(denom) < 1e-6f) {
            return -1.0f;
        }

        const float viewZ = projM32 / denom;
        return (viewZ > 0.0f) ? viewZ : -viewZ;
    }
}
