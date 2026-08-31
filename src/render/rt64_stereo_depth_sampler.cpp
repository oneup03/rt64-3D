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
    // One patch occupies a whole number of 512-byte blocks, which keeps every
    // patch's offset legal as a texture-copy placed-footprint offset.
    static constexpr uint32_t PatchStride =
        StereoDepthSampler::FootprintRowTexels * StereoDepthSampler::PatchSize * DepthTexelSize;
    static constexpr uint32_t PatchBufferSize = PatchStride * StereoDepthSampler::PatchCount;

    bool StereoDepthSampler::submit(RenderWorker *worker, RenderTarget *depthTarget) {
        if ((worker == nullptr) || (depthTarget == nullptr)) {
            return false;
        }

        // Multisampled depth can't be copied to a buffer directly, and stereo
        // is mutually exclusive with MSAA anyway.
        if (depthTarget->multisampling.sampleCount > 1) {
            return false;
        }

        RenderTexture *texture = depthTarget->texture.get();
        if (texture == nullptr) {
            return false;
        }

        const uint32_t targetWidth = depthTarget->width;
        const uint32_t targetHeight = depthTarget->height;
        if ((targetWidth < PatchSize) || (targetHeight < PatchSize)) {
            return false;
        }

        Slot &slot = slots[frameIndex % SlotCount];
        if (slot.buffer == nullptr) {
            slot.buffer = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(PatchBufferSize));
            if (slot.buffer == nullptr) {
                return false;
            }

            // Zero it so a read that lands before the GPU has ever written this
            // slot yields an obviously-invalid value rather than heap garbage.
            void *mapped = slot.buffer->map();
            if (mapped != nullptr) {
                std::memset(mapped, 0, PatchBufferSize);
                slot.buffer->unmap();
            }
        }

        worker->commandList->barriers(RenderBarrierStage::COPY,
            RenderTextureBarrier(texture, RenderTextureLayout::COPY_SOURCE));

        // Spread the grid over the middle ~90% of the frame, still stopping
        // short of the very edges where HUD-hugging geometry would drag the near
        // statistic down.
        const RenderTextureCopyLocation srcLocation = RenderTextureCopyLocation::Subresource(texture, 0);
        const uint32_t usableW = (targetWidth * 9) / 10;
        const uint32_t usableH = (targetHeight * 9) / 10;
        const uint32_t originX = (targetWidth - usableW) / 2;
        const uint32_t originY = (targetHeight - usableH) / 2;

        for (uint32_t row = 0; row < PatchRows; row++) {
            for (uint32_t col = 0; col < PatchCols; col++) {
                const uint32_t index = (row * PatchCols) + col;
                // Patch centres evenly spaced across the usable region.
                const uint32_t cx = originX + ((usableW * (2 * col + 1)) / (2 * PatchCols));
                const uint32_t cy = originY + ((usableH * (2 * row + 1)) / (2 * PatchRows));
                uint32_t left = (cx > (PatchSize / 2)) ? (cx - (PatchSize / 2)) : 0u;
                uint32_t top = (cy > (PatchSize / 2)) ? (cy - (PatchSize / 2)) : 0u;
                left = std::min(left, targetWidth - PatchSize);
                top = std::min(top, targetHeight - PatchSize);

                // The footprint format must match the depth resource's own
                // format; describing a D32_FLOAT depth surface as R32_FLOAT
                // makes the copy invalid and it silently produces nothing.
                const RenderTextureCopyLocation dstLocation = RenderTextureCopyLocation::PlacedFootprint(
                    slot.buffer.get(), RenderFormat::D32_FLOAT, FootprintRowTexels, PatchSize, 1,
                    FootprintRowTexels, uint64_t(index) * PatchStride);
                const RenderBox srcBox(int32_t(left), int32_t(top),
                    int32_t(left + PatchSize), int32_t(top + PatchSize));
                worker->commandList->copyTextureRegion(dstLocation, srcLocation, 0, 0, 0, &srcBox);
            }
        }

        // Hand the target back in the layout the rest of the frame expects.
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS,
            RenderTextureBarrier(texture, RenderTextureLayout::SHADER_READ));

        slot.queued = true;
        frameIndex++;
        return true;
    }

    StereoDepthSampler::Sample StereoDepthSampler::fetch() {
        Sample result;
        if (frameIndex < SlotCount) {
            return result;
        }

        // The oldest slot in the ring: written SlotCount frames ago.
        Slot &slot = slots[frameIndex % SlotCount];
        if (!slot.queued || (slot.buffer == nullptr)) {
            return result;
        }

        const RenderRange readRange(0, PatchBufferSize);
        const float *data = reinterpret_cast<const float *>(slot.buffer->map(0, &readRange));
        if (data == nullptr) {
            return result;
        }

        // dynamic3d 4.1 - a spatial median, never a min. One stray texel from a
        // particle or a sliver of geometry at the patch edge must not be able to
        // yank the result.
        // The NEAR statistic is computed in two levels, per patch and then across
        // patches, rather than by pooling every texel into one percentile.
        //
        // Pooling does not work. A low percentile over all ~46k samples needs an
        // object to cover several percent of everything sampled before it moves
        // the result, so a wall is found easily while a small nearby object -
        // a crate, a ledge corner - sits inside the discarded tail and the
        // percentile lands on the surface behind it. Pooling throws away the
        // spatial information that would have caught it.
        //
        // Instead: a low percentile WITHIN each patch (robust against single
        // stray texels from particles, but sensitive to anything filling a
        // meaningful part of one small patch), then a low order statistic ACROSS
        // the patch results. A small object only has to dominate one patch.
        const uint32_t centreIndex = ((PatchRows / 2) * PatchCols) + (PatchCols / 2);
        std::vector<float> patchSamples;
        std::vector<float> patchNears;
        std::vector<float> centreSamples;
        patchSamples.reserve(PatchSize * PatchSize);
        patchNears.reserve(PatchCount);
        centreSamples.reserve(PatchSize * PatchSize);

        for (uint32_t patch = 0; patch < PatchCount; patch++) {
            const float *patchData = data + ((size_t(patch) * PatchStride) / DepthTexelSize);
            patchSamples.clear();
            for (uint32_t y = 0; y < PatchSize; y++) {
                const float *row = patchData + (size_t(y) * FootprintRowTexels);
                for (uint32_t x = 0; x < PatchSize; x++) {
                    const float d = row[x];
                    // Discard cleared/far texels and anything outside a sane
                    // device range; those are sky or an untouched buffer, not a
                    // surface the player is looking at.
                    if ((d > 0.0f) && (d < 1.0f)) {
                        patchSamples.push_back(d);
                    }
                }
            }

            if (patch == centreIndex) {
                centreSamples = patchSamples;
            }

            // Require enough coverage that the percentile means something.
            if (patchSamples.size() < 32) {
                continue;
            }

            const size_t patchNearIndex = patchSamples.size() / 4; // 25th percentile
            std::nth_element(patchSamples.begin(), patchSamples.begin() + patchNearIndex, patchSamples.end());
            patchNears.push_back(patchSamples[patchNearIndex]);
        }

        slot.buffer->unmap();

        if (patchNears.empty()) {
            return result;
        }

        if (!centreSamples.empty()) {
            const size_t centreMid = centreSamples.size() / 2;
            std::nth_element(centreSamples.begin(), centreSamples.begin() + centreMid, centreSamples.end());
            result.medianDeviceDepth = centreSamples[centreMid];
        }

        // Smaller device depth is closer under a standard depth range, so the
        // nearest patch is the smallest of the per-patch values. Take the second
        // smallest when there are enough patches, so one anomalous patch cannot
        // drive the whole loop on its own.
        const size_t acrossIndex = (patchNears.size() >= 4) ? 1 : 0;
        std::nth_element(patchNears.begin(), patchNears.begin() + acrossIndex, patchNears.end());
        result.nearestDeviceDepth = patchNears[acrossIndex];
        result.valid = true;
        return result;
    }

    void StereoAutoConvergence::reset() {
        historyCount = 0;
        historyCursor = 0;
        cutCandidateFrames = 0;
        zEma = -1.0f;
        invConvSmoothed = -1.0f;
    }

    uint32_t StereoAutoConvergence::update(float nearestViewZ, uint32_t manualConvergenceTenths, uint32_t separationSlider,
                                           int32_t comfortTarget, bool sceneWantsLowConvergence) {
        // Same unit conversions the projection processor uses.
        const float separation = float(separationSlider) * (0.10f / 50.0f);
        const float manualConv = float(manualConvergenceTenths) * 2.0f;

        // Guard on separation, not on any projection term: at zero separation
        // there is no disparity to bound and the solve would divide by zero.
        if ((nearestViewZ <= 0.0f) || (separation <= 0.0f) || (manualConv <= 0.0f)) {
            return manualConvergenceTenths;
        }

        // dynamic3d 4.2 - a temporal median before the EMA, catching
        // single-frame spikes that the spatial statistic alone lets through.
        history[historyCursor] = nearestViewZ;
        historyCursor = (historyCursor + 1) % HistorySize;
        historyCount = (historyCount < HistorySize) ? (historyCount + 1) : HistorySize;

        float sorted[HistorySize];
        for (uint32_t i = 0; i < historyCount; i++) {
            sorted[i] = history[i];
        }
        std::sort(sorted, sorted + historyCount);
        const float zMedian = sorted[historyCount / 2];

        if (zEma <= 0.0f) {
            zEma = zMedian;
        }
        else {
            // Relative deadband, not absolute: a fixed threshold is
            // simultaneously too twitchy up close and too sluggish at range.
            const float rel = std::fabs(zMedian - zEma) / zEma;

            // A camera CUT, not motion. The EMA is tuned for continuous movement
            // and needs many frames to cross a large step, which is precisely
            // when a newly-close framing reads worst - the reason cutscenes felt
            // wrong even with the loop running.
            //
            // But it has to PERSIST to count. The first version snapped on any
            // single frame past the threshold, and once the near statistic became
            // sensitive enough to catch small objects that happened constantly in
            // ordinary play - an object entering one patch halves the reported
            // depth - so it snapped over and over and read as twitching. A real
            // cut stays changed for as long as the new shot lasts.
            if (rel > 1.5f) {
                cutCandidateFrames++;
            }
            else {
                cutCandidateFrames = 0;
            }

            if (cutCandidateFrames >= 3) {
                zEma = zMedian;
                invConvSmoothed = -1.0f;
                cutCandidateFrames = 0;
            }
            else if (rel > 0.02f) {
                // Asymmetric on purpose. Something moving toward the camera has
                // to be tracked quickly for comfort; something receding should
                // relax slowly, or convergence chases every small recession and
                // reads as swimmy.
                const float alpha = (zMedian < zEma) ? 0.15f : 0.05f;
                zEma += (zMedian - zEma) * alpha;
            }
        }

        // dynamic3d 4.3 - pop-out of the nearest object at a given convergence,
        // as a fraction of eye width, positive when nearer than the screen.
        // Comfort budget, expressed directly: the slider IS the permitted pop-out
        // disparity in thousandths of screen width, so 5 means 0.5%. That makes
        // 0 a real setting rather than an arbitrary floor - it puts the screen
        // plane exactly on the nearest object, the most protective the loop can
        // be. The previous mapping bottomed out at 0.005 with the slider at 0,
        // which left no room for anyone who wanted it tighter.
        //
        // A classified close-framed scene tightens this further rather than
        // overriding convergence outright, so the two mechanisms compose.
        float targetDisparity = float(std::clamp(comfortTarget, -50, 60)) * 0.001f;
        if (sceneWantsLowConvergence) {
            targetDisparity *= 0.45f;
        }
        constexpr float NearClamp = 4.0f;
        const float halfSeparation = separation * 0.5f;
        const float disparityAtManual = halfSeparation * ((manualConv / zEma) - 1.0f);

        float convTarget = manualConv;
        if (disparityAtManual > targetDisparity) {
            convTarget = zEma * (1.0f + ((2.0f * targetDisparity) / separation));
            convTarget = std::min(convTarget, manualConv);
        }
        // A negative comfort target intentionally drives convergence below the
        // nearest depth. Bound it relative to that depth as well as absolutely,
        // or at small separations the solve can push convergence to nothing.
        convTarget = std::max(convTarget, zEma * 0.25f);
        convTarget = std::max(convTarget, NearClamp);

        // dynamic3d 4.4 - smooth the RECIPROCAL. Disparity is linear in
        // 1/convergence, so a plain lerp of convergence accelerates
        // perceptually as it gets small and lunges the last part of the way.
        const float targetInv = 1.0f / convTarget;
        if (invConvSmoothed <= 0.0f) {
            invConvSmoothed = targetInv;
        }
        else {
            invConvSmoothed += (targetInv - invConvSmoothed) * 0.06f;
        }

        const float applied = std::min(1.0f / invConvSmoothed, manualConv);
        const uint32_t appliedTenths = uint32_t(std::lround(applied / 2.0f));
        return (appliedTenths < 1u) ? 1u : appliedTenths;
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
