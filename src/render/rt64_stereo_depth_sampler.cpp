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
    static constexpr uint32_t PatchBufferSize = PatchStride * StereoDepthSampler::TotalPatchCount;

    bool StereoDepthSampler::submit(RenderWorker *worker, RenderTarget *depthTarget,
                                    uint32_t contentOriginX, uint32_t contentWidth,
                                    int32_t aimCenterX, int32_t aimCenterY,
                                    int32_t aimOffsetMin, int32_t aimOffsetMax) {
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

        // Spread the grid over most of the frame, stopping short of the edges
        // where HUD-hugging geometry would drag the near statistic down.
        //
        // The vertical margins are deliberately ASYMMETRIC. Geometry that is
        // permanently close to the camera and must not be chased - a mounted
        // cannon filling the bottom of a minigame's frame, and viewmodel-like
        // props generally - sits low, because that is where a game puts things
        // attached to the viewer. The top of the frame is sky and distant
        // scenery, which is exactly what the loop wants to see. A symmetric box
        // has to choose between including that clutter and discarding the useful
        // half, and including it is what made convergence oscillate: the cannon
        // is near enough to win the near statistic but marginal enough to drop
        // in and out of it, so the loop flipped between two stable solves every
        // few frames.
        //
        // The horizontal margins are measured against the VISIBLE span, not the
        // whole target. A wider-than-16:9 eye is cropped to its centred 16:9
        // slice before the viewer sees it, so on a 32:9 desktop roughly a
        // quarter of the target at each side is rendered and thrown away.
        // Spreading the grid across all of it lets scenery well outside the
        // frame win the near statistic and pull convergence in for no visible
        // reason.
        const RenderTextureCopyLocation srcLocation = RenderTextureCopyLocation::Subresource(texture, 0);
        // The available width is checked rather than assumed: an origin close to
        // the right edge would otherwise produce a zero or inverted span.
        const uint32_t spanX = std::min(contentOriginX, targetWidth - PatchSize);
        const uint32_t spanAvail = targetWidth - spanX;
        const uint32_t spanW = std::min(std::max(contentWidth, PatchSize), spanAvail);
        const uint32_t usableW = (spanW * (100 - RoiMarginLeftPercent - RoiMarginRightPercent)) / 100;
        const uint32_t usableH = (targetHeight * (100 - RoiMarginTopPercent - RoiMarginBottomPercent)) / 100;
        const uint32_t originX = spanX + ((spanW * RoiMarginLeftPercent) / 100);
        const uint32_t originY = (targetHeight * RoiMarginTopPercent) / 100;

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

        // Aim strip: AimPatchCount tight patches spread across the range of
        // buffer positions the aim point could resolve to.
        //
        // Each patch stays 32 texels wide - narrow is the point, since every
        // texel of width is a texel of something the player is not pointing at.
        // The STRIP is wide, but it is never pooled: the caller judges each
        // patch separately and keeps exactly one.
        //
        // The offsets are recorded on the slot because they are read back
        // several frames later, by which time the separation slider - which
        // sets the span - may have moved.
        const int32_t aimSpan = std::max(aimOffsetMax - aimOffsetMin, 1);
        for (uint32_t i = 0; i < AimPatchCount; i++) {
            const int32_t offset = aimOffsetMin + int32_t((int64_t(aimSpan) * i) / int64_t(AimPatchCount - 1));
            slot.aimOffsets[i] = offset;

            int32_t left = (aimCenterX + offset) - int32_t(PatchSize / 2);
            int32_t top = aimCenterY - int32_t(PatchSize / 2);
            left = std::max(0, std::min(left, int32_t(targetWidth - PatchSize)));
            top = std::max(0, std::min(top, int32_t(targetHeight - PatchSize)));

            const RenderTextureCopyLocation dstLocation = RenderTextureCopyLocation::PlacedFootprint(
                slot.buffer.get(), RenderFormat::D32_FLOAT, FootprintRowTexels, PatchSize, 1,
                FootprintRowTexels, uint64_t(AimPatchIndex + i) * PatchStride);
            const RenderBox srcBox(left, top, left + int32_t(PatchSize), top + int32_t(PatchSize));
            worker->commandList->copyTextureRegion(dstLocation, srcLocation, 0, 0, 0, &srcBox);
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
        // The aim depth comes from the single aim patch, and takes a MEDIAN of
        // it. The "reticle sits at a depth that did not match the thing it was
        // over, reading as one eye aiming at the target and the other not"
        // symptom that a wider band was once introduced to cure was the per-eye
        // sampling offset, not a window too narrow to be representative; it is
        // fixed at the source now.
        std::vector<float> patchSamples;
        std::vector<float> patchNears;
        patchSamples.reserve(PatchSize * PatchSize);
        patchNears.reserve(PatchCount);

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

            // Require enough coverage that the percentile means something.
            if (patchSamples.size() < 32) {
                continue;
            }

            const size_t patchNearIndex = patchSamples.size() / 4; // 25th percentile
            std::nth_element(patchSamples.begin(), patchSamples.begin() + patchNearIndex, patchSamples.end());
            patchNears.push_back(patchSamples[patchNearIndex]);
        }

        // Aim strip: a MEDIAN per patch. Same validity rule as the grid patches -
        // enough covered texels for the statistic to mean something - but the
        // middle of the distribution rather than its near tail, because what the
        // crosshair should sit on is whatever FILLS a window.
        //
        // A near-percentile here (this was a ~1.5% one, effectively a minimum)
        // reports the nearest surface anywhere in a window, so a sliver of
        // something closer clipping one edge takes the whole answer.
        //
        // Read BEFORE the grid's early-out below, so the crosshair does not
        // depend on the convergence loop having a usable frame. The patches are
        // deliberately kept separate; pooling them would be the wide-window bias
        // all over again.
        for (uint32_t i = 0; i < AimPatchCount; i++) {
            const float *aimData = data + ((size_t(AimPatchIndex + i) * PatchStride) / DepthTexelSize);
            patchSamples.clear();
            for (uint32_t y = 0; y < PatchSize; y++) {
                const float *row = aimData + (size_t(y) * FootprintRowTexels);
                for (uint32_t x = 0; x < PatchSize; x++) {
                    const float d = row[x];
                    if ((d > 0.0f) && (d < 1.0f)) {
                        patchSamples.push_back(d);
                    }
                }
            }

            result.aim[i].offsetTexels = slot.aimOffsets[i];
            result.aim[i].sampleCount = uint32_t(patchSamples.size());
            if (patchSamples.size() >= 32) {
                const size_t midIndex = patchSamples.size() / 2;
                std::nth_element(patchSamples.begin(), patchSamples.begin() + midIndex, patchSamples.end());
                result.aim[i].deviceDepth = patchSamples[midIndex];
                result.aim[i].valid = true;
            }
        }

        slot.buffer->unmap();

        if (patchNears.empty()) {
            return result;
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

    // Mirrors ConvergenceWorldPerUnit in the projection processor. The solve
    // below is expressed in the clip-space parameterization and silently retunes
    // itself if the two drift apart, so they are commented on both sides rather
    // than only here.
    static constexpr float ConvergenceWorldPerUnit = 0.2f;

    uint32_t StereoAutoConvergence::update(float nearestViewZ, uint32_t manualConvergenceHundredths, uint32_t separationSlider,
                                           int32_t comfortTarget, bool sceneWantsLowConvergence) {
        // Same unit conversions the projection processor uses.
        const float separation = float(separationSlider) * (0.10f / 50.0f);
        const float manualConv = float(manualConvergenceHundredths) * ConvergenceWorldPerUnit;

        // Guard on separation, not on any projection term: at zero separation
        // there is no disparity to bound and the solve would divide by zero.
        if ((nearestViewZ <= 0.0f) || (separation <= 0.0f) || (manualConv <= 0.0f)) {
            return manualConvergenceHundredths;
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
            // Measured as a RATIO, and only in the APPROACH direction.
            //
            // The previous form tested rel, i.e. |z - ema| / ema, against 1.5.
            // That expression is bounded above by 1 whenever z is smaller than
            // ema - it is 1 - z/ema there, which cannot reach 1 however close
            // the object gets - so a threshold of 1.5 was UNREACHABLE in the
            // approach direction and the snap only ever fired for cuts to a
            // further view. Backwards: a closeup is the case the snap exists
            // for. dynamic3d 4.2 calls this out as a trap that survives review
            // because the asymmetry is invisible until the algebra is written
            // out.
            //
            // Making the ratio symmetric is the obvious correction and is its
            // own bug (dynamic3d 4.5). Anything held close to the camera and
            // moving - a pickup animation, an NPC leaning in - swings the near
            // statistic past a 2.5x ratio in BOTH directions within a second or
            // two, and a symmetric detector answers every crossing with a hard
            // snap and a smoother reset: in, out, in. That reads as the image
            // thrashing, which alarms a viewer more than the sluggishness it
            // replaced.
            //
            // So the snap is the approach half only. Recession has nothing to
            // protect against - convergence sitting nearer than the scene needs
            // costs only positive parallax, which is bounded by separation - so
            // it eases out through the EMA below at the slow alpha. Dropping
            // the recede half also breaks the alternation on its own: after an
            // approach snap the EMA sits at the near value, so the return trip
            // cannot clear the ratio a second time. The price is that a genuine
            // cut from a closeup to a vista eases over about a second instead of
            // snapping, which is the comfortable direction to be wrong in.
            //
            // Tested against the RAW sample rather than the median: the median
            // spans HistorySize frames and needs half of them before it begins
            // to cross a step at all, and that is latency the smooth path wants
            // and the cut path must not pay.
            //
            // It still has to PERSIST. Snapping on a single frame past the
            // threshold reads as twitching, because the near statistic is
            // sensitive enough that an object entering one patch can halve the
            // reported depth during ordinary play. Two frames rather than
            // three: the readback already runs several frames behind the GPU,
            // so every extra confirmation frame is one more spent at the wrong
            // convergence.
            const float approachRatio = (nearestViewZ < zEma) ? (zEma / nearestViewZ) : 1.0f;
            if (approachRatio > 2.5f) {
                cutCandidateFrames++;
            }
            else {
                cutCandidateFrames = 0;
            }

            if (cutCandidateFrames >= 2) {
                // Land on the median of the last three RAW samples. The long
                // median still holds mostly pre-cut values, so snapping to it
                // would only go part of the way; a single raw sample puts the
                // whole shot at the mercy of one frame.
                float recent[3];
                for (uint32_t i = 0; i < 3; i++) {
                    recent[i] = history[(historyCursor + HistorySize - 1 - i) % HistorySize];
                }

                std::sort(recent, recent + 3);
                zEma = recent[1];
                invConvSmoothed = -1.0f;
                cutCandidateFrames = 0;
            }
            else if (rel > 0.02f) {
                // Asymmetric on purpose. Something moving toward the camera has
                // to be tracked quickly for comfort; something receding should
                // relax slowly, or convergence chases every small recession and
                // reads as swimmy.
                const float alpha = (zMedian < zEma) ? 0.25f : 0.05f;
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
            // Asymmetric for the same reason the depth EMA is, and not redundant
            // with it: the EMA governs how fast the loop believes the scene
            // changed, this governs how fast the picture follows that belief.
            // Closing the gap protects comfort and should be prompt, while
            // easing back out has nothing to protect against and reads as the
            // image drifting if it hurries. A LARGER target reciprocal is a
            // NEARER convergence, so that is the direction to hurry.
            const float invAlpha = (targetInv > invConvSmoothed) ? 0.14f : 0.06f;
            invConvSmoothed += (targetInv - invConvSmoothed) * invAlpha;
        }

        const float applied = std::min(1.0f / invConvSmoothed, manualConv);
        // Rounded back into bridge units. This is the only place the loop's
        // continuous solve gets quantised, which is why the unit is a hundredth
        // of a slider step rather than a tenth - see ConvergenceWorldPerUnit.
        const uint32_t appliedUnits = uint32_t(std::lround(applied / ConvergenceWorldPerUnit));
        // Never 0: that value is the sentinel for "the loop has nothing to say
        // and the manual setting stands". The solve's own floor (NearClamp) is
        // far above this, so this is a degenerate-input backstop only.
        return (appliedUnits < 1u) ? 1u : appliedUnits;
    }

    float stereoDeviceDepthToViewZ(float deviceDepth, float projM22, float projM32,
                                   float vpScaleZ, float vpTranslateZ) {
        if ((deviceDepth <= 0.0f) || (deviceDepth >= 1.0f)) {
            return -1.0f;
        }

        // For a standard perspective projection, clip.z = view.z * m[2][2] + m[3][2]
        // and clip.w = -view.z, so ndc.z = -(m[2][2] + m[3][2] / view.z) and
        //     view.z = m[3][2] / (-ndc.z - m[2][2]).
        //
        // The device depth is NOT ndc.z, and undoing that takes TWO layers.
        //
        // First the NDC convention: the buffer holds [0,1] while the N64's
        // GL-style projection produces ndc.z over [-1,1]. Feeding the device
        // value in directly - which this did - overestimates distance by 1.75x
        // to 1.9x over DK64's depth range, which put the crosshair well behind
        // whatever it was aimed at and made the convergence loop think
        // everything was further away than it is.
        //
        // Second the VIEWPORT's own depth scale and translate, which dynamic3d
        // 4.1 warns is rarely the clean half the first layer makes it look
        // like. On the N64 it is a G_MAXZ fixed-point value - 32704/65536 =
        // 0.499023 on a measured title - and that 0.2% is not ignorable,
        // because it perturbs the term that survives after almost all of m22
        // cancels. Assuming exactly 0.5 leaves near objects correct and reads
        // everything beyond progressively CLOSER than it is: roughly -0.3% at
        // 10 units, -4.5% at 100, -20% at 1000. The error signature is a
        // crosshair that tracks fine up close but sits short of mid-range
        // targets, and auto-convergence quietly over-pulling.
        //
        // Fall back to the nominal remap only when no viewport was published,
        // which is the "this frame had none" case rather than a measured value.
        const float ndcZ = (vpScaleZ > 1e-6f)
            ? ((deviceDepth - vpTranslateZ) / vpScaleZ)
            : ((2.0f * deviceDepth) - 1.0f);
        const float denom = -ndcZ - projM22;
        if (std::fabs(denom) < 1e-6f) {
            return -1.0f;
        }

        const float viewZ = projM32 / denom;
        return (viewZ > 0.0f) ? viewZ : -viewZ;
    }
}
