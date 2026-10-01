//
// RT64
//

#include "rt64_projection_processor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "../include/rt64_extended_gbi.h"
#include "common/rt64_math.h"
#include "hle/rt64_workload_queue.h"
#include "rt64_render_target.h"

namespace RT64 {
    // The world projection's two depth terms, published from the render thread
    // and read by the depth sampler on the same thread a frame later. Stored as
    // raw bits so the pair can be atomics without needing a lock.
    static std::atomic<uint32_t> worldDepthM22Bits{0};
    static std::atomic<uint32_t> worldDepthM32Bits{0};
    static std::atomic<uint32_t> worldDepthVpScaleZBits{0};
    static std::atomic<uint32_t> worldDepthVpTranslateZBits{0};
    static std::atomic<bool> worldDepthTermsValid{false};

    static uint32_t floatToBits(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    static float bitsToFloat(uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    void stereoPublishWorldDepthTerms(float m22, float m32, float vpScaleZ, float vpTranslateZ) {
        worldDepthM22Bits.store(floatToBits(m22), std::memory_order_relaxed);
        worldDepthM32Bits.store(floatToBits(m32), std::memory_order_relaxed);
        worldDepthVpScaleZBits.store(floatToBits(vpScaleZ), std::memory_order_relaxed);
        worldDepthVpTranslateZBits.store(floatToBits(vpTranslateZ), std::memory_order_relaxed);
        worldDepthTermsValid.store(true, std::memory_order_relaxed);
    }

    bool stereoGetWorldDepthTerms(float &m22, float &m32, float &vpScaleZ, float &vpTranslateZ) {
        if (!worldDepthTermsValid.load(std::memory_order_relaxed)) {
            return false;
        }
        m22 = bitsToFloat(worldDepthM22Bits.load(std::memory_order_relaxed));
        m32 = bitsToFloat(worldDepthM32Bits.load(std::memory_order_relaxed));
        vpScaleZ = bitsToFloat(worldDepthVpScaleZBits.load(std::memory_order_relaxed));
        vpTranslateZ = bitsToFloat(worldDepthVpTranslateZBits.load(std::memory_order_relaxed));
        return true;
    }

    inline void adjustProjectionMatrix(interop::float4x4 &matrix, const float aspectRatioScale) {
        matrix[0][0] *= aspectRatioScale;
        matrix[1][0] *= aspectRatioScale;
        matrix[2][0] *= aspectRatioScale;
        matrix[3][0] *= aspectRatioScale;
    }

    // Goemon64Recomp3D-specific transform IDs: see patches/transform_ids.h.
    // The world (gameplay) projection receives full per-eye stereo: projection
    // off-axis (inter-eye disparity) + view translation (depth-dependent parallax).
    // The skybox receives only the projection off-axis with no view shift, which
    // gives it the maximum positive parallax of an object at infinity — exactly
    // what we want so it sits well behind the world geometry.
    // The HUD projection is shifted to a constant depth set by the user slider.
    static constexpr uint32_t GOEMON_PROJECTION_GAMEPLAY_TRANSFORM_ID = 0x00001000;
    static constexpr uint32_t GOEMON_PROJECTION_SKYBOX_TRANSFORM_ID   = 0x00001001;
    static constexpr uint32_t GOEMON_PROJECTION_HUD_TRANSFORM_ID      = 0x00001004;

    // Goemon's decomp is too thin to patch the projection-setup function
    // (func_80017D8C_1898C) the way Banjo patches viewport_setRenderPerspectiveMatrix,
    // so we can't emit gEXMatrixGroup with an explicit GAMEPLAY id on the world
    // perspective. Instead, treat *all* untagged perspective projections
    // (matrixId == G_EX_ID_AUTO, the RT64 default for projections that never
    // see a gEXMatrixGroup command) as gameplay. The world's the only common
    // perspective Goemon emits, so this matches reality. Trade-off: if the
    // game later emits an untagged perspective FMV or cutscene, it'd also get
    // the world stereo treatment instead of falling through to mono — revisit
    // once those scenes are identified.
    static bool isStereoProjectionId(uint32_t matrixId) {
        return (matrixId == GOEMON_PROJECTION_GAMEPLAY_TRANSFORM_ID) ||
               (matrixId == GOEMON_PROJECTION_SKYBOX_TRANSFORM_ID) ||
               (matrixId == G_EX_ID_AUTO);
    }

    static bool isStereoViewShiftProjectionId(uint32_t matrixId) {
        return (matrixId == GOEMON_PROJECTION_GAMEPLAY_TRANSFORM_ID) ||
               (matrixId == G_EX_ID_AUTO);
    }

    // Goemon draws some HUD items as small 3D models under a perspective
    // projection of their own, untagged like the world's, so by ID they would
    // get the world's stereo (shear plus view shift) and sit in the scene at
    // whatever depth their private camera puts them. What gives them away is
    // that camera: a fixed one a few dozen units from the origin, looking
    // straight down -Z - measured at a translation of (0, 0, -21) with a
    // rotation within 0.1 degrees of identity, near 1 and far ~1000 - where a
    // world camera is wherever the scene is, pointed wherever the shot needs.
    //
    // The caller also requires that the projection is not the frame's main
    // world projection (the one with the most draw calls), so the world can
    // never be taken for HUD even if its camera happens to be axis-aligned near
    // the origin.
    static bool viewMatrixLooksLikeHudCamera(const interop::float4x4 &viewMatrix) {
        constexpr float AxisTolerance = 0.01f;           // ~8 degrees off -Z
        constexpr float MaxCameraDistance = 200.0f;      // game units
        const float tx = float(viewMatrix[3][0]);
        const float ty = float(viewMatrix[3][1]);
        const float tz = float(viewMatrix[3][2]);
        return (std::abs(float(viewMatrix[0][0]) - 1.0f) < AxisTolerance) &&
               (std::abs(float(viewMatrix[1][1]) - 1.0f) < AxisTolerance) &&
               (std::abs(float(viewMatrix[2][2]) - 1.0f) < AxisTolerance) &&
               ((tx * tx + ty * ty + tz * tz) < (MaxCameraDistance * MaxCameraDistance));
    }

    // HUD projections that should receive the user's configured constant
    // stereo depth (no view shift). Currently just the main HUD; pause menu /
    // dialog / etc. are not tagged yet and intentionally fall through to mono.
    static bool isStereoHudProjectionId(uint32_t matrixId) {
        return (matrixId == GOEMON_PROJECTION_HUD_TRANSFORM_ID);
    }

    // dynamic3d 1.3 - the clip-space stereo parameterization.
    //
    // `separation` IS the stereo knob: the per-eye projection shear is exactly
    // this value, with no FoV term and no convergence term folded into it.
    // Because the shear is the NDC x offset at infinity, the number means
    // something the user can see:
    //
    //     total background disparity = separation x screen width
    //
    // The 0..50 slider maps linearly onto 0..0.10 of screen width, so each
    // point is 0.2% of the screen. The default (10 -> 0.020) is a comfortable
    // 2%, and the top of the range is roughly the divergence ceiling - where
    // background disparity reaches an adult IPD, the eyes are forced outward,
    // and no amount of practice can fuse it. That ceiling is IPD / screen
    // width: about 0.105 on a 27-inch 16:9 monitor and proportionally lower on
    // anything bigger (~0.05 on a 55-inch TV). Values past it stay reachable
    // because the physical screen size is not visible from here.
    //
    // This replaces the previous world-units form, where separation was a
    // game-unit eye offset and the shear was (sep / 2 / conv) * m[0][0]. That
    // form needed a per-game compression constant (Goemon's world units are
    // ~50x smaller than Banjo's, hence the old 0.02 multiplier) purely to get
    // the slider into a usable range, and it rode on the live projection scale,
    // so the depth effect quietly tracked the output aspect ratio - RT64
    // narrows m[0][0] by 1/aspectRatioScale for widescreen, so about a quarter
    // of the effect was lost going from 16:9 to 21:9. Clip space is invariant
    // to both, and to any in-game FoV change, by construction. That also means
    // the slider now means the same thing here as it does in the sibling ports,
    // so a value that is comfortable in one is comfortable in the others.
    //
    // The slider-to-fraction constant itself lives in the header as
    // StereoSeparationPerSlider, because the texture-rectangle path in
    // rt64_workload_queue.cpp has to use the same number.
    float stereoSeparationFraction(uint32_t separationSlider) {
        return static_cast<float>(separationSlider) * StereoSeparationPerSlider;
    }

    // Convergence stays what it always was: the distance at which geometry sits
    // exactly on the screen plane. It is a DISTANCE, not a disparity, so it
    // needs no calibration against the projection and no FoV term - and unlike
    // separation it is inherently per-game, because it is measured in the
    // game's own units. Goemon's cameras sit close to the action, so the useful
    // band is roughly 10..100 units and the slider is carried straight through
    // in game units rather than scaled.
    static float stereoConvergenceWorld(uint32_t convergenceSlider) {
        return static_cast<float>(convergenceSlider);
    }

    // The convergence actually applied: the slider, or the depth loop's pull-in
    // when it has one. The loop's value is already bounded by the slider on the
    // way out, but take the min here too so this can never push the screen
    // plane further than the user asked for.
    static float stereoEffectiveConvergence(uint32_t convergenceSlider, float convergenceAuto) {
        const float manual = stereoConvergenceWorld(convergenceSlider);
        return (convergenceAuto > 0.0f) ? std::min(convergenceAuto, manual) : manual;
    }

    // TEMPORARY convergence investigation: per process() call, how many
    // perspective projections drew, how many received the view shift (the only
    // place convergence acts), and the convergence they got.
    static uint32_t debugPerspectiveCount = 0;
    static uint32_t debugViewShiftCount = 0;
    static uint32_t debugOtherIdCount = 0;
    static uint32_t debugLastOtherId = 0;
    static uint32_t debugHudCameraCount = 0;
    static float debugLastConvergence = 0.0f;

    // dynamic3d 1.1 - the shear is the knob.
    //
    // The old cap on the shear term is gone along with the world-units form. It
    // existed because sep / (2 * conv) genuinely grows without bound as
    // convergence approaches zero; the shear is now just `separation`, which
    // the slider range already bounds, so there is nothing left for the cap to
    // protect against (dynamic3d 2.4).
    static void applyStereoOffAxis(interop::float4x4 &projMatrix, StereoEye eye, uint32_t separationSlider) {
        if (eye == StereoEye::None) {
            return;
        }
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        // Adds an asymmetric horizontal shift to the projection's principal point.
        // Equivalent to rebuilding the frustum with [L,R] = [-horFov+offset, horFov+offset].
        projMatrix[2][0] += eyeSign * stereoSeparationFraction(separationSlider);
    }

    // The per-eye NDC x offset for a HUD element at the configured HUD depth,
    // before the eye sign. Shared with the texture-rectangle path in
    // rt64_workload_queue.cpp, which has to produce exactly the same number for
    // 2D draws that never pass through a projection matrix.
    //
    // One offset for every kind of HUD draw. It is an NDC shift either way: on
    // an orthographic projection m[3][0] += K moves NDC x by K directly, and on
    // a perspective one m[2][0] += K moves it by K after the divide, at every
    // depth. The perspective path used to be scaled by a reference projection
    // term instead (0.975 against 2.75), so a HUD item drawn as a 3D model
    // moved about a third as far as the text and panels beside it, and the
    // pieces of one dialog pulled apart in depth.
    float stereoHudNdcOffset(uint32_t hudDepthSlider, uint32_t separationSlider) {
        const float centered = (static_cast<float>(hudDepthSlider) - 50.0f) / 50.0f; // -1..+1
        // The slider spans exactly the depths a HUD can sensibly take, in units
        // of the scene's own disparity. At 0 the shift is the full separation -
        // the depth of infinity, where the sky sits - so the HUD can never be
        // pushed behind the world's farthest point; at 50 it is on the screen
        // plane; at 100 it pops out as far as infinity sits behind.
        //
        // dynamic3d 5.2: a layer at a fixed multiple of the convergence
        // distance shifts by separation * (1/factor - 1) - proportional to
        // separation, independent of convergence - so this is that mapping with
        // the factor running from infinity (0) through 1 (50) to 1/2 (100).
        // Scaling by separation also makes the HUD go properly flat at zero
        // separation.
        //
        // This replaces 0.04 * 2.75 * separation / 0.02 at full travel - 5.5x
        // the separation - which passed infinity a few points below 50 and was
        // stopped by the clamp below around 40, while above 60 it popped out
        // further than the background sits behind. Only the middle fifth of the
        // slider was usable; now all of it is, at a fifth of the step size.
        //
        // Negated so slider > 50 produces pop-out (negative parallax) and
        // slider < 50 produces push-back (positive parallax).
        const float hudOffset = -centered * stereoSeparationFraction(separationSlider);
        const float ndcOffset = hudOffset;

        // dynamic3d 6.1 - pop-out is not divergence, so the two directions do
        // not get the same limit. Behind the screen plane the constraint is
        // physical: uncrossed disparity past an IPD forces the eyes outward and
        // cannot be fused. In front of it the eyes converge inward and there is
        // nothing to protect against, so reusing the behind-limit would only
        // clip valid pop-out. Branch on hudOffset, which is eye-independent -
        // the sign of the APPLIED shift encodes which eye, not which side of
        // the screen plane the element is on.
        const float ndcLimit = (hudOffset > 0.0f) ? StereoBehindNdcLimit : StereoPopOutNdcLimit;
        return std::max(-ndcLimit, std::min(ndcLimit, ndcOffset));
    }

    // Apply a constant per-eye horizontal shift to a HUD/UI projection matrix so
    // it sits at a user-selected stereo depth instead of flat on the screen.
    // hudDepthSlider 0..100: 50 = screen plane (no shift), below = push behind
    // the screen (positive parallax), above = pop out (negative parallax).
    //
    // dynamic3d 5.2: a layer parked at a fixed multiple of the convergence
    // distance has a shift of separation * (1/factor - 1) - proportional to
    // separation, and independent of convergence. Scaling by separation is what
    // makes HUD depth track the depth knob. The per-projection-type constants
    // below are that (1/factor - 1) mapping; they keep their empirically tuned
    // values, so at the default separation the relative depths of the HUD and
    // the world are unchanged.
    //
    // Perspective and orthographic projections need different elements:
    //   - Perspective: m[2][0] gets the principal-point offset, scaled by the
    //     REFERENCE projection term rather than the live one so the depth holds
    //     across aspect ratios. After the perspective divide this becomes a
    //     constant NDC shift.
    //   - Orthographic: there is no perspective divide, so we add a direct NDC
    //     shift via m[3][0], with the sign flipped so the slider pushes both
    //     projection types the same way.
    float stereoSnapNdcToPixel(float ndcOffset, float halfViewportWidth, float gridPixels) {
        if ((halfViewportWidth < 1.0f) || (gridPixels <= 0.0f)) {
            return ndcOffset;
        }

        // One output pixel is 1 / halfViewportWidth of NDC; one grid step is
        // gridPixels of them. Symmetric rounding, so the two eyes' opposite
        // shifts stay exactly opposite.
        const float ndcPerStep = gridPixels / halfViewportWidth;
        return std::round(ndcOffset / ndcPerStep) * ndcPerStep;
    }

    // Half the output width in pixels of the viewport a framebuffer pair is
    // drawn into: the colour image width times the pair's resolution scale,
    // derived as the workload queue does for the pair (including halving it
    // for the much taller hi-res targets) and as the framebuffer renderer then
    // sizes its viewport.
    // Also returns, in *nativePixel, the pair's resolution scale - the width in
    // output pixels of one native pixel, the block RasterPS snaps 2D texture
    // coordinates to (see stereoSnapNdcToPixel).
    static float stereoHalfViewportPixels(const Workload &workload, const FramebufferPair &fbPair, hlslpp::float2 resolutionScale,
                                          float *nativePixel) {
        const uint32_t nativeWidth = fbPair.colorImage.width;
        *nativePixel = 0.0f;
        if (nativeWidth == 0) {
            return 0.0f;
        }

        hlslpp::float2 scale = resolutionScale;
        const uint32_t heightThreshold = (workload.viFbSize[1] > 0) ? ((workload.viFbSize[1] * 3) / 2) : 360;
        if ((uint32_t(fbPair.drawColorRect.bottom(true)) >= heightThreshold) && (float(scale[1]) >= 2.0f)) {
            scale = hlslpp::max(scale / 2.0f, hlslpp::float2(1.0f, 1.0f));
        }

        scale = RenderTarget::computeFixedResolutionScale(nativeWidth, scale);
        // RasterPS uses resolutionScale.yy for its blocks in both axes.
        *nativePixel = float(scale[1]);
        return float(nativeWidth) * float(scale[0]) * 0.5f;
    }

    // ndcOffset is stereoHudNdcOffset, already rounded to the pixel grid.
    static void applyStereoHudShift(interop::float4x4 &projMatrix, StereoEye eye, float ndcOffset, bool isOrthographic) {
        if (eye == StereoEye::None) {
            return;
        }
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        if (isOrthographic) {
            projMatrix[3][0] -= eyeSign * ndcOffset;
        }
        else {
            projMatrix[2][0] += eyeSign * ndcOffset;
        }
    }

    // Translate the camera laterally along its local right axis (view space +X).
    // Combined with applyStereoOffAxis, this produces depth-dependent parallax:
    // objects at the convergence distance have zero parallax, closer objects pop
    // out, farther objects push back.
    //
    // dynamic3d 1.1: under clip-space separation the eye baseline is DERIVED per
    // frame rather than stored, and it moves with BOTH the FoV and the
    // convergence distance - 2 * separation * tan(half horizontal FoV) * conv
    // for the pair. That is what pins zero parallax at the convergence distance
    // while background disparity stays fixed at `separation`. Anything that
    // wants a physical eye offset has to read it from here rather than assuming
    // the separation slider is one.
    static void applyStereoViewShift(interop::float4x4 &viewMatrix, StereoEye eye, uint32_t separationSlider,
                                     float convergenceWorld, float projectionScale) {
        if (eye == StereoEye::None) {
            return;
        }
        // projectionScale is the live m[0][0] after the widescreen adjust; its
        // reciprocal is tan(half horizontal FoV). Guard against a degenerate
        // projection rather than dividing by ~0.
        if (projectionScale <= 1e-6f) {
            return;
        }
        const float tanHalfHorFov = 1.0f / projectionScale;
        const float halfBaseline = stereoSeparationFraction(separationSlider) * tanHalfHorFov * convergenceWorld;
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        // For a row-vector view matrix, m[3][0] is the X translation in view
        // space. Adding to it shifts world points right in view space, which is
        // equivalent to the camera moving left in world space - what we want for
        // the Left eye. Right eye gets the opposite sign.
        viewMatrix[3][0] += eyeSign * halfBaseline;
    }

    // ProjectionProcessor

    ProjectionProcessor::ProjectionProcessor() { }

    ProjectionProcessor::~ProjectionProcessor() {
        bufferUploader.reset(nullptr);
    }

    void ProjectionProcessor::setup(RenderWorker *worker) {
        bufferUploader = std::make_unique<BufferUploader>(worker->device);
    }

    void ProjectionProcessor::process(const ProcessParams &p) {
        // The frame's main world projection: of the full-width world
        // perspectives, the one with the most draw calls. Goemon also draws
        // other full-width perspectives - 3D HUD items, at much narrower FoVs -
        // and scenes are not processed in draw order, so neither the first nor
        // the last one seen is reliably the world. Taking the first was measured
        // picking a HUD projection nearly every frame, which handed the depth
        // sampler the wrong projection to invert and broke auto-convergence.
        primaryWorld = nullptr;
        uint32_t primaryWorldCalls = 0;
        for (const GameScene &scene : p.curFrame->perspectiveScenes) {
            for (const GameIndices::Projection &sceneProj : scene.projections) {
                const Workload &workload = p.workloadQueue->workloads[sceneProj.workloadIndex];
                const FramebufferPair &fbPair = workload.fbPairs[sceneProj.fbPairIndex];
                const Projection &proj = fbPair.projections[sceneProj.projectionIndex];
                if ((proj.type != Projection::Type::Perspective) || proj.scissorRect.isNull()) {
                    continue;
                }

                const uint32_t groupIndex = workload.drawData.viewProjTransformGroups[proj.transformsIndex];
                const uint32_t matrixId = workload.drawData.transformGroups[groupIndex].matrixId;
                const bool spansFullWidth = (proj.scissorRect.ulx <= fbPair.scissorRect.ulx) &&
                    (proj.scissorRect.lrx >= fbPair.scissorRect.lrx);
                if (spansFullWidth && isStereoViewShiftProjectionId(matrixId) && (proj.gameCallCount > primaryWorldCalls)) {
                    primaryWorld = &proj;
                    primaryWorldCalls = proj.gameCallCount;
                }
            }
        }

        skyInterpolationAngle = 0.0f;
        skyProjScaleX = 0.0f;
        for (uint32_t w : p.curFrame->workloads) {
            Workload &workload = p.workloadQueue->workloads[w];
            DrawData &drawData = workload.drawData;

            // Copy the data.
            drawData.modViewTransforms = drawData.viewTransforms;
            drawData.modProjTransforms = drawData.projTransforms;
            drawData.modViewProjTransforms = drawData.viewProjTransforms;
            drawData.prevViewTransforms = drawData.viewTransforms;
            drawData.prevProjTransforms = drawData.projTransforms;
            drawData.prevViewProjTransforms = drawData.viewProjTransforms;
        }

        debugPerspectiveCount = 0;
        debugViewShiftCount = 0;
        debugOtherIdCount = 0;
        debugLastOtherId = 0;
        debugHudCameraCount = 0;
        debugLastConvergence = 0.0f;

        for (size_t s = 0; s < p.curFrame->perspectiveScenes.size(); s++) {
            processScene(p, p.curFrame->perspectiveScenes[s], s);
        }

        for (size_t s = 0; s < p.curFrame->orthographicScenes.size(); s++) {
            processScene(p, p.curFrame->orthographicScenes[s], s);
        }

        if (p.stereoEye == StereoEye::Left) {
            static uint32_t convergenceLogCounter = 0;
            if ((convergenceLogCounter++ % 30) == 0) {
                fprintf(stdout, "RT64CONV persp=%u viewshift=%u hudcams=%u otherIds=%u lastOtherId=%08X conv=%.2f slider=%u auto=%.2f primaryCalls=%u\n",
                    debugPerspectiveCount, debugViewShiftCount, debugHudCameraCount, debugOtherIdCount, debugLastOtherId, debugLastConvergence,
                    p.stereoConvergence, p.stereoConvergenceAuto,
                    (primaryWorld != nullptr) ? primaryWorld->gameCallCount : 0u);
                fflush(stdout);
            }
        }
    }

    void ProjectionProcessor::processScene(const ProcessParams &p, const GameScene &scene, size_t sceneIndex) {
        for (size_t i = 0; i < scene.projections.size(); i++) {
            const GameIndices::Projection &sceneProj = scene.projections[i];
            Workload &workload = p.workloadQueue->workloads[sceneProj.workloadIndex];
            DrawData &drawData = workload.drawData;
            const FramebufferPair &fbPair = workload.fbPairs[sceneProj.fbPairIndex];
            const Projection &proj = fbPair.projections[sceneProj.projectionIndex];
            const uint16_t viewportOrigin = drawData.viewportOrigins[proj.transformsIndex];
            assert(proj.transformsIndex > 0);

            // Skip projections that didn't actually draw anything.
            if (proj.scissorRect.isNull()) {
                continue;
            }

            // Check the current mapping for the projection.
            const interop::float4x4 *prevProjMatrix = nullptr;
            const interop::float4x4 *prevViewMatrix = nullptr;
            const RigidBody *rigidBody = nullptr;
            const GameFrameMap::WorkloadMap &workloadMap = p.curFrame->frameMap.workloads[sceneProj.workloadIndex];
            if ((p.prevFrame != nullptr) && workloadMap.mapped && !workload.debuggerCamera.enabled) {
                const GameFrameMap::ViewProjectionMap &viewProjMap = workloadMap.viewProjections[proj.transformsIndex];
                if (viewProjMap.mapped) {
                    const Workload &prevWorkload = p.workloadQueue->workloads[workloadMap.prevWorkloadIndex];
                    prevViewMatrix = &prevWorkload.drawData.viewTransforms[viewProjMap.prevTransformIndex];
                    prevProjMatrix = &prevWorkload.drawData.projTransforms[viewProjMap.prevTransformIndex];
                    rigidBody = &viewProjMap.rigidBody;
                }
            }

            const uint32_t curProjGroupIndex = workload.drawData.viewProjTransformGroups[proj.transformsIndex];
            const TransformGroup &curProjGroup = workload.drawData.transformGroups[curProjGroupIndex];
            bool adjustAspectRatio = (curProjGroup.aspectMode == G_EX_ASPECT_ADJUST);
            if (curProjGroup.aspectMode == G_EX_ASPECT_AUTO) {
                FixedRect intersectionRect = proj.scissorRect;
                if (proj.usesViewport()) {
                    const interop::RSPViewport &viewport = drawData.rspViewports[proj.transformsIndex];
                    const int16_t *viewportClipRatios = &drawData.viewportClipRatios[proj.transformsIndex * 4];
                    intersectionRect = intersectionRect.intersection(viewport.rect(viewportClipRatios));
                }

                if (!intersectionRect.isEmpty()) {
                    bool coversWholeWidth = (intersectionRect.ulx <= fbPair.scissorRect.ulx) && (intersectionRect.lrx >= fbPair.scissorRect.lrx);
                    bool horizontalRatio = (intersectionRect.width(true, true) > intersectionRect.height(true, true));
                    adjustAspectRatio = (viewportOrigin == G_EX_ORIGIN_NONE) && coversWholeWidth && horizontalRatio;
                }
            }
 
            float projRatioScale = adjustAspectRatio ? (1.0f / p.aspectRatioScale) : 1.0f;
            interop::float4x4 &viewMatrix = drawData.modViewTransforms[proj.transformsIndex];
            interop::float4x4 &projMatrix = drawData.modProjTransforms[proj.transformsIndex];
            interop::float4x4 &viewProjMatrix = drawData.modViewProjTransforms[proj.transformsIndex];
            viewMatrix = drawData.viewTransforms[proj.transformsIndex];
            projMatrix = drawData.projTransforms[proj.transformsIndex];
            viewProjMatrix = drawData.viewProjTransforms[proj.transformsIndex];

            // Debugger camera.
            if (workload.debuggerCamera.enabled && (proj.type == Projection::Type::Perspective) && (workload.debuggerCamera.sceneIndex == sceneIndex)) {
                viewMatrix = workload.debuggerCamera.viewMatrix;
                projMatrix = workload.debuggerCamera.projMatrix;
            }

            adjustProjectionMatrix(projMatrix, projRatioScale);

            // See process(): picked by draw count, not by processing order.
            const bool isPrimaryWorld = (&proj == primaryWorld);

            // An untagged perspective that is really a HUD item's private camera
            // (see viewMatrixLooksLikeHudCamera) takes the HUD treatment instead
            // of the world's. Judged on the game's own view, before any stereo.
            const bool isHudCamera = (proj.type == Projection::Type::Perspective) && !isPrimaryWorld &&
                isStereoViewShiftProjectionId(curProjGroup.matrixId) &&
                viewMatrixLooksLikeHudCamera(drawData.viewTransforms[proj.transformsIndex]);
            const bool getsWorldShear = (proj.type == Projection::Type::Perspective) &&
                isStereoProjectionId(curProjGroup.matrixId) && !isHudCamera;
            const bool isOrthoProjection = (proj.type == Projection::Type::Orthographic);
            if (isHudCamera) {
                debugHudCameraCount++;
            }
            // Every orthographic projection is 2D content - the logo tiles,
            // menus and HUD panels - whatever its ID. The world-ID test only
            // means something for perspectives: Goemon tags nothing, so the
            // untagged ID has to count as world for its camera to get stereo,
            // and gating orthographic HUD on "not a world ID" left every
            // untagged orthographic draw flat on the screen plane instead of
            // at the HUD depth.
            const bool isHudProjection = isHudCamera || isOrthoProjection ||
                ((proj.type == Projection::Type::Perspective) && isStereoHudProjectionId(curProjGroup.matrixId));
            // On whole native pixels of this pair's viewport, the grid the rect
            // path rounds to as well - see stereoSnapNdcToPixel.
            float hudNativePixel = 0.0f;
            const float hudHalfViewport = isHudProjection ? stereoHalfViewportPixels(workload, fbPair, p.resolutionScale, &hudNativePixel) : 0.0f;
            const float hudNdcOffset = isHudProjection
                ? stereoSnapNdcToPixel(stereoHudNdcOffset(p.stereoHudDepth, p.stereoSeparation), hudHalfViewport, hudNativePixel)
                : 0.0f;
            if (proj.type == Projection::Type::Perspective) {
                debugPerspectiveCount++;
                if (!isStereoViewShiftProjectionId(curProjGroup.matrixId)) {
                    debugOtherIdCount++;
                    debugLastOtherId = curProjGroup.matrixId;
                }
            }

            // Apply stereoscopic off-axis projection offset for world (gameplay)
            // and skybox projections.
            if ((p.stereoMode != UserConfiguration::StereoMode::Off) && getsWorldShear) {
                applyStereoOffAxis(projMatrix, p.stereoEye, p.stereoSeparation);

                // Publish this projection's depth terms for the depth sampler.
                // The world projection is the one the sampled depth buffer was
                // rendered with, so these are the right terms to invert it.
                //
                // Goemon never emits an explicit world tag - everything untagged
                // is treated as gameplay (see isStereoProjectionId) - so the
                // world is identified as the first full-width perspective (see
                // isPrimaryWorld). Requiring full width picks the same pass the
                // depth sampler's own main-pass test picks.
                if (isPrimaryWorld) {
                    // The viewport's depth scale/translate is the second layer
                    // between ndc.z and the stored depth - see
                    // stereoDeviceDepthToViewZ for why assuming 0.5 is wrong.
                    float vpScaleZ = 0.0f;
                    float vpTranslateZ = 0.0f;
                    if (proj.usesViewport()) {
                        const interop::RSPViewport &depthViewport = drawData.rspViewports[proj.transformsIndex];
                        vpScaleZ = depthViewport.scale[2];
                        vpTranslateZ = depthViewport.translate[2];
                    }
                    stereoPublishWorldDepthTerms(projMatrix[2][2], projMatrix[3][2], vpScaleZ, vpTranslateZ);

                    // TEMPORARY sky-scroll investigation: the live horizontal
                    // and vertical scales of the world projection.
                    static uint32_t worldLogCounter = 0;
                    if ((worldLogCounter++ % 97) == 0) {
                        fprintf(stdout, "RT64WORLD m00=%.5f m11=%.5f m20=%.5f projRatioScale=%.4f aspectRatioScale=%.4f eye=%d id=%08X\n",
                            float(projMatrix[0][0]), float(projMatrix[1][1]), float(projMatrix[2][0]), projRatioScale, p.aspectRatioScale,
                            int(p.stereoEye), curProjGroup.matrixId);
                        fflush(stdout);
                    }
                }
            }

            // Apply HUD depth shift to HUD/dialog/cutscene-overlay projections so
            // the user can move them off the screen plane. When the slider is at
            // 50 (neutral) this is a no-op and the HUD stays flat as before.
            // Untagged perspective projections (FMV / cutscene playback) are left
            // alone so they don't flicker as the per-eye shifts go in opposite
            // directions.
            if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                isHudProjection &&
                (p.stereoHudDepth != 50)) {
                applyStereoHudShift(projMatrix, p.stereoEye, hudNdcOffset, isOrthoProjection);
            }

            interop::float4x4 &prevViewTransform = drawData.prevViewTransforms[proj.transformsIndex];
            interop::float4x4 &prevProjTransform = drawData.prevProjTransforms[proj.transformsIndex];
            if ((prevProjMatrix != nullptr) && (prevViewMatrix != nullptr) && (rigidBody != nullptr)) {
                const interop::float4x4 curViewTransform = viewMatrix;
                const interop::float4x4 curProjTransform = projMatrix;
                interop::float4x4 adjustedPrevProj = *prevProjMatrix;
                adjustProjectionMatrix(adjustedPrevProj, projRatioScale);
                // The previous-frame projection must receive the same stereo
                // off-axis shift as the current one. Otherwise the lerp below
                // produces a partially-shifted matrix that jitters every frame
                // because interpolation weights vary across the interpolated
                // frames the workload emits.
                if ((p.stereoMode != UserConfiguration::StereoMode::Off) && getsWorldShear) {
                    applyStereoOffAxis(adjustedPrevProj, p.stereoEye, p.stereoSeparation);
                }
                if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                    isHudProjection &&
                    (p.stereoHudDepth != 50)) {
                    applyStereoHudShift(adjustedPrevProj, p.stereoEye, hudNdcOffset, isOrthoProjection);
                }
                viewMatrix = rigidBody->lerp(p.curFrameWeight, *prevViewMatrix, curViewTransform, true);
                prevViewTransform = rigidBody->lerp(p.prevFrameWeight, *prevViewMatrix, curViewTransform, true);

                // We only interpolate the projection if the view matrix has been interpolated.
                const bool interpolateProjection = rigidBody->lerpTranslation || rigidBody->lerpRotation;
                if (interpolateProjection) {
                    projMatrix = lerpMatrix(adjustedPrevProj, curProjTransform, p.curFrameWeight);
                    prevProjTransform = lerpMatrix(adjustedPrevProj, curProjTransform, p.prevFrameWeight);
                }
                else {
                    projMatrix = curProjTransform;
                    prevProjTransform = curProjTransform;
                }
            }
            else {
                prevViewTransform = viewMatrix;
                prevProjTransform = projMatrix;
            }

            // Apply the matching lateral view-space shift for stereo. Done after
            // the lerp so both interpolation endpoints carry the same shift. Both
            // the current and previous view matrices must shift the same way so
            // motion vectors / prev-viewProj stay correct. Goemon's yaw-following
            // sky is its own skybox-tagged projection (patches/background.c), so
            // it gets the shear alone and sits at infinity.
            if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                (proj.type == Projection::Type::Perspective) &&
                isStereoViewShiftProjectionId(curProjGroup.matrixId) && !isHudCamera) {
                // m[0][0] is the post-widescreen-adjust horizontal projection
                // scale for this same transform index, and the shear above left
                // it untouched, so it is still the live tan(half FoV) reciprocal
                // the derived eye baseline needs.
                const float projectionScale = projMatrix[0][0];
                const float convergenceWorld = stereoEffectiveConvergence(p.stereoConvergence, p.stereoConvergenceAuto);
                debugViewShiftCount++;
                debugLastConvergence = convergenceWorld;
                applyStereoViewShift(viewMatrix, p.stereoEye, p.stereoSeparation, convergenceWorld, projectionScale);
                applyStereoViewShift(prevViewTransform, p.stereoEye, p.stereoSeparation, convergenceWorld, projectionScale);
            }

            // The bearing, in the INTERPOLATED view, of the direction straight
            // ahead of the CURRENT camera: how far a camera-tracking sky has to
            // be rotated on this frame, since the game positioned it for the
            // current heading. Rotation only - the stereo view shift is a
            // translation and has no effect at infinity - and the projection
            // shear is the same offset at infinity either way.
            //
            // Row vectors: v_view = v_world * V. The view's rotation is
            // orthonormal up to a uniform scale, so its transpose inverts it up
            // to a scale that cancels in the ratio below: the current camera's
            // forward (view -Z) in world space is -column 2 of the current V.
            if (isPrimaryWorld) {
                const interop::float4x4 &curView = drawData.viewTransforms[proj.transformsIndex];
                float vx = 0.0f;
                float vz = 0.0f;
                for (int i = 0; i < 3; i++) {
                    const float forward = -float(curView[i][2]);
                    vx += forward * float(viewMatrix[i][0]);
                    vz += forward * float(viewMatrix[i][2]);
                }
                // vz < 0 is in front; a point behind the interpolated camera
                // means a degenerate cut, not a turn.
                if (vz < -1e-6f) {
                    skyInterpolationAngle = std::atan2(vx, -vz);
                }
                skyProjScaleX = float(projMatrix[0][0]);
            }

            viewProjMatrix = hlslpp::mul(viewMatrix, projMatrix);

            interop::float4x4 &prevViewProjTransform = drawData.prevViewProjTransforms[proj.transformsIndex];
            prevViewProjTransform = hlslpp::mul(prevViewTransform, prevProjTransform);
        }
    }

    void ProjectionProcessor::upload(const ProcessParams &p) {
        uploads.clear();

        for (uint32_t w : p.curFrame->workloads) {
            Workload &workload = p.workloadQueue->workloads[w];
            const DrawData &drawData = workload.drawData;
            DrawBuffers &drawBuffers = workload.drawBuffers;
            std::pair<size_t, size_t> uploadRange = { 0, drawData.viewProjTransforms.size() };
            uploads.emplace_back(BufferUploader::Upload{ drawData.modViewProjTransforms.data(), uploadRange, sizeof(interop::float4x4), RenderBufferFlag::STORAGE, { }, &drawBuffers.viewProjTransformsBuffer });
        }

        bufferUploader->submit(p.worker, uploads);
    }
};