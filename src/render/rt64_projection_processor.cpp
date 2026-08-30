//
// RT64
//

#include "rt64_projection_processor.h"

#include <algorithm>

#include "common/rt64_math.h"
#include "hle/rt64_workload_queue.h"

namespace RT64 {
    inline void adjustProjectionMatrix(interop::float4x4 &matrix, const float aspectRatioScale) {
        matrix[0][0] *= aspectRatioScale;
        matrix[1][0] *= aspectRatioScale;
        matrix[2][0] *= aspectRatioScale;
        matrix[3][0] *= aspectRatioScale;
    }

    // BanjoRecomp3D-specific transform IDs: see patches/transform_ids.h.
    // The world (gameplay) projection receives full per-eye stereo: projection
    // off-axis (inter-eye disparity) + view translation (depth-dependent parallax).
    // The skybox receives only the projection off-axis with no view shift, which
    // gives it the maximum positive parallax of an object at infinity — exactly
    // what we want so it sits well behind the world geometry.
    // HUD/menu/cutscene-overlay projections (HUD, PRESS_START, BK_LOGO, COPYRIGHT,
    // GAME_OVER, THE_END, TRANSITION, PILLARBOX, PORTRAITs) deliberately stay flat
    // at the screen plane to remain readable.
    static constexpr uint32_t BANJO_PROJECTION_GAMEPLAY_TRANSFORM_ID     = 0x00001000;
    static constexpr uint32_t BANJO_PROJECTION_SKYBOX_TRANSFORM_ID       = 0x00001001;
    static constexpr uint32_t BANJO_PROJECTION_TRANSITION_TRANSFORM_ID   = 0x00001002;
    static constexpr uint32_t BANJO_PROJECTION_PRESS_START_TRANSFORM_ID  = 0x00001003;
    static constexpr uint32_t BANJO_PROJECTION_HUD_TRANSFORM_ID          = 0x00001004;
    static constexpr uint32_t BANJO_PROJECTION_BK_LOGO_TRANSFORM_ID      = 0x00001005;
    static constexpr uint32_t BANJO_PROJECTION_COPYRIGHT_TRANSFORM_ID    = 0x00001006;
    static constexpr uint32_t BANJO_PROJECTION_GAME_OVER_TRANSFORM_ID    = 0x00001007;
    static constexpr uint32_t BANJO_PROJECTION_THE_END_TRANSFORM_ID      = 0x00001008;
    static constexpr uint32_t BANJO_PROJECTION_PILLARBOX_TRANSFORM_ID    = 0x00001009;
    static constexpr uint32_t BANJO_PROJECTION_DIALOG_BUBBLE_TRANSFORM_ID = 0x0000100A;
    static constexpr uint32_t BANJO_PROJECTION_PORTRAIT_TRANSFORM_ID_START = 0x00001100;
    static constexpr uint32_t BANJO_PROJECTION_PORTRAIT_TRANSFORM_ID_END   = 0x000011FF;

    static bool isStereoProjectionId(uint32_t matrixId) {
        return (matrixId == BANJO_PROJECTION_GAMEPLAY_TRANSFORM_ID) ||
               (matrixId == BANJO_PROJECTION_SKYBOX_TRANSFORM_ID);
    }

    static bool isStereoViewShiftProjectionId(uint32_t matrixId) {
        return (matrixId == BANJO_PROJECTION_GAMEPLAY_TRANSFORM_ID);
    }

    // HUD / dialog / text / cutscene-overlay projections that should receive
    // the user's configured constant stereo depth (no view shift). The pillarbox
    // is intentionally excluded because it's a screen-edge mask and looks weird
    // if it moves out of plane. The transition fade is also excluded for the
    // same reason.
    static bool isStereoHudProjectionId(uint32_t matrixId) {
        if (matrixId == BANJO_PROJECTION_HUD_TRANSFORM_ID) return true;
        if (matrixId == BANJO_PROJECTION_PRESS_START_TRANSFORM_ID) return true;
        if (matrixId == BANJO_PROJECTION_BK_LOGO_TRANSFORM_ID) return true;
        if (matrixId == BANJO_PROJECTION_COPYRIGHT_TRANSFORM_ID) return true;
        if (matrixId == BANJO_PROJECTION_GAME_OVER_TRANSFORM_ID) return true;
        if (matrixId == BANJO_PROJECTION_THE_END_TRANSFORM_ID) return true;
        if (matrixId == BANJO_PROJECTION_DIALOG_BUBBLE_TRANSFORM_ID) return true;
        if ((matrixId >= BANJO_PROJECTION_PORTRAIT_TRANSFORM_ID_START) &&
            (matrixId <= BANJO_PROJECTION_PORTRAIT_TRANSFORM_ID_END)) return true;
        return false;
    }

    // The dialog bubble specifically needs a stronger HUD shift than the
    // default perspective-HUD path provides, so it lands at the same depth as
    // the text rectangles drawn alongside it. Other perspective-HUD elements
    // (press-start logo, BK title logo, etc.) keep the original P[0][0]-based
    // shift so we don't alter behavior the user already signed off on.
    static bool isStereoBubbleProjectionId(uint32_t matrixId) {
        return (matrixId == BANJO_PROJECTION_DIALOG_BUBBLE_TRANSFORM_ID);
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
    // The 0..50 slider maps linearly onto 0..0.10 of screen width. The default
    // (50 -> 0.100) lands within 4% of what the previous
    // (sep / 2 / conv) * m[0][0] form produced at the default sliders on a 16:9
    // window (0.0966), so the shipped look carries over - and it is now the top
    // of the range rather than its midpoint, since everything above it was past
    // what a typical screen can be fused at.
    //
    // Calibration reference: the divergence ceiling - where background
    // disparity reaches an adult IPD, the eyes are forced outward, and no
    // amount of practice can fuse it - is IPD / screen width. That is about
    // 0.105 on a 27-inch 16:9 monitor and proportionally lower on anything
    // bigger (~0.05 on a 55-inch TV). Values past it stay reachable because the
    // physical screen size is not visible from here.
    static constexpr float SeparationPerSlider = 0.10f / 50.0f;

    // The separation the HUD constants further down were tuned against, which
    // is the top of the slider rather than its default - those constants were
    // matched by eye back when the old form produced 0.0966 here. This is a
    // calibration anchor, NOT the shipped default (that is 10, a fifth of
    // this), so do not "fix" it to track the default: doing so would multiply
    // every HUD offset by five and throw away the tuning.
    static constexpr float HudReferenceSeparation = 50.0f * SeparationPerSlider;

    static float stereoSeparation(uint32_t separationSlider) {
        return static_cast<float>(separationSlider) * SeparationPerSlider;
    }

    // Convergence stays what it always was: the distance at which geometry sits
    // exactly on the screen plane. It arrives in tenths of a slider unit
    // (1..500 = 0.1..50), so the game-unit conversion is 2 per tenth: the
    // 0.1..50 slider spans 2..1000 game units.
    static float stereoConvergenceWorld(uint32_t convergenceTenths) {
        return static_cast<float>(convergenceTenths) * 2.0f;
    }

    // BK's horizontal projection scale at the aspect ratio the HUD constants
    // below were tuned at. The game builds every perspective projection from
    // one guPerspective call at a fixed 40-degree vertical FoV over the VI's
    // 4:3 source, and RT64 then narrows m[0][0] by 1/aspectRatioScale for
    // widescreen, so at 16:9 this is 1 / ((16/9) * tan(20 deg)). Used in place
    // of the live m[0][0] so HUD depth stops tracking the output aspect ratio.
    static constexpr float ReferenceProjectionScale = 1.5455f;

    // dynamic3d 1.1 - the shear is the knob.
    //
    // This used to be (sep / 2 / conv) * m[0][0], which rode on the live
    // projection scale: RT64 narrows m[0][0] by 1/aspectRatioScale for
    // widescreen, so the depth effect quietly scaled with the output aspect
    // ratio (about a quarter of it lost going from 16:9 to 21:9, and a third
    // gained going to 4:3). Clip space is invariant to that, and to any in-game
    // FoV change, by construction.
    //
    // The old cap on the shear term is gone with it. It existed because
    // sep / (2 * conv) genuinely grows without bound as convergence approaches
    // zero; the shear is now just `separation`, which the slider range already
    // bounds, so there is nothing left for the cap to protect against
    // (dynamic3d 2.4).
    static void applyStereoOffAxis(interop::float4x4 &projMatrix, StereoEye eye, uint32_t separationSlider) {
        if (eye == StereoEye::None) {
            return;
        }
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        // Adds an asymmetric horizontal shift to the projection's principal
        // point. Equivalent to rebuilding the frustum with
        // [L, R] = [-horFov + offset, horFov + offset].
        projMatrix[2][0] += eyeSign * stereoSeparation(separationSlider);
    }

    // Apply a constant per-eye horizontal shift to a HUD/UI projection matrix so
    // it sits at a user-selected stereo depth instead of flat on the screen.
    // hudDepthSlider 0..100: 50 = screen plane (no shift), below = push behind
    // the screen (positive parallax), above = pop out (negative parallax).
    //
    // dynamic3d 5.2: a layer parked at a fixed multiple of the convergence
    // distance has a shift of separation * (1/factor - 1) - proportional to
    // separation, and independent of convergence. Scaling by separation is what
    // makes HUD depth track the depth knob, and what makes the HUD go properly
    // flat when separation is 0 (the old fixed offset split the HUD even with
    // the 3D effect dialled all the way down). The per-projection-type
    // constants below are that (1/factor - 1) mapping; they keep their
    // empirically tuned values, so the relative depths of text, icons and the
    // dialog bubble are unchanged at the default separation.
    static void applyStereoHudShift(interop::float4x4 &projMatrix, StereoEye eye, uint32_t hudDepthSlider, uint32_t separationSlider, bool isOrthographic, bool matchOrthoScale = false) {
        if (eye == StereoEye::None) {
            return;
        }
        const float centered = (static_cast<float>(hudDepthSlider) - 50.0f) / 50.0f; // -1..+1
        constexpr float maxHudOffset = 0.04f;
        // Negate so slider > 50 produces pop-out (negative parallax) and
        // slider < 50 produces push-back (positive parallax).
        const float separationScale = stereoSeparation(separationSlider) / HudReferenceSeparation;
        const float hudOffset = -centered * maxHudOffset * separationScale;
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;

        // Map onto an NDC x offset, per projection type:
        //   Perspective: m[2][0] += K becomes a constant NDC shift of K after
        //     the perspective divide. Scaled by the reference projection term
        //     rather than the live one so the depth holds across aspect ratios.
        //   Orthographic: no perspective divide, so m[3][0] += K shifts NDC by
        //     +K directly, and the sign is flipped below to keep the slider
        //     pushing both projection types the same way. The 2.75 was matched
        //     by eye against the perspective path so dialog text and icons sit
        //     at the same depth at the same slider value.
        //   Dialog bubble: its transform chain runs through gameplay's view
        //     matrix, which amplifies the projection-side shift, so it needs
        //     its own empirical constant rather than the perspective one.
        float ndcOffset = hudOffset;
        if (isOrthographic) {
            ndcOffset *= 2.75f;
        }
        else if (matchOrthoScale) {
            ndcOffset *= 1.3f;
        }
        else {
            ndcOffset *= ReferenceProjectionScale;
        }

        // dynamic3d 6.1 - pop-out is not divergence, so the two directions do
        // not get the same limit. Behind the screen plane the constraint is
        // physical: uncrossed disparity past an IPD forces the eyes outward and
        // cannot be fused. In front of it the eyes converge inward and there is
        // nothing to protect against, so reusing the behind-limit would only
        // clip valid pop-out. Branch on hudOffset, which is eye-independent -
        // the sign of the applied shift encodes which eye, not which side of
        // the screen plane the element is on.
        constexpr float BehindNdcLimit = 0.10f;   // ~5% of eye width
        constexpr float PopOutNdcLimit = 0.30f;   // ~15% of eye width
        const float ndcLimit = (hudOffset > 0.0f) ? BehindNdcLimit : PopOutNdcLimit;
        ndcOffset = std::max(-ndcLimit, std::min(ndcLimit, ndcOffset));

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
    // out, farther objects push back. Without this shift every object would get
    // the same constant disparity and the whole image would just slide sideways
    // - the symptom the user reported as "can't get pop-out".
    //
    // dynamic3d 1.1: under clip-space separation the eye baseline is derived
    // per frame rather than stored, and it moves with BOTH the FoV and the
    // convergence distance - 2 * separation * tan(half horizontal FoV) * conv
    // for the pair. That is what pins zero parallax at the convergence distance
    // while background disparity stays fixed at `separation`. Anything that
    // wants a physical eye offset has to read it from here rather than assuming
    // the separation slider is one.
    static void applyStereoViewShift(interop::float4x4 &viewMatrix, StereoEye eye, uint32_t separationSlider, uint32_t convergenceTenths, float projectionScale) {
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
        const float halfBaseline = stereoSeparation(separationSlider) * tanHalfHorFov * stereoConvergenceWorld(convergenceTenths);
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

        for (size_t s = 0; s < p.curFrame->perspectiveScenes.size(); s++) {
            processScene(p, p.curFrame->perspectiveScenes[s], s);
        }

        for (size_t s = 0; s < p.curFrame->orthographicScenes.size(); s++) {
            processScene(p, p.curFrame->orthographicScenes[s], s);
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

            // Apply stereoscopic off-axis projection offset for world (gameplay)
            // and skybox projections.
            if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                (proj.type == Projection::Type::Perspective) &&
                isStereoProjectionId(curProjGroup.matrixId)) {
                applyStereoOffAxis(projMatrix, p.stereoEye, p.stereoSeparation);
            }

            // Apply HUD depth shift to HUD/dialog/cutscene-overlay projections so
            // the user can move them off the screen plane. When the slider is at
            // 50 (neutral) this is a no-op and the HUD stays flat as before.
            // Any orthographic projection counts as HUD in BK (the world is the
            // only thing the game renders in perspective), plus the explicit HUD
            // perspective IDs (pause menu, GAME OVER, dialog overlay, etc.).
            // Gameplay and skybox perspective projections are explicitly skipped
            // so the world stereo isn't disturbed.
            {
                // Apply HUD depth to orthographic projections (BK uses ortho only
                // for UI/2D content) and to perspective projections explicitly
                // tagged with HUD-like IDs. Untagged perspective projections —
                // such as FMV/cutscene playback — are left alone so they don't
                // flicker as the per-eye shifts go in opposite directions.
                const bool isOrtho = (proj.type == Projection::Type::Orthographic);
                const bool isHudProjection = (!isStereoProjectionId(curProjGroup.matrixId)) &&
                    (isOrtho || isStereoHudProjectionId(curProjGroup.matrixId));
                const bool matchOrthoScale = isStereoBubbleProjectionId(curProjGroup.matrixId);
                if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                    isHudProjection &&
                    (p.stereoHudDepth != 50)) {
                    applyStereoHudShift(projMatrix, p.stereoEye, p.stereoHudDepth, p.stereoSeparation, isOrtho, matchOrthoScale);
                }
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
                if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                    (proj.type == Projection::Type::Perspective) &&
                    isStereoProjectionId(curProjGroup.matrixId)) {
                    applyStereoOffAxis(adjustedPrevProj, p.stereoEye, p.stereoSeparation);
                }
                {
                    const bool isOrtho = (proj.type == Projection::Type::Orthographic);
                    const bool isHudProjection = (!isStereoProjectionId(curProjGroup.matrixId)) &&
                        (isOrtho || isStereoHudProjectionId(curProjGroup.matrixId));
                    const bool matchOrthoScale = isStereoBubbleProjectionId(curProjGroup.matrixId);
                    if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                        isHudProjection &&
                        (p.stereoHudDepth != 50)) {
                        applyStereoHudShift(adjustedPrevProj, p.stereoEye, p.stereoHudDepth, p.stereoSeparation, isOrtho, matchOrthoScale);
                    }
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
            // motion vectors / prev-viewProj stay correct. Note: the skybox does
            // NOT get the view shift — combined with the projection off-axis, that
            // produces the maximum positive parallax (i.e., infinity).
            if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                (proj.type == Projection::Type::Perspective) &&
                isStereoViewShiftProjectionId(curProjGroup.matrixId)) {
                // m[0][0] is the post-widescreen-adjust horizontal projection
                // scale for this same transform index, and the shear above left
                // it untouched, so it is still the live tan(half FoV) reciprocal
                // the derived eye baseline needs.
                const float projectionScale = projMatrix[0][0];
                applyStereoViewShift(viewMatrix, p.stereoEye, p.stereoSeparation, p.stereoConvergence, projectionScale);
                applyStereoViewShift(prevViewTransform, p.stereoEye, p.stereoSeparation, p.stereoConvergence, projectionScale);
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