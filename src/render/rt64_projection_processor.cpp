//
// RT64
//

#include "rt64_projection_processor.h"

#include <cmath>

#include "../include/rt64_extended_gbi.h"
#include "common/rt64_math.h"
#include "hle/rt64_workload_queue.h"

namespace RT64 {
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

    // Heuristic skybox detector for untagged (G_EX_ID_AUTO) perspective
    // projections. A skybox is the only common N64 perspective draw whose
    // view matrix has near-zero translation — the camera sits at the origin
    // so the sky cube/sphere appears infinitely far regardless of where the
    // gameplay camera is. Gameplay projections always have a meaningful
    // world-space camera position baked into the view's translation column.
    //
    // Threshold chosen to be well below any sensible N64 world-space camera
    // position (game units, typically hundreds to thousands) but above
    // floating-point noise from interpolation. If a skybox ever drifts a
    // tiny bit off origin (e.g. parented to the player's XZ), bump this.
    //
    // The check pairs with the existing isStereoProjectionId AUTO fallback:
    // an untagged perspective gets the off-axis shift unconditionally (so it
    // still renders in stereo), but the view shift is gated on this
    // heuristic so skybox-like draws land at infinity instead of getting
    // pulled to the convergence plane.
    static bool viewMatrixLooksLikeSkybox(const interop::float4x4 &viewMatrix) {
        const float tx = static_cast<float>(viewMatrix[3][0]);
        const float ty = static_cast<float>(viewMatrix[3][1]);
        const float tz = static_cast<float>(viewMatrix[3][2]);
        constexpr float kSkyboxTranslationThreshold = 1.0f;   // game units
        return (std::abs(tx) < kSkyboxTranslationThreshold)
            && (std::abs(ty) < kSkyboxTranslationThreshold)
            && (std::abs(tz) < kSkyboxTranslationThreshold);
    }

    // HUD projections that should receive the user's configured constant
    // stereo depth (no view shift). Currently just the main HUD; pause menu /
    // dialog / etc. are not tagged yet and intentionally fall through to mono.
    static bool isStereoHudProjectionId(uint32_t matrixId) {
        return (matrixId == GOEMON_PROJECTION_HUD_TRANSFORM_ID);
    }

    // Convert the user-facing 0..100 sliders to world-space stereo parameters.
    // Centralised here so applyStereoOffAxis and applyStereoViewShift always agree.
    //
    // Goemon uses smaller world-space distances than other N64 ports (cameras
    // typically sit much closer to the action), so the same slider value
    // produces ~50x more apparent disparity than it would in a game like
    // Banjo. The 0.02 multiplier on separation compresses the slider so
    // default (50) produces the comfortable eye-offset (~0.025) that the
    // playtester landed on after manually setting slider=1; the full 1..100
    // range then maps to "subtle .. clearly visible" rather than
    // "barely-anything .. eye-strain".
    static void stereoWorldUnits(uint32_t separationSlider, uint32_t convergenceSlider,
                                 float &separationWorld, float &convergenceWorld) {
        separationWorld = static_cast<float>(separationSlider) * 0.02f;             // 0..2 game units
        convergenceWorld = static_cast<float>(convergenceSlider) * 20.0f;           // 20..2000 game units (slider min 1)
    }

    static void applyStereoOffAxis(interop::float4x4 &projMatrix, StereoEye eye, uint32_t separationSlider, uint32_t convergenceSlider) {
        if (eye == StereoEye::None) {
            return;
        }
        float separationWorld;
        float convergenceWorld;
        stereoWorldUnits(separationSlider, convergenceSlider, separationWorld, convergenceWorld);
        float eyeOffset = 0.5f * separationWorld / convergenceWorld;
        constexpr float maxEyeOffset = 0.08f;
        if (eyeOffset > maxEyeOffset) eyeOffset = maxEyeOffset;
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        // Adds an asymmetric horizontal shift to the projection's principal point.
        // Equivalent to rebuilding the frustum with [L,R] = [-horFov+offset, horFov+offset].
        projMatrix[2][0] += eyeSign * eyeOffset * projMatrix[0][0];
    }

    // Apply a constant per-eye horizontal shift to a HUD/UI projection matrix so
    // it sits at a user-selected stereo depth instead of flat on the screen.
    // hudDepthSlider 0..100: 50 = screen plane (no shift), below = push behind
    // the screen (positive parallax), above = pop out (negative parallax).
    //
    // Perspective and orthographic projections need different elements:
    //   - Perspective: m[2][0] gets the principal-point offset, scaled by m[0][0].
    //     After the perspective divide this becomes a constant NDC shift.
    //   - Orthographic: there is no perspective divide, so we add a direct NDC
    //     shift via m[3][0].
    static void applyStereoHudShift(interop::float4x4 &projMatrix, StereoEye eye, uint32_t hudDepthSlider, bool isOrthographic) {
        if (eye == StereoEye::None) {
            return;
        }
        const float centered = (static_cast<float>(hudDepthSlider) - 50.0f) / 50.0f; // -1..+1
        constexpr float maxHudOffset = 0.04f;
        // Negate so slider > 50 produces pop-out (negative parallax) and
        // slider < 50 produces push-back (positive parallax).
        const float hudOffset = -centered * maxHudOffset;
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        if (isOrthographic) {
            // Match the visible magnitude of the perspective HUD shift so an
            // ortho HUD and perspective HUD at the same slider value land at
            // approximately the same depth.
            constexpr float perspectiveToOrthoScale = 2.75f;
            projMatrix[3][0] -= eyeSign * hudOffset * perspectiveToOrthoScale;
        }
        else {
            projMatrix[2][0] += eyeSign * hudOffset * projMatrix[0][0];
        }
    }

    // Translate the camera laterally along its local right axis (view space +X)
    // by ±separationWorld/2. Combined with applyStereoOffAxis, this produces
    // depth-dependent parallax: objects at convergenceWorld have zero parallax,
    // closer objects pop out, farther objects push back.
    static void applyStereoViewShift(interop::float4x4 &viewMatrix, StereoEye eye, uint32_t separationSlider, uint32_t convergenceSlider) {
        if (eye == StereoEye::None) {
            return;
        }
        float separationWorld;
        float convergenceWorld;
        stereoWorldUnits(separationSlider, convergenceSlider, separationWorld, convergenceWorld);
        // Match the clamp applied to the projection shift so the view and
        // projection stay geometrically consistent at extreme slider settings.
        const float separationCap = 0.16f * convergenceWorld;
        const float effectiveSeparation = (separationWorld > separationCap) ? separationCap : separationWorld;
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        // For a row-vector view matrix, m[3][0] is the X translation in view
        // space. Adding to it shifts world points right in view space, which is
        // equivalent to the camera moving left in world space — what we want for
        // the Left eye. Right eye gets the opposite sign.
        viewMatrix[3][0] += eyeSign * (effectiveSeparation * 0.5f);
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
                applyStereoOffAxis(projMatrix, p.stereoEye, p.stereoSeparation, p.stereoConvergence);
            }

            // Apply HUD depth shift to HUD/dialog/cutscene-overlay projections so
            // the user can move them off the screen plane. When the slider is at
            // 50 (neutral) this is a no-op and the HUD stays flat as before.
            // Untagged perspective projections (FMV / cutscene playback) are left
            // alone so they don't flicker as the per-eye shifts go in opposite
            // directions.
            {
                const bool isOrtho = (proj.type == Projection::Type::Orthographic);
                const bool isHudProjection = (!isStereoProjectionId(curProjGroup.matrixId)) &&
                    (isOrtho || isStereoHudProjectionId(curProjGroup.matrixId));
                if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                    isHudProjection &&
                    (p.stereoHudDepth != 50)) {
                    applyStereoHudShift(projMatrix, p.stereoEye, p.stereoHudDepth, isOrtho);
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
                    applyStereoOffAxis(adjustedPrevProj, p.stereoEye, p.stereoSeparation, p.stereoConvergence);
                }
                {
                    const bool isOrtho = (proj.type == Projection::Type::Orthographic);
                    const bool isHudProjection = (!isStereoProjectionId(curProjGroup.matrixId)) &&
                        (isOrtho || isStereoHudProjectionId(curProjGroup.matrixId));
                    if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                        isHudProjection &&
                        (p.stereoHudDepth != 50)) {
                        applyStereoHudShift(adjustedPrevProj, p.stereoEye, p.stereoHudDepth, isOrtho);
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
            // motion vectors / prev-viewProj stay correct.
            //
            // NOTE on the skybox: in a typical N64 game the skybox renders with
            // the camera at the origin and gets only the projection off-axis,
            // which puts it at infinity. Goemon's skybox doesn't seem to follow
            // that pattern — the viewMatrixLooksLikeSkybox heuristic doesn't
            // fire — so leaving the view shift on here for now. Revisit once
            // we know whether Goemon's sky is its own perspective scene or
            // distant geometry inside the gameplay perspective.
            if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                (proj.type == Projection::Type::Perspective) &&
                isStereoViewShiftProjectionId(curProjGroup.matrixId)) {
                applyStereoViewShift(viewMatrix, p.stereoEye, p.stereoSeparation, p.stereoConvergence);
                applyStereoViewShift(prevViewTransform, p.stereoEye, p.stereoSeparation, p.stereoConvergence);
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