//
// RT64
//

#include "rt64_projection_processor.h"

#include "common/rt64_math.h"
#include "hle/rt64_workload_queue.h"

namespace RT64 {
    inline void adjustProjectionMatrix(interop::float4x4 &matrix, const float aspectRatioScale) {
        matrix[0][0] *= aspectRatioScale;
        matrix[1][0] *= aspectRatioScale;
        matrix[2][0] *= aspectRatioScale;
        matrix[3][0] *= aspectRatioScale;
    }

    // Dinosaur Planet transform IDs: see patches/include/matrix_groups.h.
    //
    // The world (gameplay) projection receives full per-eye stereo: projection
    // off-axis (inter-eye disparity) + view translation (depth-dependent parallax).
    // The skybox receives only the projection off-axis with no view shift, which
    // gives it the maximum positive parallax of an object at infinity — exactly
    // what we want so it sits well behind the world geometry.
    // HUD/menu/overlay projections deliberately stay flat at the user's chosen
    // constant depth so they remain readable.
    //
    // Unlike the Banjo/Goemon ports, Dino's world projection was ALREADY tagged
    // before stereo existed: patches/src/core/camera.c's camSetupRSPMatrices
    // emits CAMERA_MTX_GROUP_ID_START + gCameraSelector (0x10..0x1F, one per
    // camera selector) for RT64's frame interpolation. Stereo reuses that tag
    // rather than introducing a parallel one, so the gameplay path needs no new
    // patch at all. Only the skybox and HUD need their own IDs.
    static constexpr uint32_t DINO_PROJECTION_CAMERA_TRANSFORM_ID_START = 0x00000010;
    static constexpr uint32_t DINO_PROJECTION_CAMERA_TRANSFORM_ID_END   = 0x0000001F;
    static constexpr uint32_t DINO_PROJECTION_SKYBOX_TRANSFORM_ID       = 0x00001001;
    static constexpr uint32_t DINO_PROJECTION_HUD_TRANSFORM_ID          = 0x00001004;

    static bool isStereoCameraProjectionId(uint32_t matrixId) {
        return (matrixId >= DINO_PROJECTION_CAMERA_TRANSFORM_ID_START) &&
               (matrixId <= DINO_PROJECTION_CAMERA_TRANSFORM_ID_END);
    }

    // Projections that get the off-axis shear: world geometry and the skybox.
    static bool isStereoProjectionId(uint32_t matrixId) {
        return isStereoCameraProjectionId(matrixId) ||
               (matrixId == DINO_PROJECTION_SKYBOX_TRANSFORM_ID);
    }

    // Projections that additionally get the lateral view shift. The skybox is
    // excluded on purpose — that omission is what places it at infinity.
    static bool isStereoViewShiftProjectionId(uint32_t matrixId) {
        return isStereoCameraProjectionId(matrixId);
    }

    // HUD / menu / text projections that should receive the user's configured
    // constant stereo depth (no view shift).
    static bool isStereoHudProjectionId(uint32_t matrixId) {
        return (matrixId == DINO_PROJECTION_HUD_TRANSFORM_ID);
    }

    // Convert the user-facing 0..100 sliders to world-space stereo parameters.
    // Centralised here so applyStereoOffAxis and applyStereoViewShift always agree.
    //
    // Dino's world scale: near plane 4.0, far plane 10000.0 (camera.c), and the
    // camera DLLs work at 20..250 units from the subject (84_camnormal, 91_camlockon,
    // 92_camshipbattle, 86_cam1stperson). Convergence therefore wants to land near
    // ~200 units at the default slider of 20, which is why the multiplier is 10 and
    // not Banjo's 20 (its cameras sit roughly twice as far out).
    //
    // Separation is then chosen so the default (sep 50, conv 20) produces an
    // eyeOffset of 0.5 * 7.5 / 200 = 0.019 — comfortably below the 0.08 clamp, so
    // the slider has usable range in both directions instead of saturating.
    //
    // RETUNE THESE TWO FIRST if the 3D reads too strong/weak in game; everything
    // else in this file is geometry and should not need touching.
    static constexpr float kSeparationWorldScale  = 0.15f;   // slider 50  -> 7.5 units
    static constexpr float kConvergenceWorldScale = 10.0f;   // slider 20  -> 200 units

    static void stereoWorldUnits(uint32_t separationSlider, uint32_t convergenceSlider,
                                 float &separationWorld, float &convergenceWorld) {
        separationWorld = static_cast<float>(separationSlider) * kSeparationWorldScale;
        convergenceWorld = static_cast<float>(convergenceSlider) * kConvergenceWorldScale;
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
    // Slider 0..100 -> signed NDC-space HUD offset, before the per-projection-type
    // scaling below. Shared by the projection path and the texture-rectangle path.
    static constexpr float kMaxHudOffset = 0.04f;

    // For perspective, m[2][0] += K produces an NDC.x shift of -K (after the
    // right-handed perspective divide). For ortho, m[3][0] += K produces an NDC.x
    // shift of +K directly. To make the slider push HUD the same direction across
    // both projection types we apply the opposite sign to ortho. The scale factor
    // is 1/tan(FOV/2), which matches the visible magnitude so text and icons sit at
    // the same depth at the same slider value.
    //
    // Dino's default vertical FOV is 60 degrees (camInit / camResetProjection;
    // camSetFOV clamps to 40..90), so this is 1/tan(30 deg) = 1.732 — NOT Banjo's
    // 2.75, which was derived from its 40 degree FOV.
    static constexpr float kPerspectiveToOrthoScale = 1.7320508f;

    static float stereoHudNdcOffset(StereoEye eye, uint32_t hudDepthSlider) {
        if ((eye == StereoEye::None) || (hudDepthSlider == 50)) {
            return 0.0f;
        }
        const float centered = (static_cast<float>(hudDepthSlider) - 50.0f) / 50.0f; // -1..+1
        // Negate so slider > 50 produces pop-out (negative parallax) and
        // slider < 50 produces push-back (positive parallax).
        const float hudOffset = -centered * kMaxHudOffset;
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        return eyeSign * hudOffset;
    }

    float stereoHudRectOffsetX(StereoEye eye, uint32_t hudDepthSlider) {
        // Negation mirrors the orthographic branch below so rectangles shift in
        // the same direction as everything else.
        return -stereoHudNdcOffset(eye, hudDepthSlider) * kPerspectiveToOrthoScale;
    }

    static void applyStereoHudShift(interop::float4x4 &projMatrix, StereoEye eye, uint32_t hudDepthSlider, bool isOrthographic) {
        const float signedOffset = stereoHudNdcOffset(eye, hudDepthSlider);
        if (signedOffset == 0.0f) {
            return;
        }
        if (isOrthographic) {
            projMatrix[3][0] -= signedOffset * kPerspectiveToOrthoScale;
        }
        else {
            projMatrix[2][0] += signedOffset * projMatrix[0][0];
        }
    }

    // Translate the camera laterally along its local right axis (view space +X)
    // by ±separationWorld/2. Combined with applyStereoOffAxis, this produces
    // depth-dependent parallax: objects at convergenceWorld have zero parallax,
    // closer objects pop out, farther objects push back. Without this shift,
    // every object gets the same constant disparity and the whole image just
    // slides sideways — the symptom the user reported as "can't get pop-out".
    static void applyStereoViewShift(interop::float4x4 &viewMatrix, StereoEye eye, uint32_t separationSlider, uint32_t convergenceSlider) {
        if (eye == StereoEye::None) {
            return;
        }
        float separationWorld;
        float convergenceWorld;
        stereoWorldUnits(separationSlider, convergenceSlider, separationWorld, convergenceWorld);
        // Match the clamp applied to the projection shift so the view and
        // projection stay geometrically consistent at extreme slider settings.
        // The 0.16 factor mirrors the projection's 0.08 max-offset cap (since
        // the projection cap of 0.08 corresponds to sep/(2*conv) = 0.08, i.e.,
        // sep <= 0.16 * conv).
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

            // Apply HUD depth shift to HUD/menu projections so the user can move
            // them off the screen plane. When the slider is at 50 (neutral) this
            // is a no-op and the HUD stays flat as before.
            //
            // Unlike the Banjo port, an orthographic projection alone does NOT
            // qualify as HUD here: Dinosaur Planet renders its shadow textures
            // through camSetOrthoProjectionMatrix (shadowtex.c), and shifting
            // those per-eye would corrupt the shadows rather than move a UI layer.
            // Only an explicit HUD tag qualifies, so anything untagged — shadow
            // passes, FMV/cutscene playback — is left alone and cannot flicker as
            // the per-eye shifts go in opposite directions. isOrtho still selects
            // WHICH matrix element receives the shift.
            {
                const bool isOrtho = (proj.type == Projection::Type::Orthographic);
                const bool isHudProjection = isStereoHudProjectionId(curProjGroup.matrixId);
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
                // Same HUD shift on the previous-frame projection, for the same
                // reason the off-axis shift above is duplicated.
                {
                    const bool isOrtho = (proj.type == Projection::Type::Orthographic);
                    const bool isHudProjection = isStereoHudProjectionId(curProjGroup.matrixId);
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
            // motion vectors / prev-viewProj stay correct. Note: the skybox does
            // NOT get the view shift — combined with the projection off-axis, that
            // produces the maximum positive parallax (i.e., infinity).
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