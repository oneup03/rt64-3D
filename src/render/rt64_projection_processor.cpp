//
// RT64
//

#include "rt64_projection_processor.h"

#include <algorithm>
#include <atomic>

#include "common/rt64_math.h"
#include "hle/rt64_workload_queue.h"

namespace RT64 {
    // Published from the render thread (the only writer) and read from the same
    // thread by the depth sampler's caller a few lines later, so a plain pair of
    // atomics is enough — no ordering between them matters, since a torn read
    // across a FOV change is one frame of a slightly wrong distance on a value
    // that is temporally smoothed anyway.
    static std::atomic<float> stereoWorldProjM22{0.0f};
    static std::atomic<float> stereoWorldProjM32{0.0f};
    static std::atomic<bool> stereoWorldDepthTermsValid{false};

    static std::atomic<float> stereoWorldVpScaleZ{0.5f};
    static std::atomic<float> stereoWorldVpTranslateZ{0.5f};

    void stereoPublishWorldDepthTerms(float m22, float m32, float vpScaleZ, float vpTranslateZ) {
        stereoWorldProjM22.store(m22, std::memory_order_relaxed);
        stereoWorldProjM32.store(m32, std::memory_order_relaxed);
        stereoWorldVpScaleZ.store(vpScaleZ, std::memory_order_relaxed);
        stereoWorldVpTranslateZ.store(vpTranslateZ, std::memory_order_relaxed);
        stereoWorldDepthTermsValid.store(true, std::memory_order_relaxed);
    }

    bool stereoGetWorldDepthTerms(float &m22, float &m32, float &vpScaleZ, float &vpTranslateZ) {
        if (!stereoWorldDepthTermsValid.load(std::memory_order_relaxed)) {
            return false;
        }
        m22 = stereoWorldProjM22.load(std::memory_order_relaxed);
        m32 = stereoWorldProjM32.load(std::memory_order_relaxed);
        vpScaleZ = stereoWorldVpScaleZ.load(std::memory_order_relaxed);
        vpTranslateZ = stereoWorldVpTranslateZ.load(std::memory_order_relaxed);
        return true;
    }

    static std::atomic<float> stereoAimViewZ{-1.0f};
    static std::atomic<int32_t> stereoAimScreenX{-1};
    static std::atomic<int32_t> stereoAimScreenY{-1};

    void stereoSetAimScreenPoint(int32_t x, int32_t y) {
        stereoAimScreenX.store(x, std::memory_order_relaxed);
        stereoAimScreenY.store(y, std::memory_order_relaxed);
    }

    bool stereoGetAimScreenPoint(int32_t &x, int32_t &y) {
        x = stereoAimScreenX.load(std::memory_order_relaxed);
        y = stereoAimScreenY.load(std::memory_order_relaxed);
        return (x >= 0) && (y >= 0);
    }

    void stereoStoreAimViewZ(float z) {
        stereoAimViewZ.store(z, std::memory_order_relaxed);
    }

    float stereoLoadAimViewZ() {
        return stereoAimViewZ.load(std::memory_order_relaxed);
    }

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
    // A RANGE, because the patches now tag each sky-family source with its own id
    // so the inspector can name which one a draw came from. They all get the same
    // treatment here. See patches/include/matrix_groups.h.
    static constexpr uint32_t DINO_PROJECTION_SKYBOX_ID_FIRST           = 0x00001001;
    static constexpr uint32_t DINO_PROJECTION_SKYBOX_ID_LAST            = 0x0000100F;
    // Moved clear of that range. Nothing emits this today -- the HUD is shifted
    // through the texture-rectangle path instead -- but leaving it inside the
    // skybox span would make the two collide the moment something did.
    static constexpr uint32_t DINO_PROJECTION_HUD_TRANSFORM_ID          = 0x00001020;

    static bool isStereoCameraProjectionId(uint32_t matrixId) {
        return (matrixId >= DINO_PROJECTION_CAMERA_TRANSFORM_ID_START) &&
               (matrixId <= DINO_PROJECTION_CAMERA_TRANSFORM_ID_END);
    }

    // Projections that get the off-axis shear: world geometry and the skybox.
    static bool isStereoProjectionId(uint32_t matrixId) {
        return isStereoCameraProjectionId(matrixId) ||
               ((matrixId >= DINO_PROJECTION_SKYBOX_ID_FIRST) &&
                (matrixId <= DINO_PROJECTION_SKYBOX_ID_LAST));
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

    // dynamic3d 1.3 — the clip-space stereo parameterization.
    //
    // `separation` IS the stereo knob: the per-eye projection shear is exactly
    // this value, with no FOV term and no convergence term folded into it.
    // Because the shear is the NDC x offset at infinity, the number means
    // something the user can see:
    //
    //     total background disparity = separation x screen width
    //
    // The 0..100 slider maps linearly onto 0..0.10 of screen width. The default
    // (18 -> 0.018) reproduces what the previous (sep / 2 / conv) * m[0][0] form
    // produced at the old default sliders (sep 50, conv 20) on a 16:9 window,
    // which was 0.01827 — so the shipped look carries over unchanged.
    //
    // Calibration reference: the divergence ceiling — where background disparity
    // reaches an adult IPD, the eyes are forced outward, and no amount of
    // practice can fuse it — is IPD / screen width. That is about 0.105 on a
    // 27-inch 16:9 monitor and proportionally lower on anything bigger (~0.05 on
    // a 55-inch TV). The top of the slider sits just under it; values that high
    // stay reachable because the physical screen size is not visible from here.
    static constexpr float kSeparationPerSlider = 0.10f / 100.0f;

    static float stereoSeparation(uint32_t separationSlider) {
        return static_cast<float>(separationSlider) * kSeparationPerSlider;
    }

    // The separation the HUD constants further down were tuned against. Those
    // constants were matched by eye back when the HUD offset was a FIXED value
    // that did not scale with separation at all, so the anchor is whatever
    // separation was in effect then — the default, not the top of the slider.
    //
    // This is a calibration anchor, not a default to keep in sync with anything;
    // changing it rescales every HUD offset.
    static constexpr float kHudReferenceSeparation = 18.0f * kSeparationPerSlider;

    // Convergence stays what it always was: the distance at which geometry sits
    // exactly on the screen plane. It arrives in TENTHS of a slider unit
    // (1..1000 = 0.1..100), so the game-unit conversion is 1 per tenth — the
    // 0.1..100 slider spans 1..1000 game units, and whole slider values land on
    // exactly the world units they did under the old `slider * 10` mapping.
    //
    // Dino's world scale: near plane 4.0, far plane 10000.0 (camera.c), and the
    // camera DLLs work at 20..250 units from the subject (84_camnormal,
    // 91_camlockon, 92_camshipbattle, 86_cam1stperson), so the default of 200
    // units sits just past the far end of the normal camera distance.
    //
    // The tenths exist for the depth-driven convergence loop, which solves for a
    // continuous value: at whole slider units its output would quantise to steps
    // of 10 game units, which is coarse enough to see as stepping.
    static float stereoConvergenceWorld(uint32_t convergenceTenths) {
        return static_cast<float>(convergenceTenths) * 1.0f;
    }

    // Dinosaur Planet's horizontal projection scale at the aspect ratio the HUD
    // constants below were tuned at (16:9). Used in place of the live m[0][0] so
    // HUD depth stops tracking the output aspect ratio.
    //
    // Derived from the game rather than measured:
    //   gFovY   = 60.0, gAspect = 1.3333334  (camera.c camInit / camResetProjection)
    //   guPerspectiveF gives m[1][1] = cot(30 deg) = 1.7320508 and
    //                        m[0][0] = 1.7320508 / 1.3333334 = 1.2990381
    //   RT64 then applies adjustProjectionMatrix(proj, 1 / aspectRatioScale) for
    //   widescreen, and at a 16:9 output over a 4:3 source that scale is
    //   (16/9) / (4/3) = 1.3333333, giving 1.2990381 / 1.3333333 = 0.9742786.
    //
    // Deliberately frozen. camSetFOV drives the live FOV continuously (lock-on,
    // first-person, cutscene framing, clamped 40..90 degrees), and RT64 rescales
    // m[0][0] for the window shape, so using the live term would make HUD depth
    // track both.
    static constexpr float kReferenceProjectionScale = 0.9742786f;

    // dynamic3d 1.1 — the shear is the knob.
    //
    // This used to be (sep / 2 / conv) * m[0][0] with a separate FOV auto-scale
    // multiplied into sep. Both are gone:
    //
    //   * The m[0][0] term rode on the live projection scale, which RT64 has
    //     already narrowed by 1/aspectRatioScale for widescreen — so the depth
    //     effect quietly scaled with the output aspect ratio. Clip space is
    //     invariant to that by construction.
    //   * The FOV auto-scale multiplied separation by tan(fov/2)/tan(refFov/2)
    //     and the projection then divided it straight back out through m[0][0].
    //     The two cancelled exactly (dynamic3d 3), so the whole block — and the
    //     reference-FOV constant it needed — was doing nothing but adding a
    //     clamp. Clip-space separation is FOV-independent for free, and exact on
    //     the first frame of a hard FOV cut rather than after an EMA settles.
    //
    // The old cap on the shear is gone with them. It existed because
    // sep / (2 * conv) grows without bound as convergence approaches zero; the
    // shear is now just `separation`, which the slider range already bounds, so
    // there is nothing left for the cap to protect against (dynamic3d 2.4).
    static void applyStereoOffAxis(interop::float4x4 &projMatrix, StereoEye eye, uint32_t separationSlider) {
        if (eye == StereoEye::None) {
            return;
        }
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        // Adds an asymmetric horizontal shift to the projection's principal point.
        // Equivalent to rebuilding the frustum with [L,R] = [-horFov+offset, horFov+offset].
        projMatrix[2][0] += eyeSign * stereoSeparation(separationSlider);
    }

    // Apply a constant per-eye horizontal shift to a HUD/UI projection matrix so
    // it sits at a user-selected stereo depth instead of flat on the screen.
    // hudDepthSlider 0..100: 50 = screen plane (no shift), below = push behind
    // the screen (positive parallax), above = pop out (negative parallax).
    //
    // Perspective and orthographic projections need different elements:
    //   - Perspective: m[2][0] gets the principal-point offset, scaled by the
    //     REFERENCE projection term rather than the live one so the depth holds
    //     across aspect ratios and in-game FOV changes.
    //   - Orthographic: there is no perspective divide, so we add a direct NDC
    //     shift via m[3][0].
    // Slider 0..100 -> signed NDC-space HUD offset, before the per-projection-type
    // scaling below. Shared by the projection path and the texture-rectangle path.
    static constexpr float kMaxHudOffset = 0.04f;

    // For perspective, m[2][0] += K produces an NDC.x shift of -K (after the
    // right-handed perspective divide). For ortho, m[3][0] += K produces an NDC.x
    // shift of +K directly. To make the slider push HUD the same direction across
    // both projection types we apply the opposite sign to ortho. The 1.732 was
    // matched by eye against the perspective path so text and icons sit at the
    // same depth at the same slider value.
    static constexpr float kPerspectiveToOrthoScale = 1.7320508f;

    // dynamic3d 5.2: a layer parked at a fixed multiple of the convergence
    // distance has a shift of separation * (1/factor - 1) — proportional to
    // separation, and independent of convergence. Scaling by separation is what
    // makes HUD depth track the depth knob, and what makes the HUD go properly
    // flat when separation is 0 (the old fixed offset split the HUD even with the
    // 3D effect dialled all the way down). The per-projection-type constants
    // above are that (1/factor - 1) mapping and keep their empirically tuned
    // values, so the HUD is unchanged at the default separation.
    static float stereoHudNdcOffset(StereoEye eye, uint32_t hudDepthSlider, uint32_t separationSlider) {
        if ((eye == StereoEye::None) || (hudDepthSlider == 50)) {
            return 0.0f;
        }
        const float centered = (static_cast<float>(hudDepthSlider) - 50.0f) / 50.0f; // -1..+1
        // Negate so slider > 50 produces pop-out (negative parallax) and
        // slider < 50 produces push-back (positive parallax).
        const float separationScale = stereoSeparation(separationSlider) / kHudReferenceSeparation;
        const float hudOffset = -centered * kMaxHudOffset * separationScale;
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        return eyeSign * hudOffset;
    }

    // dynamic3d 6.1 — pop-out is not divergence, so the two directions do not get
    // the same limit. Behind the screen plane the constraint is physical:
    // uncrossed disparity past an IPD forces the eyes outward and cannot be
    // fused. In front of it the eyes converge inward and there is nothing to
    // protect against, so reusing the behind-limit would only clip valid pop-out.
    //
    // Branch on the EYE-INDEPENDENT offset, never on the sign of the applied
    // shift: eyeSign is folded into the latter, so its sign encodes which eye,
    // not which side of the screen plane the element is on.
    static float clampHudNdcOffset(float ndcOffset, float unsignedHudOffset) {
        constexpr float BehindNdcLimit = 0.10f;   // ~5% of eye width
        constexpr float PopOutNdcLimit = 0.30f;   // ~15% of eye width
        const float ndcLimit = (unsignedHudOffset > 0.0f) ? BehindNdcLimit : PopOutNdcLimit;
        return std::clamp(ndcOffset, -ndcLimit, ndcLimit);
    }

    // Depth-aware crosshair (dynamic3d 5.1). Same NDC pipeline as the HUD
    // offset below -- same ortho scale, same asymmetric clamp, same final
    // negation -- so the reticle moves in the same direction as every other
    // rect. Only the source of the offset differs: a sampled depth rather than
    // a slider position.
    //
    // dynamic3d 1.2 in NDC: zero at the convergence distance, tending to
    // +separation (full background disparity, behind the glass) as the aim
    // point recedes, and going negative (pop-out) nearer than convergence.
    // The sign convention is the HUD's: positive is BEHIND the screen plane,
    // which is what clampHudNdcOffset branches on.
    //
    // With no usable depth the reticle recedes to infinity rather than falling
    // back to the screen plane. Aiming at open sky should put it far away, and
    // a reticle that snaps forward to the glass whenever the sample drops out
    // is far more jarring than one that sits deep.
    float stereoAimRectOffsetX(StereoEye eye, float aimViewZ, uint32_t separationSlider,
                               uint32_t convergenceTenths) {
        if ((eye == StereoEye::None) || (separationSlider == 0)) {
            return 0.0f;
        }

        const float separation = stereoSeparation(separationSlider);
        const float convergence = stereoConvergenceWorld(convergenceTenths);

        // No usable depth resolves to INFINITY, never to the screen plane and
        // never to HUD depth. Aiming at open sky, or at anything the sampler
        // cannot resolve, should leave the reticle sitting deep: one that snaps
        // forward to the glass whenever a sample drops out is far more
        // distracting than one that sits too far away, and the sample drops out
        // exactly when there is nothing near to look at.
        //
        // ratio is convergence/z, so ratio = 0 IS the z -> infinity limit, and
        // the offset below becomes +separation -- the same background disparity
        // the world projection gives geometry at infinity. Spelled out rather
        // than left to fall out of the arithmetic, because it is a deliberate
        // behavioural choice that a later simplification could quietly undo.
        constexpr float AtInfinityRatio = 0.0f;
        const float ratio = (aimViewZ > 0.0f) ? (convergence / aimViewZ) : AtInfinityRatio;
        const float aimOffset = -separation * (ratio - 1.0f);
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        const float signedOffset = eyeSign * aimOffset;
        if (signedOffset == 0.0f) {
            return 0.0f;
        }

        // NO kPerspectiveToOrthoScale here, unlike the HUD path below.
        //
        // That constant is a HUD calibration -- "matched by eye" so that text and
        // icons sit at the same apparent depth whether they arrive through a
        // perspective or an orthographic projection. It answers a question about
        // agreement between two UI paths. This function answers a different one:
        // the reticle has to land on the same disparity the WORLD gets at the
        // depth being aimed at, and that quantity is fixed by the projection, not
        // by a UI convention.
        //
        // Working the two stereo transforms through for a point at distance d:
        //
        //   off-axis shear  projMatrix[2][0] += eyeSign * separation
        //                   -> dNDC.x = -eyeSign * separation          (constant)
        //   view shift      viewMatrix[3][0] += eyeSign * separation * conv / m00
        //                   -> dNDC.x = +eyeSign * separation * conv / d
        //
        //   total           eyeSign * separation * (conv/d - 1)
        //
        // which is exactly -signedOffset. Multiplying by 1.732 on top put the
        // reticle at 1.73x the disparity the world uses, so it sat well behind
        // whatever it was pointed at -- worst in the mid range, where disparity
        // is large enough to see but the reticle has not yet saturated at the
        // background plane. It vanishes at the convergence distance, where both
        // the correct and the inflated value are zero.
        //
        // screenOffset.x is in the same NDC the projection produces (RasterVS adds
        // it after the scale, as `screenOffset * ndcPos.w` against a clip position),
        // so no unit conversion belongs here either.
        return -clampHudNdcOffset(signedOffset, aimOffset);
    }

    float stereoHudRectOffsetX(StereoEye eye, uint32_t hudDepthSlider, uint32_t separationSlider) {
        const float signedOffset = stereoHudNdcOffset(eye, hudDepthSlider, separationSlider);
        if (signedOffset == 0.0f) {
            return 0.0f;
        }
        // Mirrors the orthographic branch below — same scale, same negation — so
        // rectangles shift in the same direction and by the same amount as
        // ortho-projected UI. The unsigned offset used for the clamp branch is
        // recovered by dividing eyeSign back out.
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        const float unsignedHudOffset = signedOffset * eyeSign;
        return -clampHudNdcOffset(signedOffset * kPerspectiveToOrthoScale, unsignedHudOffset);
    }

    static void applyStereoHudShift(interop::float4x4 &projMatrix, StereoEye eye, uint32_t hudDepthSlider,
                                    uint32_t separationSlider, bool isOrthographic) {
        const float signedOffset = stereoHudNdcOffset(eye, hudDepthSlider, separationSlider);
        if (signedOffset == 0.0f) {
            return;
        }
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        const float unsignedHudOffset = signedOffset * eyeSign;
        if (isOrthographic) {
            projMatrix[3][0] -= clampHudNdcOffset(signedOffset * kPerspectiveToOrthoScale, unsignedHudOffset);
        }
        else {
            projMatrix[2][0] += clampHudNdcOffset(signedOffset * kReferenceProjectionScale, unsignedHudOffset);
        }
    }

    // Translate the camera laterally along its local right axis (view space +X).
    // Combined with applyStereoOffAxis, this produces depth-dependent parallax:
    // objects at the convergence distance have zero parallax, closer objects pop
    // out, farther objects push back. Without this shift, every object gets the
    // same constant disparity and the whole image just slides sideways — the
    // symptom the user reported as "can't get pop-out".
    //
    // dynamic3d 1.1: under clip-space separation the eye baseline is DERIVED per
    // frame rather than stored, and it moves with BOTH the FOV and the
    // convergence distance — 2 * separation * tan(half horizontal FOV) * conv for
    // the pair. That is what pins zero parallax at the convergence distance while
    // background disparity stays fixed at `separation`. Anything that wants a
    // physical eye offset has to read it from here rather than assuming the
    // separation slider is one.
    //
    // The old clamp that mirrored the projection's max-offset cap is gone with
    // that cap: there is no longer a divide by convergence to protect.
    static void applyStereoViewShift(interop::float4x4 &viewMatrix, StereoEye eye, uint32_t separationSlider,
                                     uint32_t convergenceTenths, float projectionScale) {
        if (eye == StereoEye::None) {
            return;
        }
        // projectionScale is the live m[0][0] after the widescreen adjust; its
        // reciprocal is tan(half horizontal FOV). Unlike the HUD path this WANTS
        // the live value — the baseline has to match the frustum the eye is
        // actually rendered with, or zero parallax stops landing at the
        // convergence distance. Guard against a degenerate projection rather
        // than dividing by ~0.
        if (projectionScale <= 1e-6f) {
            return;
        }
        const float tanHalfHorFov = 1.0f / projectionScale;
        const float halfBaseline = stereoSeparation(separationSlider) * tanHalfHorFov *
                                   stereoConvergenceWorld(convergenceTenths);
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        // For a row-vector view matrix, m[3][0] is the X translation in view
        // space. Adding to it shifts world points right in view space, which is
        // equivalent to the camera moving left in world space — what we want for
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

            // Snapshot the horizontal projection scale BEFORE the stereo shear
            // touches the matrix, for the derived eye baseline the view shift
            // needs (dynamic3d 1.1). The shear only writes m[2][0], so reading
            // it after would give the same answer today — but taking it here
            // makes that independence explicit rather than incidental.
            //
            // Both eye passes see the same projection at this point, so they
            // necessarily derive the same baseline.
            const float stereoProjectionScale = projMatrix[0][0];

            // Dinosaur Planet presents whole screens -- the file select, and
            // full-screen images drawn as a grid of textured tiles -- through the
            // ordinary camera projection, so they arrive tagged as world geometry
            // and take world stereo depth at whatever distance those tiles happen
            // to be modelled at.
            //
            // What marks them out is that the FRAME renders no depth at all --
            // depthRead and depthWrite accumulate zCmp()/zUpd() per pair, and no
            // pair in the workload sets either. A frame that renders no depth has
            // no world in it to be at a distance from.
            //
            // The question has to be asked of the whole frame rather than of this
            // one pair. A flush can split a run of Z-disabled draws into a pair of
            // their own mid-gameplay -- rain does exactly this -- and that pair on
            // its own looks identical to a menu screen. Flattening it took the
            // weather out of the world.
            //
            // Such a screen gets NO stereo treatment at all: not the world shear,
            // not the lateral view shift, and not the HUD depth offset either. It
            // is a flat image, and the screen plane is the only place a flat image
            // belongs -- parking it at HUD depth would push the entire picture off
            // the glass just as wrongly as world depth scatters it. The matching
            // suppression for texture rectangles, which is how the file select
            // screen draws, is in FramebufferRenderer.
            //
            // Frame-level rather than per-projection: if the frame draws no depth
            // then nothing in it does, whatever projection it arrived under.
            const bool isFlatPresentation = !workload.anyDepthUsed();

            // Publish the world projection's depth terms for the auto-convergence
            // depth sampler. The camera projection only — the skybox shares the
            // stereo shear but not the depth range, and inverting a sampled depth
            // with the wrong one puts every distance in the wrong place.
            //
            // These elements are untouched by the stereo shear (which only writes
            // m[2][0]), so it does not matter that this runs before it.
            // A flat presentation is excluded for the same reason the skybox is:
            // its depth range describes nothing the sampler will read back.
            if ((proj.type == Projection::Type::Perspective) &&
                isStereoCameraProjectionId(curProjGroup.matrixId) &&
                !isFlatPresentation) {
                const interop::RSPViewport &depthViewport = drawData.rspViewports[proj.transformsIndex];
                stereoPublishWorldDepthTerms(projMatrix[2][2], projMatrix[3][2],
                    depthViewport.scale[2], depthViewport.translate[2]);
            }

            // Apply stereoscopic off-axis projection offset for world (gameplay)
            // and skybox projections.
            if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                (proj.type == Projection::Type::Perspective) &&
                isStereoProjectionId(curProjGroup.matrixId) &&
                !isFlatPresentation) {
                applyStereoOffAxis(projMatrix, p.stereoEye, p.stereoSeparation);
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
                    !isFlatPresentation &&
                    (p.stereoHudDepth != 50)) {
                    applyStereoHudShift(projMatrix, p.stereoEye, p.stereoHudDepth, p.stereoSeparation, isOrtho);
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
                    isStereoProjectionId(curProjGroup.matrixId) &&
                    !isFlatPresentation) {
                    // Deliberately the CURRENT frame's scale, not one derived from
                    // adjustedPrevProj: the point is that both lerp endpoints carry
                    // an identical stereo transform. Deriving it per-endpoint would
                    // reintroduce the jitter this duplication exists to prevent.
                    applyStereoOffAxis(adjustedPrevProj, p.stereoEye, p.stereoSeparation);
                }
                // Same HUD shift on the previous-frame projection, for the same
                // reason the off-axis shift above is duplicated.
                {
                    const bool isOrtho = (proj.type == Projection::Type::Orthographic);
                    const bool isHudProjection = isStereoHudProjectionId(curProjGroup.matrixId);
                    if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                        isHudProjection &&
                        !isFlatPresentation &&
                        (p.stereoHudDepth != 50)) {
                        applyStereoHudShift(adjustedPrevProj, p.stereoEye, p.stereoHudDepth, p.stereoSeparation, isOrtho);
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
                isStereoViewShiftProjectionId(curProjGroup.matrixId) &&
                !isFlatPresentation) {
                applyStereoViewShift(viewMatrix, p.stereoEye, p.stereoSeparation, p.stereoConvergence, stereoProjectionScale);
                applyStereoViewShift(prevViewTransform, p.stereoEye, p.stereoSeparation, p.stereoConvergence, stereoProjectionScale);
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