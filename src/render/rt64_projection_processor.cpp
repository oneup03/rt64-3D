//
// RT64
//

#include "rt64_projection_processor.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <tuple>

#include "common/rt64_math.h"
#include "hle/rt64_workload_queue.h"

namespace RT64 {
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

    // Reticle bounds, accumulated while the frame's draws are matched and
    // promoted at the start of the next pass. Render-thread only, but published
    // through atomics to match the rest of the stereo bridge.
    static float reticlePendingMinX = 0.0f, reticlePendingMaxX = 0.0f;
    static float reticlePendingMinY = 0.0f, reticlePendingMaxY = 0.0f;
    static bool reticlePendingAny = false;
    static std::atomic<uint32_t> reticleCenterXBits{0};
    static std::atomic<uint32_t> reticleCenterYBits{0};
    static std::atomic<bool> reticleCenterValid{false};

    void stereoAccumulateReticleBounds(float minX, float maxX, float minY, float maxY) {
        if (!reticlePendingAny) {
            reticlePendingMinX = minX;
            reticlePendingMaxX = maxX;
            reticlePendingMinY = minY;
            reticlePendingMaxY = maxY;
            reticlePendingAny = true;
            return;
        }

        reticlePendingMinX = std::min(reticlePendingMinX, minX);
        reticlePendingMaxX = std::max(reticlePendingMaxX, maxX);
        reticlePendingMinY = std::min(reticlePendingMinY, minY);
        reticlePendingMaxY = std::max(reticlePendingMaxY, maxY);
    }

    void stereoPromoteReticleBounds() {
        if (reticlePendingAny) {
            const float cx = (reticlePendingMinX + reticlePendingMaxX) * 0.5f;
            const float cy = (reticlePendingMinY + reticlePendingMaxY) * 0.5f;
            uint32_t bx, by;
            std::memcpy(&bx, &cx, sizeof(bx));
            std::memcpy(&by, &cy, sizeof(by));
            reticleCenterXBits.store(bx, std::memory_order_relaxed);
            reticleCenterYBits.store(by, std::memory_order_relaxed);
            reticleCenterValid.store(true, std::memory_order_relaxed);
        }
        else {
            // No reticle drawn last pass. Drop the position rather than leave a
            // stale one: the next thing to need it may be a different scene.
            reticleCenterValid.store(false, std::memory_order_relaxed);
        }

        reticlePendingAny = false;
    }

    bool stereoGetReticleCenter(float &x, float &y) {
        if (!reticleCenterValid.load(std::memory_order_relaxed)) {
            return false;
        }

        const uint32_t bx = reticleCenterXBits.load(std::memory_order_relaxed);
        const uint32_t by = reticleCenterYBits.load(std::memory_order_relaxed);
        std::memcpy(&x, &bx, sizeof(x));
        std::memcpy(&y, &by, sizeof(y));
        return true;
    }

    inline void adjustProjectionMatrix(interop::float4x4 &matrix, const float aspectRatioScale) {
        matrix[0][0] *= aspectRatioScale;
        matrix[1][0] *= aspectRatioScale;
        matrix[2][0] *= aspectRatioScale;
        matrix[3][0] *= aspectRatioScale;
    }

    // DK64-specific transform IDs: see patches/common_structs.h (interpolationIDs).
    // The world (camera) projection receives full per-eye stereo: projection
    // off-axis (inter-eye disparity) + view translation (depth-dependent parallax).
    // The sky gradient shares the world's treatment rather than sitting at
    // infinity - see isStereoViewShiftProjectionId for why.
    // HUD/menu/overlay content is ORTHOGRAPHIC in DK64 and carries the world's
    // tag, so it is classified by projection type rather than by ID - see
    // isStereoHudTarget.
    //
    // ⚠ DK64 does NOT use Banjo's 0x1000-block projection IDs. 0x1000 here is
    // MTXTAG_ACTORS (0x100 IDs allocated per actor, so 0x1000..0x100FFF), and
    // matching it as the world projection would give every actor in the game
    // the camera's per-eye treatment. The IDs below are DK64's own, and the
    // HUD range is new space carved out below MTXTAG_MAINMENU_BARREL (0xF00).
    //
    // Occupied elsewhere in DK64's ID space, do not reuse: 0x000-0x005 (the
    // low enum tags), 0xF00 (main-menu barrel), 0x1000+ (actors), 0x101000+
    // (props), 0x102000+ (sprites), 0x200000+ (text), 0x400000+ (fluids,
    // solar flare, rap lyrics).
    static constexpr uint32_t DK64_MTXTAG_FRAMEBUFFERTRANSITION = 0x00000002;
    static constexpr uint32_t DK64_MTXTAG_SKYBOXBLEND           = 0x00000004;
    static constexpr uint32_t DK64_MTXTAG_CAMERAPROJECTION      = 0x00000005;

    // HUD projections are matched as a RANGE rather than one constant per
    // element, so the recompilation patches can add new HUD projection sites
    // without a matching edit over here. The upper half of the range is for
    // perspective HUD elements that need to land at the same depth as the
    // orthographic text drawn alongside them (Banjo's "dialog bubble" case).
    static constexpr uint32_t DK64_PROJECTION_HUD_ID_START      = 0x00000600;
    static constexpr uint32_t DK64_PROJECTION_HUD_ID_END        = 0x0000067F;
    static constexpr uint32_t DK64_PROJECTION_HUD_MATCHORTHO_ID_START = 0x00000680;
    static constexpr uint32_t DK64_PROJECTION_HUD_MATCHORTHO_ID_END   = 0x000006FF;

    // MTXTAG_PROJ_AT_INFINITY in patches/common_structs.h - the sun and its lens
    // flare, tagged at their draw site in patches/patches_matrix.c. They are
    // screen-space elements standing in for something infinitely far away, so
    // they belong at maximum positive parallax rather than on the screen plane.
    static constexpr uint32_t DK64_PROJECTION_AT_INFINITY_ID    = 0x00000700;
    static constexpr uint32_t DK64_PROJECTION_WEATHER_ID        = 0x00000710;

    static bool isStereoInfinityProjectionId(uint32_t matrixId) {
        return (matrixId == DK64_PROJECTION_AT_INFINITY_ID);
    }

    // At-infinity content, decided by projection TYPE as well as tag.
    //
    // Creepy Castle's moon is drawn as a two-triangle ORTHOGRAPHIC sprite
    // carrying the skybox tag. Orthographic normally means 2D content, which put
    // the moon on the screen plane, and at two triangles near the middle of the
    // frame it could also satisfy the first-person reticle test and be dragged
    // to aim depth. It is neither: it is sky, and sky belongs at maximum
    // positive parallax.
    //
    // The same tag on a PERSPECTIVE projection is the sky dome, and DK64's sky
    // functions do not restore the camera group afterwards, so world geometry
    // inherits it too. Those must keep the world treatment - shear plus view
    // shift - or the outdoor hub loses its convergence entirely. Hence the
    // split on type rather than on tag alone.
    static bool isStereoInfinityTarget(uint32_t matrixId, bool isOrtho) {
        if (isStereoInfinityProjectionId(matrixId)) {
            return true;
        }
        return isOrtho && (matrixId == DK64_MTXTAG_SKYBOXBLEND);
    }

    static bool isStereoProjectionId(uint32_t matrixId) {
        return (matrixId == DK64_MTXTAG_CAMERAPROJECTION) ||
               (matrixId == DK64_MTXTAG_SKYBOXBLEND);
    }

    // Which projections get the view-space eye baseline, i.e. depth-dependent
    // parallax. This is the ONLY place convergence acts, so anything excluded
    // here renders at a single flat depth that the convergence slider cannot
    // move.
    //
    // MTXTAG_SKYBOXBLEND is included even though Banjo excludes its skybox
    // equivalent, because in DK64 that tag is not exclusively the sky. The sky
    // gradient functions (func_global_asm_80704B20 / func_global_asm_807069A4
    // in patches/patches_framebuffer.c) *end* by emitting a SKYBOXBLEND
    // projection group and never restore the camera group, so in outdoor maps
    // every piece of world geometry drawn afterwards inherits it. Excluding it
    // gave the whole outdoor hub the at-infinity treatment: uniform disparity,
    // and convergence doing nothing.
    //
    // Since the two share a tag they must share a treatment, and world is the
    // correct one - DK64 draws its sky as real geometry at a finite distance
    // rather than a true infinite skybox, so parallax on it is not wrong. The
    // proper fix, if the sky ever needs to sit at infinity again, is on the
    // patch side: give the sky its own ID and re-tag the camera projection
    // after it, rather than reintroducing the split here.
    static bool isStereoViewShiftProjectionId(uint32_t matrixId) {
        return (matrixId == DK64_MTXTAG_CAMERAPROJECTION) ||
               (matrixId == DK64_MTXTAG_SKYBOXBLEND);
    }

    // Full-screen effect overlays that must stay welded to the screen plane.
    //
    // These are ORTHOGRAPHIC draws that cover the whole viewport - screen wipes
    // under the transition tag, and the sky blend, fog and underwater tint under
    // the skybox tag (both emitted by func_global_asm_80704B20 /
    // func_global_asm_807069A4 in patches/patches_framebuffer.c, which build a
    // full-screen quad rather than 2D content). They are part of the picture
    // rather than something laid on top of it, so moving them off-plane reads as
    // a coloured sheet floating in front of or behind the world.
    //
    // Note this only applies to the ORTHOGRAPHIC use of the skybox tag; the
    // perspective use is the sky geometry itself and keeps the world treatment.
    static bool isStereoScreenOverlayId(uint32_t matrixId) {
        return (matrixId == DK64_MTXTAG_FRAMEBUFFERTRANSITION) ||
               (matrixId == DK64_MTXTAG_SKYBOXBLEND) ||
               (matrixId == DK64_PROJECTION_WEATHER_ID);
    }

    // Perspective-only HUD IDs, for menus or overlays the game draws in 3D.
    // Orthographic content does not go through here - see isStereoHudTarget.
    static bool isStereoHudProjectionId(uint32_t matrixId) {
        return ((matrixId >= DK64_PROJECTION_HUD_ID_START) &&
                (matrixId <= DK64_PROJECTION_HUD_MATCHORTHO_ID_END));
    }

    // Does this projection get the user's configured HUD depth?
    //
    // DK64 reuses the world's projection-group tags for its 2D content: the
    // HUD, text boxes, menus and icons are all drawn as ORTHOGRAPHIC
    // projections that still carry MTXTAG_CAMERAPROJECTION (or
    // MTXTAG_SKYBOXBLEND, inherited the same way world geometry inherits it).
    // So the world test has to be type-aware -- an orthographic projection is
    // never world geometry no matter which tag it carries. Testing the ID
    // alone left every menu and text box pinned to the screen plane.
    // NOTE on the full-screen tints (underwater blue, sun glare):
    //
    // These still track HUD Depth and should not. They cannot be fixed here.
    // Measurement showed they are drawn under matrixId 5 through the SAME 4x
    // orthographic projection as the HUD text and icons - same tag, same
    // matrix, same projection group - so nothing visible at this layer tells
    // them apart. An attempt to separate them by projection scale (on the
    // theory that full-screen quads use DK64's 1x projection and HUD content
    // its 4x one) was measured to be wrong and has been removed rather than
    // left in as a special case that risks flattening 1x HUD content.
    //
    // The fix has to come from the recompilation patches: emit a distinct
    // projection group around the tint draws so they arrive here with an ID of
    // their own, then add that ID to isStereoScreenOverlayId.
    //
    // projScaleX is still threaded through for the bring-up logging, which is
    // what separates DK64's several same-tagged orthographic projections.
    static bool isStereoHudTarget(uint32_t matrixId, bool isPerspective, float projScaleX) {
        (void)projScaleX;
        // At-infinity content gets its own treatment and must not also be given
        // the HUD depth shift.
        if (isStereoInfinityTarget(matrixId, !isPerspective)) {
            return false;
        }
        if (isPerspective) {
            // World and sky keep their own treatment; only explicitly tagged
            // perspective HUD elements get the depth shift.
            return !isStereoProjectionId(matrixId) && isStereoHudProjectionId(matrixId);
        }
        // Everything orthographic is 2D content, except the full-screen effects
        // that carry a tag of their own.
        return !isStereoScreenOverlayId(matrixId);
    }

    // Perspective HUD elements that need a stronger shift than the default
    // perspective-HUD path provides, so they land at the same depth as the
    // orthographic text rectangles drawn alongside them.
    static bool isStereoBubbleProjectionId(uint32_t matrixId) {
        return ((matrixId >= DK64_PROJECTION_HUD_MATCHORTHO_ID_START) &&
                (matrixId <= DK64_PROJECTION_HUD_MATCHORTHO_ID_END));
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
    // exactly on the screen plane. It arrives in HUNDREDTHS of a slider unit
    // (10..2000 = 0.1..20), and the 0.1..20 slider spans 2..400 game units, so
    // one bridge unit is 0.2 game units.
    //
    // This was tenths (2 game units per step) and is the resolution the
    // depth-driven loop's output is rounded to on its way back through the
    // config bridge. At the close convergences auto-convergence pulls to, a
    // 2-unit step is several percent of the applied value; the disparity it
    // moves by is large enough to see, so the loop looked like it was stepping
    // rather than easing even though its own solve is continuous. Nothing about
    // the user-facing slider changed - it still moves in tenths.
    //
    // StereoAutoConvergence::update in the depth sampler applies this same
    // conversion by hand and rounds its result back into these units. The two
    // must stay in step, or the solve silently retunes itself.
    static constexpr float ConvergenceWorldPerUnit = 0.2f;

    static float stereoConvergenceWorld(uint32_t convergenceHundredths) {
        return static_cast<float>(convergenceHundredths) * ConvergenceWorldPerUnit;
    }

    // DK64's horizontal projection scale at the aspect ratio the HUD constants
    // below were tuned at (16:9). Used in place of the live m[0][0] so HUD
    // depth stops tracking the output aspect ratio.
    //
    // MEASURED from the live projection at 16:9, not derived. The game side of
    // it checks out against the ROM:
    //   fovy        = 45.0 (D_global_asm_807444B8, global_asm .data)
    //   game aspect = 1.0  (func_global_asm_8062A850() * D_global_asm_807444BC;
    //                       the multiplier is 1.0 because widescreen_enabled
    //                       defaults to 0 and the recomp leaves it there - RT64
    //                       does the widescreen expansion instead)
    //   so guPerspectiveF gives m[1][1] = cot(fovy/2) = 2.41420, which is
    //   exactly what the projection log reports.
    //
    // The horizontal term is aspect-dependent, and the projection log shows DK64
    // producing BOTH of the values you would predict:
    //     2.41420 / (16/9)  = 1.35799  (widescreen, the common case)
    //     2.41420 / (4/3)   = 1.81065  (4:3, e.g. character select)
    // so this constant is a choice of tuning aspect rather than a single truth.
    // 16:9 is the one to tune against, hence 1.35799.
    //
    // Deliberately frozen: DK64's FoV is a variable, not a compile-time
    // constant, so using the live m[0][0] would make HUD depth track both the
    // window shape and any in-game FoV change.
    static constexpr float ReferenceProjectionScale = 1.35799f;

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

    // Places a projection's contents at infinity: a constant per-eye disparity of
    // exactly `separation`, with no depth-dependent parallax, which is what
    // objects at an unbounded distance produce (dynamic3d 1.2, the z -> infinity
    // limit of shift_px).
    //
    // The orthographic form is the one that matters here: an ortho projection
    // has clip.w = 1, so ndc.x = x*m[0][0] + m[3][0] and a constant NDC offset is
    // just added to m[3][0]. The perspective form matches applyStereoOffAxis,
    // whose shear already yields exactly this in the z -> infinity limit.
    //
    // Sign: for the perspective case ndc.x tends to -m[2][0] as z recedes, i.e.
    // -eyeSign * separation, so the orthographic case has to subtract to match.
    // This mirrors applyStereoHudShift, which likewise adds for perspective and
    // subtracts for orthographic.
    static void applyStereoInfinityShift(interop::float4x4 &projMatrix, StereoEye eye,
                                         uint32_t separationSlider, bool isOrthographic) {
        if (eye == StereoEye::None) {
            return;
        }

        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        const float separation = stereoSeparation(separationSlider);
        if (isOrthographic) {
            projMatrix[3][0] -= eyeSign * separation;
        }
        else {
            projMatrix[2][0] += eyeSign * separation;
        }
    }

    // dynamic3d 6.1 - pop-out is not divergence, so the two directions do not
    // get the same limit. Behind the screen plane the constraint is physical:
    // uncrossed disparity past an IPD forces the eyes outward and cannot be
    // fused. In front of it the eyes converge inward and there is nothing to
    // protect against, so reusing the behind-limit would only clip valid
    // pop-out.
    //
    // unsignedOffset is the EYE-INDEPENDENT term the shift was built from,
    // positive when the element belongs BEHIND the screen plane. The branch has
    // to be on that and not on ndcOffset: eyeSign is folded into the applied
    // shift, so its sign encodes which eye rather than which side of the glass
    // the element is on.
    static float clampStereoNdcOffset(float ndcOffset, float unsignedOffset) {
        constexpr float BehindNdcLimit = 0.10f;   // ~5% of eye width
        constexpr float PopOutNdcLimit = 0.30f;   // ~15% of eye width
        const float ndcLimit = (unsignedOffset > 0.0f) ? BehindNdcLimit : PopOutNdcLimit;
        return std::max(-ndcLimit, std::min(ndcLimit, ndcOffset));
    }

    // Depth-aware crosshair (dynamic3d 5.1 / 1.2 in NDC).
    //
    // Zero at the convergence distance, tending to the full background
    // disparity as the aim point recedes, and going negative (pop-out) nearer
    // than convergence.
    //
    // NO orthographic scale factor here, unlike applyStereoHudShift below. That
    // 2.75 is a HUD calibration - matched by eye so that text and icons sit at
    // the same apparent depth whether they arrive through a perspective or an
    // orthographic projection - and it answers a question about agreement
    // between two UI paths. This function answers a different one: the reticle
    // has to land on the same disparity the WORLD gets at the depth being aimed
    // at, and that quantity is fixed by the projection, not by a UI convention.
    // dynamic3d 5.1 records a port that inherited such a constant and put its
    // reticle at 1.73x the world's disparity, correct at the convergence
    // distance and visibly too deep through the mid range.
    //
    // Working the two stereo transforms through for a point at distance d:
    //   off-axis shear  m[2][0] += eyeSign * separation
    //                     -> dNDC.x = -eyeSign * separation      (constant)
    //   view shift      m[3][0] += eyeSign * separation * conv / m00
    //                     -> dNDC.x = +eyeSign * separation * conv / d
    //   total             eyeSign * separation * (conv/d - 1)
    //
    // A non-positive aimViewZ means the sampler resolved nothing, and that
    // resolves to INFINITY rather than to the screen plane. Aiming at open sky
    // should leave the reticle sitting deep; one that snaps forward to the
    // glass whenever a sample drops out is far more distracting, and the sample
    // drops out exactly when there is nothing near to look at. Spelled out
    // rather than left to fall out of the arithmetic, because it is a
    // deliberate choice a later simplification could quietly undo.
    float stereoAimRectOffsetX(StereoEye eye, float aimViewZ, uint32_t separationSlider,
                               uint32_t convergenceHundredths) {
        if ((eye == StereoEye::None) || (separationSlider == 0)) {
            return 0.0f;
        }

        const float separation = stereoSeparation(separationSlider);
        const float convergence = stereoConvergenceWorld(convergenceHundredths);
        constexpr float AtInfinityRatio = 0.0f;
        const float ratio = (aimViewZ > 0.0f) ? (convergence / aimViewZ) : AtInfinityRatio;
        // Positive when the aim point is BEHIND the screen plane, which is the
        // sense clampStereoNdcOffset branches on.
        const float aimOffset = separation * (1.0f - ratio);
        const float eyeSign = (eye == StereoEye::Left) ? +1.0f : -1.0f;
        // Negated to mirror applyStereoHudShift's orthographic branch, so the
        // reticle moves in the same direction as every other rectangle.
        return -clampStereoNdcOffset(eyeSign * aimOffset, aimOffset);
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

        ndcOffset = clampStereoNdcOffset(ndcOffset, hudOffset);

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
    static void applyStereoViewShift(interop::float4x4 &viewMatrix, StereoEye eye, uint32_t separationSlider, uint32_t convergenceHundredths, float projectionScale) {
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
        const float halfBaseline = stereoSeparation(separationSlider) * tanHalfHorFov * stereoConvergenceWorld(convergenceHundredths);
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

                // Publish this projection's depth terms for the depth sampler.
                // The world projection is the one the sampled depth buffer was
                // rendered with, so these are the right terms to invert it.
                //
                // The VIEWPORT's depth scale and translate go with them
                // (dynamic3d 4.1). The buffer does not hold ndc.z: the RSP
                // viewport applies its own scale and translate on top, and on
                // the N64 that scale is a G_MAXZ fixed-point value slightly
                // under a half rather than exactly a half. Assuming 0.5 reads near
                // objects correctly and everything beyond progressively CLOSER
                // than it is - about -5% at 100 units and -20% at 1000 - which
                // shows up as a crosshair that sits short of mid-range targets
                // and a convergence loop quietly over-pulling because it
                // believes the scene is nearer than it is.
                //
                // Only published when the projection actually carries a
                // viewport; an identity one would describe a depth range
                // nothing was rendered with, so a zero scale is sent instead
                // and stereoDeviceDepthToViewZ falls back to the 2*d-1 form.
                if (isStereoViewShiftProjectionId(curProjGroup.matrixId)) {
                    float vpScaleZ = 0.0f;
                    float vpTranslateZ = 0.0f;
                    if (proj.usesViewport()) {
                        const interop::RSPViewport &depthViewport = drawData.rspViewports[proj.transformsIndex];
                        vpScaleZ = depthViewport.scale[2];
                        vpTranslateZ = depthViewport.translate[2];
                    }

                    stereoPublishWorldDepthTerms(projMatrix[2][2], projMatrix[3][2], vpScaleZ, vpTranslateZ);
                }
            }

            // Place at-infinity content (the sun and its lens flare) at maximum
            // positive parallax so it sits behind all world geometry.
            if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                isStereoInfinityTarget(curProjGroup.matrixId,
                    proj.type == Projection::Type::Orthographic)) {
                applyStereoInfinityShift(projMatrix, p.stereoEye, p.stereoSeparation,
                    proj.type == Projection::Type::Orthographic);
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
                const bool isHudProjection = isStereoHudTarget(curProjGroup.matrixId, !isOrtho, projMatrix[0][0]);
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

                // Same shift on the previous-frame projection, or the lerp that
                // follows produces a half-shifted matrix that jitters on every
                // interpolated frame.
                if ((p.stereoMode != UserConfiguration::StereoMode::Off) &&
                    isStereoInfinityTarget(curProjGroup.matrixId,
                        proj.type == Projection::Type::Orthographic)) {
                    applyStereoInfinityShift(adjustedPrevProj, p.stereoEye, p.stereoSeparation,
                        proj.type == Projection::Type::Orthographic);
                }
                {
                    const bool isOrtho = (proj.type == Projection::Type::Orthographic);
                    const bool isHudProjection = isStereoHudTarget(curProjGroup.matrixId, !isOrtho, projMatrix[0][0]);
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