//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // Push constants for the stereo composition pixel shader. Mirrors
    // RT64::UserConfiguration::StereoMode integer values:
    //   1 = Side-by-Side
    //   2 = Top-and-Bottom
    //   3 = Row Interlaced
    struct StereoComposeCB {
        float2 videoResolution;   // offset 0
        float2 textureResolution; // offset 8 — ends the first 16-byte chunk
        float gamma;              // offset 16
        uint stereoMode;          // offset 20
        // When non-zero, the shader bypasses VI-style sampling (videoResolution
        // / textureResolution clamping, gamma correction, opaque-alpha output)
        // and instead samples the source UV directly so a UI texture rendered
        // at the swap-chain's full resolution gets blitted into each eye half
        // with its original alpha intact.
        uint useUIOverlayMode;    // offset 24
        // Padding so the next field (a float2) doesn't cross the 32-byte
        // boundary. HLSL constant-buffer packing forbids vectors from
        // straddling 16-byte boundaries; without this pad the shader sees
        // contentOrigin at offset 32 while C++ writes it at 28, which
        // scrambled useUIOverlayMode and broke the UI overlay path.
        uint _pad0;               // offset 28
        // Top-left corner of the eye texture's content area, in normalized
        // texture-UV space. Zero when content fills the full texture; positive
        // when Expand mode placed the content in the middle of a wider canvas
        // with pillarbox bars on the sides. Used in stereo compose to skip
        // those pillarbox bars so each SbS/TaB/Interlaced eye slot shows the
        // game content directly. Default 0 keeps backward compatibility.
        float2 contentOrigin;     // offset 32

        // The right eye's equivalents of the three fields above.
        //
        // The two eye textures are separate render targets, created and resized
        // independently — RT64 grows a target when the game needs more and never
        // shrinks it, so the left and right targets can legitimately end up
        // different sizes. Normalizing both eyes by one texture resolution then
        // samples the wrong region of whichever eye did not match, which shows up
        // as one eye zoomed relative to the other.
        //
        // Offsets 40 and 56 both sit inside a 16-byte chunk (32..48 and 48..64),
        // so no additional padding is needed — but keep that in mind before
        // reordering, per the _pad0 note above.
        float2 rightVideoResolution;   // offset 40
        float2 rightTextureResolution; // offset 48
        float2 rightContentOrigin;     // offset 56 — closes the 64-byte chunk

        // Ghost-reduction (anti-crosstalk) range compression, output3d 3.4.
        // ghostContrast squeezes toward mid-grey, leaving (1-contrast)/2 of
        // headroom at each end of the range; ghostBlackFloor raises the black
        // floor only, which is where a cancelling display's own correction
        // clips. 1.0 and 0.0 respectively are exact no-ops and the shader skips
        // the math entirely at those values.
        //
        // Two floats land at 64 and 68, both inside the 64..80 chunk, so neither
        // straddles a boundary — but they take the struct to 72 bytes, and the
        // TAIL has to be padded out too.
        //
        // HLSL rounds a constant buffer up to a 16-byte multiple, so the shader
        // side of this is 80 bytes (20 dwords) whatever C++ says. The D3D12
        // backend sizes the root constants as ceil(sizeof / 4) = 18 dwords, and
        // a root signature that declares fewer constants than the shader's
        // cbuffer needs is rejected at pipeline creation. Padding to 80 makes
        // the two agree.
        //
        // Per the _pad0 note above, a float2 added at 72 would also straddle the
        // 80-byte boundary — so put any new vector after this padding, not
        // before it.
        float ghostContrast;      // offset 64
        float ghostBlackFloor;    // offset 68
        float _pad1;              // offset 72
        float _pad2;              // offset 76 — closes the 80-byte chunk
    };
#ifdef HLSL_CPU
};
#endif
