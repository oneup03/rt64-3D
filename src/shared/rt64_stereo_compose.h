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
    };
#ifdef HLSL_CPU
};
#endif
