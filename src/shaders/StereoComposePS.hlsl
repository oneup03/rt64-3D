//
// RT64
//

#include "shared/rt64_stereo_compose.h"

[[vk::push_constant]] ConstantBuffer<StereoComposeCB> gConstants : register(b0);
Texture2D<float4> gLeftEye  : register(t1);
Texture2D<float4> gRightEye : register(t2);
SamplerState gSampler : register(s3);

// World pass: mirrors VideoInterfacePS::SampleInput (border clamping, gamma
// correction, opaque-alpha output) but allows the content area to live in
// the middle of the texture (Expand mode pillarboxes the texture). UI overlay
// pass: samples the source UV directly and returns the texel as-is so alpha-
// transparent regions stay transparent and the UI's full-screen layout isn't
// squeezed to a sub-region.
// videoRes / texRes / origin describe the content rect of THIS eye's texture.
// They are passed per eye rather than read from gConstants directly because the
// two eye textures are independent render targets and can differ in size.
float4 SampleEye(Texture2D<float4> tex, float2 uv, float2 videoRes, float2 texRes, float2 origin) {
    if (gConstants.useUIOverlayMode != 0) {
        // Slightly reduce the UI's opacity so gameplay remains visible through
        // it. 0.75 keeps the GUI clearly readable but lets enough of the world
        // bleed through to feel non-blocking.
        float4 c = tex.SampleLevel(gSampler, saturate(uv), 0);
        c.a *= 0.75f;
        return c;
    }
    // Content rect in normalized texture UV space:
    //   origin = origin
    //   extent = videoRes / texRes
    // For full-texture content (Original / non-Expand modes) origin is 0 and
    // extent is 1, matching the original VI sampling exactly.
    const float2 ContentExtent = videoRes / texRes;
    const float2 ContentMin = origin;
    const float2 ContentMax = ContentMin + ContentExtent;
    const float2 HalfPixel = float2(0.5f, 0.5f) / texRes;
    float2 outsideBorder = step(ContentMax, uv) + step(uv + HalfPixel, ContentMin);
    float4 sampledColor = tex.SampleLevel(gSampler, clamp(uv, ContentMin + HalfPixel, ContentMax - HalfPixel), 0);
    float4 gammaCorrectedColor = pow(sampledColor, gConstants.gamma);
    gammaCorrectedColor.rgb *= max(1.0f - outsideBorder.x - outsideBorder.y, 0.0f);
    gammaCorrectedColor.a = 1.0f;
    return gammaCorrectedColor;
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    // Default: pass through left eye (matches VideoInterfacePSRegular behavior so
    // that any unrecognized mode produces a usable mono frame instead of black).
    float2 leftSampleUv = uv;
    float2 rightSampleUv = uv;
    bool useRight = false;
    bool mixEyes = false;

    switch (gConstants.stereoMode) {
    case 1: // Side-by-Side: x in [0, 0.5) -> left, x in [0.5, 1] -> right.
        if (uv.x < 0.5f) {
            leftSampleUv = float2(uv.x * 2.0f, uv.y);
        } else {
            rightSampleUv = float2((uv.x - 0.5f) * 2.0f, uv.y);
            useRight = true;
        }
        break;
    case 2: // Top-and-Bottom: y in [0, 0.5) -> left, y in [0.5, 1] -> right.
        if (uv.y < 0.5f) {
            leftSampleUv = float2(uv.x, uv.y * 2.0f);
        } else {
            rightSampleUv = float2(uv.x, (uv.y - 0.5f) * 2.0f);
            useRight = true;
        }
        break;
    case 3: // Row interlaced: alternate scanlines (even = left, odd = right).
        useRight = (uint(pos.y) & 1u) == 1u;
        break;
    case 4: // Column interlaced: alternate pixel columns (even = left, odd = right).
        useRight = (uint(pos.x) & 1u) == 1u;
        break;
    case 5: // Checkerboard: alternate eye per 1x1 cell (sum of row+col parity).
        useRight = ((uint(pos.x) + uint(pos.y)) & 1u) == 1u;
        break;
    case 6: // Anaglyph (red/cyan): sample both eyes, mix channels.
        mixEyes = true;
        break;
    default:
        break;
    }

    // Map the per-eye uv (0..1 across each eye's slot) onto that eye texture's
    // content rect: [contentOrigin, contentOrigin + videoResolution /
    // textureResolution]. When contentOrigin is zero and videoResolution ==
    // textureResolution this collapses to the identity uv mapping, matching
    // the original VI sampling.
    //
    // Each eye uses its OWN rect. The eye textures are separate render targets
    // and can differ in size, in which case a shared mapping would sample the
    // wrong region of one of them and show it zoomed against the other.
    leftSampleUv  = gConstants.contentOrigin + (leftSampleUv  / gConstants.textureResolution) * gConstants.videoResolution;
    rightSampleUv = gConstants.rightContentOrigin + (rightSampleUv / gConstants.rightTextureResolution) * gConstants.rightVideoResolution;

    if (mixEyes) {
        // UI overlay path always has both eye slots bound to the same UI
        // texture and runs through an AlphaBlend pipeline that depends on
        // the sample's real alpha. Mixing channels and forcing alpha=1.0
        // would write solid (mix-rgb) across the whole swap chain — for
        // pixels where the UI is transparent the rgb is zero, so the entire
        // screen would blank out. Sidestep the mix in UI mode and return the
        // UI sample untouched so the overlay blends normally.
        if (gConstants.useUIOverlayMode != 0) {
            return SampleEye(gLeftEye, leftSampleUv, gConstants.videoResolution, gConstants.textureResolution, gConstants.contentOrigin);
        }
        // Red-cyan anaglyph using a full cross-talk RGB matrix. Each output
        // channel takes weighted contributions from BOTH eyes (positive from
        // its "own" eye, small negatives from the other) which cancels the
        // imperfect blocking of consumer red/cyan filters and produces a
        // noticeably less retinal-rivalry-prone image than the simpler
        // Dubois matrix that ignores cross-eye terms. Wear glasses with the
        // red lens on the LEFT eye.
        float4 cA = SampleEye(gLeftEye,  leftSampleUv, gConstants.videoResolution, gConstants.textureResolution, gConstants.contentOrigin);
        float4 cB = SampleEye(gRightEye, rightSampleUv, gConstants.rightVideoResolution, gConstants.rightTextureResolution, gConstants.rightContentOrigin);
        float r = saturate( 0.437 * cA.r + 0.449 * cA.g + 0.164 * cA.b
                           - 0.011 * cB.r - 0.032 * cB.g - 0.007 * cB.b);
        float g = saturate(-0.062 * cA.r - 0.062 * cA.g - 0.024 * cA.b
                           + 0.377 * cB.r + 0.761 * cB.g + 0.009 * cB.b);
        float b = saturate(-0.048 * cA.r - 0.050 * cA.g - 0.017 * cA.b
                           - 0.026 * cB.r - 0.093 * cB.g + 1.234 * cB.b);
        return float4(r, g, b, 1.0f);
    }
    if (useRight) {
        return SampleEye(gRightEye, rightSampleUv, gConstants.rightVideoResolution, gConstants.rightTextureResolution, gConstants.rightContentOrigin);
    }
    return SampleEye(gLeftEye, leftSampleUv, gConstants.videoResolution, gConstants.textureResolution, gConstants.contentOrigin);
}
