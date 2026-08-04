//
// RT64
//

#pragma once

// Only compiled when the LeiaSR SDK was found by CMake. Falls back to a no-op
// shim so the rest of the renderer can include this header unconditionally.
#ifdef LEIASR_SUPPORTED

#include <cstdint>
#include <d3d12.h>
#include <dxgi.h>
#include <memory>

namespace RT64 {
    // Thin wrapper around SR-lib's SimulatedReality::SRInterfaceDX12, which in
    // turn owns the SDK's SRContext + IDX12Weaver1. Exposes a single weave()
    // entry point that the present queue calls after composing the per-eye
    // textures into a single SbS image.
    //
    // The SDK-facing details (context lifetime, weaver creation ordering,
    // destroy-not-delete, the DX12 per-frame setter dance) live in SR-lib now,
    // so this is only the rt64-shaped adapter around them.
    //
    // The SDK's runtime requires the SR Platform service installed on the
    // user's machine. If it's not present, initialize() returns false and the
    // caller should fall back to a non-Leia presentation path (the existing
    // SbS-to-swap-chain blit).
    struct LeiaSRWeaver {
        LeiaSRWeaver();
        ~LeiaSRWeaver();

        // Creates the SR context and DX12 weaver. Safe to call multiple times;
        // returns the existing state if already initialized. Returns false if
        // the SR Platform service is missing or the device/window are null.
        bool initialize(ID3D12Device *device, HWND window);

        // Tears down the weaver and the context. Call before the D3D12 device
        // goes away (e.g. on resize or shutdown). Safe to call from any state.
        void shutdown();

        // True if initialize() succeeded and the weaver is ready to weave.
        bool isAvailable() const;

        // Configures the weaver for a single weave() call:
        //   - inputTexture: SbS-packed stereo image. Must be in SHADER_READ.
        //   - inputWidth/Height: dimensions of inputTexture in pixels.
        //   - inputFormat: DXGI format of inputTexture.
        //   - outputFormat: DXGI format of the current render target.
        //   - commandList: command list to record weave commands into.
        //   - viewport / scissorRect: where on the bound render target to
        //     write. Typically full swap-chain extent.
        //
        // Records weaving commands into commandList. Caller is responsible for
        // binding the destination render target and managing barriers around
        // the call.
        void weave(ID3D12Resource *inputTexture, int inputWidth, int inputHeight,
                   DXGI_FORMAT inputFormat, DXGI_FORMAT outputFormat,
                   ID3D12GraphicsCommandList *commandList,
                   const D3D12_VIEWPORT &viewport, const D3D12_RECT &scissorRect);

    private:
        // Pimpl so this header doesn't pull in SR.hpp (and through it d3d9.h /
        // d3d11_1.h) transitively. The .cpp owns that include.
        struct State;
        std::unique_ptr<State> state;
    };
} // namespace RT64

#endif // LEIASR_SUPPORTED
