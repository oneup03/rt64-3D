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

// Forward-declare so we don't pull the entire SDK into rt64's public surface.
namespace SR {
    class SRContext;
    class IDX12Weaver1;
}

namespace RT64 {
    // Thin wrapper around LeiaSR's SRContext + IDX12Weaver1. Owns the context
    // and weaver, exposes a single weave() entry point that the present queue
    // calls after composing the per-eye textures into a single SbS image.
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
        // Hidden behind unique_ptrs of forward-declared SDK types so this
        // header doesn't pull in sr/weaver/dx12weaver.h transitively. The
        // .cpp owns the full SDK includes.
        struct State;
        std::unique_ptr<State> state;
    };
} // namespace RT64

#endif // LEIASR_SUPPORTED
