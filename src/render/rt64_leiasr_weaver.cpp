//
// RT64
//

#include "rt64_leiasr_weaver.h"

#ifdef LEIASR_SUPPORTED

#include <cstdio>
#include <exception>
#include <windows.h>

#include "SR.hpp"

namespace RT64 {
    struct LeiaSRWeaver::State {
        // SR-lib owns the SRContext and the IDX12Weaver1 behind this handle,
        // including the create()/deleteSRContext() pairing and destroying the
        // weaver before the context. Released with Delete(), not delete.
        SimulatedReality::SRInterfaceDX12 *sr = nullptr;
        DXGI_FORMAT lastOutputFormat = DXGI_FORMAT_UNKNOWN;
    };

    LeiaSRWeaver::LeiaSRWeaver() : state(std::make_unique<State>()) {}

    LeiaSRWeaver::~LeiaSRWeaver() {
        shutdown();
    }

    bool LeiaSRWeaver::initialize(ID3D12Device *device, HWND window) {
        if (state->sr != nullptr) {
            return true;
        }
        if (device == nullptr || window == nullptr) {
            return false;
        }

        // One call replaces what used to be context creation, weaver creation
        // and context initialization here. SR-lib performs them in the required
        // order (initialize() strictly after the weaver exists, or eye tracking
        // silently never engages) and converts the SDK's exceptions — notably
        // ServerNotAvailableException when the SR service isn't running — into
        // an HRESULT, so nothing escapes into rt64.
        //
        // It also probes the delay-loaded SR DLLs — both the core runtime and
        // the DirectX weaver DLL — before touching any SDK entry point, so a
        // machine without SR Platform installed returns a failed HRESULT here
        // instead of raising SEH out of the delay-load helper. rt64 used to run
        // that probe itself; SR-lib does it for every backend now.
        SimulatedReality::SRInterfaceDX12 *sr = nullptr;
        const HRESULT hr = SimulatedReality::CreateSRInterfaceDX12(device, window, &sr);
        if (FAILED(hr) || sr == nullptr) {
            fprintf(stderr, "LeiaSR: CreateSRInterfaceDX12 failed (hr 0x%08lX). Falling back to non-Leia presentation.\n",
                    static_cast<unsigned long>(hr));
            return false;
        }

        state->sr = sr;

        // The compose intermediate is plain UNORM RGBA (linear values in an
        // UNORM container) and so is the swap chain. Tell the weaver not to do
        // shader-side sRGB conversions either direction so it doesn't
        // double-gamma what's already correct.
        //
        // This now lands after SRContext::initialize() rather than before it,
        // since SR-lib does the initialize inside the create call. It is weaver
        // state, not context state, so the ordering doesn't matter to the SDK.
        state->sr->SetShaderSRGBConversion(false, false);

        return true;
    }

    void LeiaSRWeaver::shutdown() {
        // Delete() destroys the weaver and then the context, in that order.
        if (state->sr != nullptr) {
            state->sr->Delete();
            state->sr = nullptr;
        }
        state->lastOutputFormat = DXGI_FORMAT_UNKNOWN;
    }

    bool LeiaSRWeaver::isAvailable() const {
        return state->sr != nullptr;
    }

    void LeiaSRWeaver::weave(ID3D12Resource *inputTexture, DXGI_FORMAT outputFormat,
                             ID3D12GraphicsCommandList *commandList,
                             const D3D12_VIEWPORT &viewport, const D3D12_RECT &scissorRect) {
        if (state->sr == nullptr || inputTexture == nullptr || commandList == nullptr) {
            return;
        }

        // Re-bound every frame: the pointer can be the same resource while the
        // SDK's internal cached view has been invalidated by external state
        // changes (resize, etc). SR-lib reads the width, height and format off
        // the resource desc.
        state->sr->SetInputTexture(inputTexture);

        if (outputFormat != state->lastOutputFormat) {
            state->sr->SetOutputFormat(outputFormat);
            state->lastOutputFormat = outputFormat;
        }

        // The caller is responsible for having already reset the command list's
        // own rasterizer viewport/scissor to the swap-chain extent — D3D12
        // rasterizes against RSSetViewports, not against what the weaver is
        // told here. See the note in SR.hpp.
        try {
            state->sr->Weave(commandList, viewport, scissorRect);
        }
        catch (const std::exception &e) {
            fprintf(stderr, "LeiaSR: weave() threw: %s\n", e.what());
        }
        catch (...) {
            fprintf(stderr, "LeiaSR: weave() threw unknown exception\n");
        }
    }
} // namespace RT64

#endif // LEIASR_SUPPORTED
