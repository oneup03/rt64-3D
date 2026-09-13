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
    // Guarded because these are extern "C" HRESULT entry points into
    // delay-loaded DLLs, and a missing or mismatched DLL raises SEH out of the
    // delay-load helper, which a C++ catch cannot intercept. Holds no C++
    // objects with destructors, as __try/__except requires.
    static HRESULT setLensHintGuarded(bool enable, DWORD *raisedCode) {
        __try {
            return enable ? SimulatedReality::SREnableLensHint()
                          : SimulatedReality::SRDisableLensHint();
        }
        __except (EXCEPTION_EXECUTE_HANDLER) {
            *raisedCode = GetExceptionCode();
            return E_FAIL;
        }
    }

    struct LeiaSRWeaver::State {
        // SR-lib owns the SRContext and the IDX12Weaver1 behind this handle,
        // including the create()/deleteSRContext() pairing and destroying the
        // weaver before the context. Released with Delete(), not delete.
        SimulatedReality::SRInterfaceDX12 *sr = nullptr;
        DXGI_FORMAT lastOutputFormat = DXGI_FORMAT_UNKNOWN;
        // OUR request, not the panel's state - the service arbitrates between
        // every connected application and the panel may well be lensed because
        // something else asked. Tracked here so a release we never matched with
        // an enable stays a no-op rather than becoming the thing that loads the
        // SR DLLs on a machine that has none.
        bool lensHintRequested = false;
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
        // Release the lens BEFORE the context goes. The hint is a Sense owned by
        // the SRContext and dies with it, so a disable issued afterwards reaches
        // nothing and the service carries on seeing this process asking for the
        // lens - which on a switchable panel means it stays lensed with nobody
        // weaving. This ordering matters just as much on a mid-session weave
        // failure, which tears the context down early.
        setLensHint(false);

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

    void LeiaSRWeaver::setLensHint(bool enable) {
        if (state->lensHintRequested == enable) {
            return;
        }

        // Never ask FOR the lens without a context to ask through. Releasing it
        // without one is already handled above: with no context we cannot have
        // enabled it, so the request matches and we return before touching the
        // SDK.
        if (enable && (state->sr == nullptr)) {
            return;
        }

        DWORD raisedCode = 0;
        const HRESULT hr = setLensHintGuarded(enable, &raisedCode);

        // Record the request whatever came back, so a display that cannot honour
        // it is asked once rather than once per frame. E_NOINTERFACE is the
        // normal answer on a fixed-lens panel - there is no switchable layer to
        // hold a preference about - and is not worth reporting.
        state->lensHintRequested = enable;

        if (raisedCode != 0) {
            fprintf(stderr, "LeiaSR: lens hint raised exception 0x%08lX.\n",
                    static_cast<unsigned long>(raisedCode));
            fflush(stderr);
        }
        else if (FAILED(hr) && (hr != E_NOINTERFACE)) {
            fprintf(stderr, "LeiaSR: lens hint failed with 0x%08lX.\n",
                    static_cast<unsigned long>(hr));
            fflush(stderr);
        }
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
