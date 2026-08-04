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
    // LeiaSR DLLs are delay-loaded so the exe can start without the SR Platform
    // installed. Probe them with LoadLibraryW once before any SDK call. If
    // either DLL is missing we silently disable LeiaSR for the session —
    // touching SDK entry points after that point would trigger the delay-load
    // helper's SEH exception and crash the game.
    //
    // SR-lib runs the same probe internally, but only for
    // SimulatedRealityCore.dll. CreateDX12Weaver lives in
    // SimulatedRealityDirectX.dll, which is delay-loaded separately, so we keep
    // checking that one here — a machine with Core present and DirectX missing
    // would otherwise get past SR-lib's guard and SEH on the weaver call.
    static bool sLeiaSRDllsChecked = false;
    static bool sLeiaSRDllsAvailable = false;

    static bool leiaSRDllsAvailable() {
        if (sLeiaSRDllsChecked) {
            return sLeiaSRDllsAvailable;
        }
        sLeiaSRDllsChecked = true;
        // Probe both DLLs the EXE statically references from the SR linkage.
        // SR Platform installs both into the system path; if either is
        // missing, treat the platform as absent.
        const HMODULE core = LoadLibraryW(L"SimulatedRealityCore.dll");
        const HMODULE dx   = LoadLibraryW(L"SimulatedRealityDirectX.dll");
        sLeiaSRDllsAvailable = (core != nullptr) && (dx != nullptr);
        if (!sLeiaSRDllsAvailable) {
            // We intentionally don't unload the ones that did load: another
            // DLL in the process may already depend on them.
            fprintf(stderr, "LeiaSR: SR Platform DLLs not found on this system; weaving disabled.\n");
        }
        return sLeiaSRDllsAvailable;
    }

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

        // Delay-loaded DLLs: probe with LoadLibraryW first. If the SR Platform
        // isn't installed, calling any SDK entry point would invoke the
        // delay-load helper which throws SEH on missing DLLs.
        if (!leiaSRDllsAvailable()) {
            return false;
        }

        // One call replaces what used to be context creation, weaver creation
        // and context initialization here. SR-lib performs them in the required
        // order (initialize() strictly after the weaver exists, or eye tracking
        // silently never engages) and converts the SDK's exceptions — notably
        // ServerNotAvailableException when the SR service isn't running — into
        // an HRESULT, so nothing escapes into rt64.
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

    void LeiaSRWeaver::weave(ID3D12Resource *inputTexture, int inputWidth, int inputHeight,
                             DXGI_FORMAT inputFormat, DXGI_FORMAT outputFormat,
                             ID3D12GraphicsCommandList *commandList,
                             const D3D12_VIEWPORT &viewport, const D3D12_RECT &scissorRect) {
        if (state->sr == nullptr || inputTexture == nullptr || commandList == nullptr) {
            return;
        }

        // Re-bound every frame: the pointer can be the same resource while the
        // SDK's internal cached view has been invalidated by external state
        // changes (resize, etc). SR-lib reads the width, height and format off
        // the resource desc, so the inputWidth/inputHeight/inputFormat we were
        // handed are no longer passed through — the descriptor is the same data
        // and cannot disagree with the actual allocation.
        (void)inputWidth;
        (void)inputHeight;
        (void)inputFormat;
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
