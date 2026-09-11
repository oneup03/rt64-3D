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
    // Structured-exception guard around the SDK entry point.
    //
    // SR-lib contains the two calls it knows can throw (SRContext::create and
    // initialize) but calls SR::CreateDX12Weaver bare, so anything that one
    // raises travels straight through an extern "C" HRESULT function whose whole
    // contract is to report failure as a return code, and out through rt64 into
    // the present thread — where nothing catches it and the process dies. That is
    // the wrong outcome for an optional output mode: failing to bring up the
    // weaver should drop us to Side-by-Side, never take the game down.
    //
    // __try/__except rather than catch(...) because the two failure shapes here
    // are not both C++ exceptions. A missing or mismatched delay-loaded DLL
    // raises SEH from the delay-load helper, which catch(...) cannot intercept
    // unless the whole translation unit is built /EHa. __except catches SEH and,
    // since MSVC implements C++ exceptions on top of SEH, C++ throws as well
    // (reported as 0xE06D7363).
    //
    // The tradeoff is that unwinding the throwing frame is skipped, so SDK
    // internal state may be left dirty. That is acceptable precisely here: on
    // this path we disable LeiaSR for the rest of the session and never call
    // into the SDK again. This function deliberately holds no C++ objects with
    // destructors, which __try/__except also requires.
    static HRESULT createSRInterfaceGuarded(ID3D12Device *device, HWND window,
                                            SimulatedReality::SRInterfaceDX12 **out, DWORD *raisedCode) {
        __try {
            return SimulatedReality::CreateSRInterfaceDX12(device, window, out);
        }
        __except (EXCEPTION_EXECUTE_HANDLER) {
            *raisedCode = GetExceptionCode();
            return E_FAIL;
        }
    }

    // Same guard for the per-frame path. Covers the whole SDK sequence, not just
    // Weave: SetInputTexture reads the resource desc and rebinds the weaver's
    // view, which is as capable of faulting as the weave itself. Returns 0 on
    // success, otherwise the raised exception code. Holds no C++ objects with
    // destructors, as __try/__except requires.
    static DWORD weaveGuarded(SimulatedReality::SRInterfaceDX12 *sr,
                              ID3D12Resource *inputTexture, bool setOutputFormat, DXGI_FORMAT outputFormat,
                              ID3D12GraphicsCommandList *commandList,
                              const D3D12_VIEWPORT &viewport, const D3D12_RECT &scissorRect) {
        __try {
            // Re-bound every frame: the pointer can be the same resource while
            // the SDK's internal cached view has been invalidated by external
            // state changes (resize, etc). SR-lib reads the width, height and
            // format off the resource desc.
            sr->SetInputTexture(inputTexture);
            if (setOutputFormat) {
                sr->SetOutputFormat(outputFormat);
            }
            sr->Weave(commandList, viewport, scissorRect);
            return 0;
        }
        __except (EXCEPTION_EXECUTE_HANDLER) {
            return GetExceptionCode();
        }
    }

    // Guarded for the same reason creation and weaving are: these are extern "C"
    // HRESULT entry points into delay-loaded DLLs, and a missing or mismatched
    // DLL raises SEH out of the delay-load helper, which a C++ catch cannot
    // intercept. Holds no C++ objects with destructors, as __try/__except needs.
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
        // OUR request, not the panel's state -- the service arbitrates between
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
        fprintf(stderr, "LeiaSR: creating SR interface (device %p, hwnd %p)...\n",
                static_cast<void *>(device), static_cast<void *>(window));
        fflush(stderr);

        SimulatedReality::SRInterfaceDX12 *sr = nullptr;
        DWORD raisedCode = 0;
        const HRESULT hr = createSRInterfaceGuarded(device, window, &sr, &raisedCode);
        if (FAILED(hr) || sr == nullptr) {
            if (raisedCode != 0) {
                fprintf(stderr, "LeiaSR: CreateSRInterfaceDX12 raised exception 0x%08lX. "
                                "Falling back to non-Leia presentation.\n",
                        static_cast<unsigned long>(raisedCode));
            }
            else {
                fprintf(stderr, "LeiaSR: CreateSRInterfaceDX12 failed (hr 0x%08lX). "
                                "Falling back to non-Leia presentation.\n",
                        static_cast<unsigned long>(hr));
            }
            fflush(stderr);
            return false;
        }

        fprintf(stderr, "LeiaSR: SR interface created.\n");
        fflush(stderr);

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
        // lens -- which on a switchable panel means it stays lensed with nobody
        // weaving. This ordering matters just as much on the mid-session weave
        // failure below, which tears the context down early.
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
        // normal answer on a fixed-lens panel -- there is no switchable layer to
        // hold a preference about -- and is not worth reporting. SR-lib latches
        // the failed create on its side too.
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

        const bool setOutputFormat = (outputFormat != state->lastOutputFormat);

        // The caller is responsible for having already reset the command list's
        // own rasterizer viewport/scissor to the swap-chain extent — D3D12
        // rasterizes against RSSetViewports, not against what the weaver is
        // told here. See the note in SR.hpp.
        //
        // Guarded the same way as creation, and for the same reason: this runs
        // on the present thread every frame, so anything escaping it kills the
        // process. A C++ catch alone would miss SEH from the delay-load helper.
        const DWORD raisedCode = weaveGuarded(state->sr, inputTexture, setOutputFormat, outputFormat,
                                              commandList, viewport, scissorRect);
        if (raisedCode == 0) {
            if (setOutputFormat) {
                state->lastOutputFormat = outputFormat;
            }
        }
        else {
            // Tear down rather than raise again on every subsequent frame. The
            // present queue sees isAvailable() go false and composes plain
            // Side-by-Side to the swap chain from here on.
            fprintf(stderr, "LeiaSR: weave raised exception 0x%08lX. Disabling LeiaSR for this session.\n",
                    static_cast<unsigned long>(raisedCode));
            fflush(stderr);
            shutdown();
        }
    }
} // namespace RT64

#endif // LEIASR_SUPPORTED
