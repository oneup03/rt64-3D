//
// RT64
//

#include "rt64_leiasr_weaver.h"

#ifdef LEIASR_SUPPORTED

#include <cstdio>
#include <exception>
#include <windows.h>

#include "sr/management/srcontext.h"
#include "sr/weaver/dx12weaver.h"
#include "sr/weaver/WeaverTypes.h"

namespace RT64 {
    // LeiaSR DLLs are delay-loaded so the exe can start without the SR Platform
    // installed. Probe them with LoadLibraryW once before any SDK call. If
    // either DLL is missing we silently disable LeiaSR for the session —
    // touching SDK entry points after that point would trigger the delay-load
    // helper's SEH exception and crash the game.
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
        // The SR context manages the connection to the SR Platform service.
        // SDK uses raw-pointer ownership: create via SR::SRContext::create()
        // and clean up with delete. Wrapping it in unique_ptr would call the
        // wrong destructor path (SRContext is not a plain new'd object even
        // though the create() return type suggests it).
        SR::SRContext *context = nullptr;
        SR::IDX12Weaver1 *weaver = nullptr;
        DXGI_FORMAT lastOutputFormat = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT lastInputFormat = DXGI_FORMAT_UNKNOWN;
        int lastInputWidth = 0;
        int lastInputHeight = 0;
    };

    LeiaSRWeaver::LeiaSRWeaver() : state(std::make_unique<State>()) {}

    LeiaSRWeaver::~LeiaSRWeaver() {
        shutdown();
    }

    bool LeiaSRWeaver::initialize(ID3D12Device *device, HWND window) {
        if (state->weaver != nullptr) {
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

        // SR Platform may not be installed. The SDK communicates failures via
        // both return codes and C++ exceptions (especially in srcontext setup),
        // so we wrap both in a try/catch and return false on any error rather
        // than letting it propagate up and crash the game.
        try {
            // SDK factory: returns raw pointer, throws ServerNotAvailableException
            // when the SR Platform service isn't running yet (caught below as
            // std::exception so we silently fall back).
            state->context = SR::SRContext::create();
            if (state->context == nullptr) {
                fprintf(stderr, "LeiaSR: SRContext::create returned nullptr. Falling back to non-Leia presentation.\n");
                return false;
            }

            SR::IDX12Weaver1 *weaver = nullptr;
            const ::WeaverErrorCode err = SR::CreateDX12Weaver(state->context, device, window, &weaver);
            if (err != WeaverSuccess || weaver == nullptr) {
                fprintf(stderr, "LeiaSR: CreateDX12Weaver failed (error %d). Falling back to non-Leia presentation.\n", static_cast<int>(err));
                delete state->context;
                state->context = nullptr;
                return false;
            }
            state->weaver = weaver;

            // The compose intermediate is plain UNORM RGBA (linear values in
            // an UNORM container) and so is the swap chain. Tell the weaver
            // not to do shader-side sRGB conversions either direction so it
            // doesn't double-gamma what's already correct.
            try {
                state->weaver->setShaderSRGBConversion(false, false);
            }
            catch (...) {
                // Non-fatal — fall back to whatever default the SDK picked.
            }

            // Activate the SR context AFTER constructing the weaver: this is
            // what starts the eye-tracking pipeline (face tracker, system
            // sense, etc.). Without it the weaver produces an image but the
            // lenticular interleave is computed against a default (no-track)
            // eye position, which presents as a flat / static pattern that
            // doesn't respond to the viewer's head movement.
            try {
                state->context->initialize();
            }
            catch (const std::exception &e) {
                fprintf(stderr, "LeiaSR: SRContext::initialize threw: %s. Tracker may not engage.\n", e.what());
            }
            catch (...) {
                fprintf(stderr, "LeiaSR: SRContext::initialize threw unknown exception. Tracker may not engage.\n");
            }

            return true;
        }
        catch (const std::exception &e) {
            fprintf(stderr, "LeiaSR: initialization exception: %s. Falling back to non-Leia presentation.\n", e.what());
            if (state->context) {
                delete state->context;
                state->context = nullptr;
            }
            return false;
        }
        catch (...) {
            fprintf(stderr, "LeiaSR: unknown initialization failure. Falling back to non-Leia presentation.\n");
            if (state->context) {
                delete state->context;
                state->context = nullptr;
            }
            return false;
        }
    }

    void LeiaSRWeaver::shutdown() {
        // Weaver must be destroyed before the context that owns it.
        if (state->weaver != nullptr) {
            state->weaver->destroy();
            state->weaver = nullptr;
        }
        if (state->context != nullptr) {
            delete state->context;
            state->context = nullptr;
        }
        state->lastOutputFormat = DXGI_FORMAT_UNKNOWN;
        state->lastInputFormat = DXGI_FORMAT_UNKNOWN;
        state->lastInputWidth = 0;
        state->lastInputHeight = 0;
    }

    bool LeiaSRWeaver::isAvailable() const {
        return state->weaver != nullptr;
    }

    void LeiaSRWeaver::weave(ID3D12Resource *inputTexture, int inputWidth, int inputHeight,
                             DXGI_FORMAT inputFormat, DXGI_FORMAT outputFormat,
                             ID3D12GraphicsCommandList *commandList,
                             const D3D12_VIEWPORT &viewport, const D3D12_RECT &scissorRect) {
        if (state->weaver == nullptr || inputTexture == nullptr || commandList == nullptr) {
            return;
        }

        // Only update format/dimensions when they actually change. The SDK
        // tolerates re-set calls but they're not free internally, and we'd
        // rather not spend cycles re-binding identical state every frame.
        if (inputTexture != nullptr &&
            (inputFormat != state->lastInputFormat ||
             inputWidth != state->lastInputWidth ||
             inputHeight != state->lastInputHeight)) {
            state->weaver->setInputViewTexture(inputTexture, inputWidth, inputHeight, inputFormat);
            state->lastInputFormat = inputFormat;
            state->lastInputWidth = inputWidth;
            state->lastInputHeight = inputHeight;
        }
        else {
            // Re-bind the texture each frame even when dimensions match: the
            // pointer can be the same resource but the SDK's internal cached
            // view may be invalidated by external state changes (resize, etc).
            state->weaver->setInputViewTexture(inputTexture, inputWidth, inputHeight, inputFormat);
        }

        if (outputFormat != state->lastOutputFormat) {
            state->weaver->setOutputFormat(outputFormat);
            state->lastOutputFormat = outputFormat;
        }

        state->weaver->setCommandList(commandList);
        state->weaver->setViewport(viewport);
        state->weaver->setScissorRect(scissorRect);

        try {
            state->weaver->weave();
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
