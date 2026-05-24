//
// RT64
//

#pragma once

#include "common/rt64_profiling_timer.h"
#include "gui/rt64_inspector.h"
#include "render/rt64_leiasr_weaver.h"
#include "render/rt64_stereo_renderer.h"
#include "render/rt64_vi_renderer.h"

#include "rt64_application_window.h"
#include "rt64_present.h"
#include "rt64_shared_queue_resources.h"

#define PRESENT_QUEUE_SIZE 4

namespace RT64 {
    struct WorkloadQueue;

    struct PresentQueue {
        struct External {
            ApplicationWindow *appWindow = nullptr;
            RenderDevice *device = nullptr;
            RenderSwapChain *swapChain = nullptr;
            RenderWorker *presentGraphicsWorker = nullptr;
            WorkloadQueue *workloadQueue = nullptr;
            SharedQueueResources *sharedResources = nullptr;
            const ShaderLibrary *shaderLibrary = nullptr;
            UserConfiguration::GraphicsAPI createdGraphicsAPI = UserConfiguration::GraphicsAPI::OptionCount;
        };

        External ext;
        std::array<Present, PRESENT_QUEUE_SIZE> presents;
        int threadCursor;
        int writeCursor;
        int barrierCursor;
        std::mutex cursorMutex;
        std::condition_variable cursorCondition;
        uint64_t presentId;
        std::mutex presentIdMutex;
        std::condition_variable presentIdCondition;
        std::thread *presentThread = nullptr;
        std::mutex threadMutex;
        std::atomic<bool> presentThreadRunning = false;
        std::recursive_mutex inspectorMutex;
        std::mutex screenFbChangePoolMutex;
        Framebuffer scratchFb;
        FramebufferChangePool scratchFbChangePool;
        FramebufferChangePool screenFbChangePool;
        std::atomic<bool> viewRDRAM = false;
        std::vector<std::unique_ptr<RenderFramebuffer>> swapChainFramebuffers;
        std::unique_ptr<RenderCommandSemaphore> acquiredSemaphore;
        std::vector<std::unique_ptr<RenderCommandSemaphore>> drawSemaphores;
        std::unique_ptr<VIRenderer> viRenderer;
        std::unique_ptr<StereoRenderer> stereoRenderer;
        std::unique_ptr<Inspector> inspector;
        // Off-screen render target for the Configuration GUI (recompui drawHook)
        // when stereo is active. Re-allocated when swap chain size changes.
        // Cleared to transparent before each frame's drawHook call, then sampled
        // by the alpha-overlay stereo pipeline so the UI appears mirrored across
        // each eye half of the SbS/TaB/Interlaced output.
        std::unique_ptr<RenderTexture> stereoUITexture;
        std::unique_ptr<RenderFramebuffer> stereoUIFramebuffer;
        uint32_t stereoUITextureWidth = 0;
        uint32_t stereoUITextureHeight = 0;
        // LeiaSR intermediate: a desktop-resolution SbS-packed texture that
        // the stereo compose pass writes to, and that the lenticular weaver
        // reads from. Re-allocated when desktop resolution changes. Only
        // touched when stereoMode == LeiaSR and the LeiaSR SDK was found at
        // build time and successfully initialized at runtime.
#   ifdef LEIASR_SUPPORTED
        std::unique_ptr<RenderTexture> leiaSRComposeTexture;
        std::unique_ptr<RenderFramebuffer> leiaSRComposeFramebuffer;
        uint32_t leiaSRComposeWidth = 0;
        uint32_t leiaSRComposeHeight = 0;
        LeiaSRWeaver leiaSRWeaver;
        bool leiaSRInitAttempted = false; // Stop retrying initialize() after the SDK refuses once.
#   endif
        ProfilingTimer presentProfiler = ProfilingTimer(120);
        Timestamp presentTimestamp;
        VIHistory viHistory;
        bool presentWaitEnabled = false;

        PresentQueue();
        ~PresentQueue();
        void reset();
        void advanceToNextPresent();
        void repeatLastPresent();
        uint32_t previousWriteCursor() const;
        void waitForIdle();
        void waitForPresentId(uint64_t waitId);
        void setup(const External &ext);
        void threadPresent(const Present &present, bool &swapChainValid);
        void skipInterpolation();
        void notifyPresentId(const Present &present);
        void threadAdvanceBarrier();
        void threadLoop();
    };
};