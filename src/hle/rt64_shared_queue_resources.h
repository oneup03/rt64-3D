//
// RT64
//

#pragma once

#include <mutex>

#include "common/rt64_common.h"
#include "common/rt64_emulator_configuration.h"
#include "common/rt64_enhancement_configuration.h"
#include "render/rt64_render_target_manager.h"

#include "rt64_framebuffer_manager.h"

#if RT_ENABLED
#   include "render/rt64_raytracing_resources.h"
#endif

namespace RT64 {
    struct InterpolatedFrameCounters {
        uint32_t presented = 0;
        uint32_t available = 0;
        uint32_t count = 0;
        bool skipped = false;
    };

    struct SharedQueueResources {
        // Render configuration.
        uint32_t swapChainWidth = 0;
        uint32_t swapChainHeight = 0;
        uint32_t swapChainRate = 0;
        hlslpp::float2 resolutionScale = 1.0f;
        uint32_t targetRate = 0;
        UserConfiguration userConfig;
        EnhancementConfiguration enhancementConfig;
        EmulatorConfiguration emulatorConfig;
        bool swapChainSizeChanged = false;
        bool rtConfigChanged = true;
        bool userConfigChanged = true;
        bool fbConfigChanged = true;
        bool newConfigValidated = false;
        std::mutex configurationMutex;

        // Render resource managers.
        FramebufferManager framebufferManager;
        RenderTargetManager renderTargetManager;
        std::mutex managerMutex;

        // Workload to present state.
        uint32_t viOriginalRate = 0;
        std::vector<uint32_t> colorImageAddressVector;
        std::unordered_set<uint32_t> colorImageAddressSet;
        std::vector<std::unique_ptr<RenderTarget>> interpolatedColorTargets;
        // Right-eye color target for two-pass stereoscopic rendering. Allocated
        // lazily on the first stereo frame and reused across frames. Paired with
        // the natural color target for the non-interpolated path (or for the
        // first frame in the non-MSAA interpolated path, which also uses the
        // natural target on the left).
        std::unique_ptr<RenderTarget> stereoRightColorTarget;
        // Parallel to interpolatedColorTargets — one right-eye target per
        // interpolated left-eye frame so both eyes update at the same rate.
        std::vector<std::unique_ptr<RenderTarget>> stereoRightInterpolatedTargets;
        InterpolatedFrameCounters interpolatedFrames[2];
        uint32_t interpolatedFramesIndex = 0;
        std::mutex interpolatedMutex;
        std::condition_variable interpolatedCondition;
        std::mutex workloadMutex;

        void setSwapChainSize(uint32_t width, uint32_t height) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            if ((swapChainWidth != width) || (swapChainHeight != height)) {
                swapChainWidth = width;
                swapChainHeight = height;
                swapChainSizeChanged = true;
            }
        }

        void setSwapChainRate(uint32_t rate) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            swapChainRate = rate;
        }

        void setUserConfig(const UserConfiguration &newUserConfig, bool discardFBs) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            userConfig = newUserConfig;
            userConfigChanged = true;
            fbConfigChanged = discardFBs;
            newConfigValidated = false;
        }

        void setEmulatorConfig(const EmulatorConfiguration &newEmulatorConfig) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            emulatorConfig = newEmulatorConfig;
        }

        void setEnhancementConfig(const EnhancementConfiguration &newEnhancementConfig) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            enhancementConfig = newEnhancementConfig;
        }

        void updateMultisampling(const RenderMultisampling &multisampling) {
            interpolatedFrames[0].skipped = true;
            interpolatedFrames[1].skipped = true;
            renderTargetManager.destroyAll();
            renderTargetManager.setMultisampling(multisampling);
            // Clear the stereo right-eye targets. They hold GPU resources from
            // before the multisampling change; the workload thread re-allocates
            // them on the next stereo frame.
            stereoRightColorTarget.reset();
            stereoRightInterpolatedTargets.clear();
        }

#   if RT_ENABLED
        RaytracingConfiguration rtConfig;

        void setRtConfig(const RaytracingConfiguration &newRtConfig) {
            std::scoped_lock<std::mutex> configurationLock(configurationMutex);
            rtConfig = newRtConfig;
            rtConfigChanged = true;
        }
#   endif
    };
};