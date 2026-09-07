//
// RT64
//

#pragma once

#include <json/json.hpp>

using json = nlohmann::json;

namespace RT64 {
    struct UserConfiguration {
        static const int ResolutionMultiplierLimit;

        enum class GraphicsAPI {
            D3D12,
            Vulkan,
            Metal,
            Automatic,
            OptionCount
        };
        
        enum class Resolution {
            Original,
            WindowIntegerScale,
            Manual,
            OptionCount
        };

        enum class DisplayBuffering {
            Double,
            Triple,
            OptionCount
        };

        enum class Antialiasing {
            None,
            MSAA2X,
            MSAA4X,
            MSAA8X,
            OptionCount
        };

        enum class Filtering {
            Nearest,
            Linear,
            AntiAliasedPixelScaling,
            OptionCount
        };

        enum class AspectRatio {
            Original,
            Expand,
            Manual,
            OptionCount
        };

        enum class Upscale2D {
            Original,
            ScaledOnly,
            All,
            OptionCount
        };

        enum class RefreshRate {
            Original,
            Display,
            Manual,
            OptionCount
        };

        enum class InternalColorFormat {
            Standard,
            High,
            Automatic,
            OptionCount
        };

        enum class HardwareResolve {
            Disabled,
            Enabled,
            Automatic,
            OptionCount
        };

        enum class StereoMode {
            Off,
            SideBySide,
            TopAndBottom,
            RowInterlaced,
            ColumnInterlaced,
            Checkerboard,
            Anaglyph,
            LeiaSR,
            OptionCount
        };

        GraphicsAPI graphicsAPI;
        Resolution resolution;
        DisplayBuffering displayBuffering;
        Antialiasing antialiasing;
        double resolutionMultiplier;
        int downsampleMultiplier;
        Filtering filtering;
        AspectRatio aspectRatio;
        double aspectTarget;
        AspectRatio extAspectRatio;
        double extAspectTarget;
        Upscale2D upscale2D;
        bool threePointFiltering;
        RefreshRate refreshRate;
        int refreshRateTarget;
        InternalColorFormat internalColorFormat;
        HardwareResolve hardwareResolve;
        bool idleWorkActive;
        bool developerMode;
        StereoMode stereoMode;
        // Slider values set from the host application; converted to clip-space
        // separation / world-space convergence by the renderer.
        //
        // Separation is 0..100 and maps onto 0..0.10 of screen width. Under the
        // clip-space parameterization it IS the projection shear, so the value
        // is the background disparity the viewer actually sees, expressed in the
        // same terms as the divergence ceiling that bounds it (~0.105 on a
        // 27-inch 16:9 panel, lower on anything bigger).
        //
        // Convergence is in TENTHS of its slider, 1..1000 (= 0.1..100), so the
        // slider can step below 1 without this bridge going floating point — and
        // so the depth-driven loop's continuous solve does not quantise into
        // visible steps.
        uint32_t stereoSeparation;
        uint32_t stereoConvergence;
        // HUD/textbox depth slider. 50 = screen plane (mono). Below 50 the HUD
        // is pushed behind the screen; above 50 it pops out toward the viewer.
        // The resulting shift scales with stereoSeparation, so the HUD goes flat
        // along with the world when the 3D effect is dialled to zero.
        uint32_t stereoHudDepth;
        // Ghost-reduction (anti-crosstalk) range compression applied by the
        // stereo compose shader. Every stereo display leaks part of each eye's
        // image into the other, and how visible that leak is depends on the
        // brightness difference between the eyes, so compressing the range
        // before the image reaches the display reduces what's visible.
        //   stereoGhostContrast:   0..100 percent. 100 = off (no squeeze).
        //   stereoGhostBlackFloor: 0..100 percent. 0 = off (no lift).
        // Both are exact no-ops at their defaults and the shader skips the math
        // entirely when they are both there.
        uint32_t stereoGhostContrast;
        uint32_t stereoGhostBlackFloor;
        // Non-zero while the user's Auto Convergence toggle is on. The
        // depth-driven convergence loop is gated on this.
        uint32_t stereoAutoConvergence;
        // Comfort budget for the depth-driven convergence loop, in thousandths
        // of screen width of permitted pop-out. SIGNED: 0 puts the screen plane
        // exactly on the nearest object, and negative values pull it in FRONT of
        // that object so the whole scene sits behind the glass.
        int32_t stereoComfortTarget;

        UserConfiguration();
        void validate();
        uint32_t msaaSampleCount() const;
        static uint32_t msaaSampleCount(Antialiasing antialiasing);
        static bool isGraphicsAPISupported(GraphicsAPI graphicsAPI);
        static GraphicsAPI resolveGraphicsAPI(GraphicsAPI graphicsAPI);
    };

    extern void to_json(json &j, const UserConfiguration &cfg);
    extern void from_json(const json &j, UserConfiguration &cfg);

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::GraphicsAPI, {
        { UserConfiguration::GraphicsAPI::D3D12, "D3D12" },
        { UserConfiguration::GraphicsAPI::Vulkan, "Vulkan" },
        { UserConfiguration::GraphicsAPI::Metal, "Metal" },
        { UserConfiguration::GraphicsAPI::Automatic, "Automatic" },
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::Resolution, {
        { UserConfiguration::Resolution::Original, "Original" },
        { UserConfiguration::Resolution::WindowIntegerScale, "WindowIntegerScale" },
        { UserConfiguration::Resolution::Manual, "Manual" }
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::AspectRatio, {
        { UserConfiguration::AspectRatio::Original, "Original" },
        { UserConfiguration::AspectRatio::Expand, "Expand" },
        { UserConfiguration::AspectRatio::Manual, "Manual" }
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::Antialiasing, {
        { UserConfiguration::Antialiasing::None, "None" },
        { UserConfiguration::Antialiasing::MSAA2X, "MSAA2X" },
        { UserConfiguration::Antialiasing::MSAA4X, "MSAA4X" },
        { UserConfiguration::Antialiasing::MSAA8X, "MSAA8X" }
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::Filtering, {
        { UserConfiguration::Filtering::Nearest, "Nearest" },
        { UserConfiguration::Filtering::Linear, "Linear" },
        { UserConfiguration::Filtering::AntiAliasedPixelScaling, "AntiAliasedPixelScaling" }
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::Upscale2D, {
        { UserConfiguration::Upscale2D::Original, "Original" },
        { UserConfiguration::Upscale2D::ScaledOnly, "ScaledOnly" },
        { UserConfiguration::Upscale2D::All, "All" }
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::RefreshRate, {
        { UserConfiguration::RefreshRate::Original, "Original" },
        { UserConfiguration::RefreshRate::Display, "Display" },
        { UserConfiguration::RefreshRate::Manual, "Manual" }
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::InternalColorFormat, {
        { UserConfiguration::InternalColorFormat::Standard, "Standard" },
        { UserConfiguration::InternalColorFormat::High, "High" },
        { UserConfiguration::InternalColorFormat::Automatic, "Automatic" }
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::HardwareResolve, {
        { UserConfiguration::HardwareResolve::Disabled, "Disabled" },
        { UserConfiguration::HardwareResolve::Enabled, "Enabled" },
        { UserConfiguration::HardwareResolve::Automatic, "Automatic" }
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(UserConfiguration::StereoMode, {
        { UserConfiguration::StereoMode::Off, "Off" },
        { UserConfiguration::StereoMode::SideBySide, "SideBySide" },
        { UserConfiguration::StereoMode::TopAndBottom, "TopAndBottom" },
        { UserConfiguration::StereoMode::RowInterlaced, "RowInterlaced" },
        { UserConfiguration::StereoMode::ColumnInterlaced, "ColumnInterlaced" },
        { UserConfiguration::StereoMode::Checkerboard, "Checkerboard" },
        { UserConfiguration::StereoMode::Anaglyph, "Anaglyph" },
        { UserConfiguration::StereoMode::LeiaSR, "LeiaSR" }
    });

    struct ConfigurationJSON {
        static bool read(UserConfiguration &cfg, std::istream &stream);
        static bool write(const UserConfiguration &cfg, std::ostream &stream);
    };
};