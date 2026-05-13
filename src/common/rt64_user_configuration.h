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
        // Slider value 0..100 set from the host application; converted to
        // world-space units by the renderer.
        uint32_t stereoSeparation;
        uint32_t stereoConvergence;
        // HUD/textbox depth slider. 50 = screen plane (mono). Below 50 the HUD
        // is pushed behind the screen; above 50 it pops out toward the viewer.
        uint32_t stereoHudDepth;

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