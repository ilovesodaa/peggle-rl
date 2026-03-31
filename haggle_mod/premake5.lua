-- peggle-rl-bridge Premake5 build script
-- Builds the named-pipe bridge mod DLL for Peggle Deluxe via the Haggle SDK
--
-- Usage:
--   1. Clone Haggle SDK alongside this repo (or set HAGGLE_SDK_PATH)
--   2. Run: premake5 vs2022 --file=premake5.lua
--   3. Open build/peggle-rl-bridge.sln in Visual Studio and build

-- Configuration
local HAGGLE_SDK_PATH = os.getenv("HAGGLE_SDK_PATH") or "../haggle"

workspace "peggle-rl-bridge"
    startproject "peggle-rl-bridge"
    location "build/"
    targetdir "%{wks.location}/bin/%{cfg.buildcfg}-%{cfg.platform}/"
    objdir "%{wks.location}/obj/%{prj.name}/%{cfg.buildcfg}-%{cfg.platform}/"

    largeaddressaware "on"
    editandcontinue "off"
    staticruntime "on"
    systemversion "latest"
    characterset "unicode"
    warnings "extra"

    flags {
        "multiprocessorcompile",
    }

    platforms { "x86" }  -- Peggle Deluxe is 32-bit

    configurations { "Release", "Debug" }

    defines { "_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS" }

    filter "platforms:x86"
        architecture "x86"

    filter "Release"
        defines "NDEBUG"
        optimize "full"
        runtime "release"
        symbols "off"

    filter "Debug"
        defines "DEBUG"
        optimize "debug"
        runtime "debug"
        symbols "on"

    -- Bridge mod DLL
    project "peggle-rl-bridge"
        targetname "peggle-rl-bridge"
        language "c++"
        cppdialect "C++17"
        kind "sharedlib"
        warnings "off"

        pchheader "stdafx.hpp"
        pchsource "src/stdafx.cpp"
        forceincludes "stdafx.hpp"

        dependson { "MinHook", "Haggle" }
        links { "MinHook", "Haggle" }

        includedirs {
            "src/",
            HAGGLE_SDK_PATH .. "/src/haggle/",
            HAGGLE_SDK_PATH .. "/deps/minhook/include/",
        }

        files { "src/**" }

    -- MinHook (static lib dependency)
    group "Dependencies"
    project "MinHook"
        targetname "MinHook"
        language "c++"
        kind "staticlib"
        files { HAGGLE_SDK_PATH .. "/deps/minhook/src/**" }
        includedirs { HAGGLE_SDK_PATH .. "/deps/minhook/include/" }

    -- Haggle SDK (shared lib dependency)
    project "Haggle"
        targetname "haggle-sdk"
        language "c++"
        kind "sharedlib"
        warnings "off"

        pchheader "stdafx.hpp"
        pchsource(HAGGLE_SDK_PATH .. "/src/haggle/stdafx.cpp")
        forceincludes "stdafx.hpp"

        dependson { "MinHook" }
        links { "MinHook" }

        includedirs {
            HAGGLE_SDK_PATH .. "/src/haggle/",
            HAGGLE_SDK_PATH .. "/deps/minhook/include/",
        }

        files { HAGGLE_SDK_PATH .. "/src/haggle/**" }
