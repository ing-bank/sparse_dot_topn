set(CMAKE_CXX_STANDARD ${SDTN_CPP_STANDARD})
if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
    set(SDTN_DEVMODE_OPTIONS -Wall -Wextra -Wunused-variable -Wunused-const-variable)
else()
    set(SDTN_DEVMODE_OPTIONS)
endif()

if(SDTN_ENABLE_DEBUG)
    set(SDTN_DEFAULT_BUILD_TYPE Debug)
else()
    set(SDTN_DEFAULT_BUILD_TYPE Release)
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${SDTN_DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${SDTN_DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "Asan" "MinSizeRel" "RelWithDebInfo")
endif ()
