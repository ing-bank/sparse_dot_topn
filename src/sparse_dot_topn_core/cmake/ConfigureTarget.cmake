target_include_directories(_sparse_dot_topn_core PUBLIC ${SDTN_INCLUDE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(_sparse_dot_topn_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(_sparse_dot_topn_core PRIVATE SDTN_OMP_ENABLED=TRUE)
endif()

target_compile_definitions(_sparse_dot_topn_core PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})

# -- Optional
if(SDTN_ENABLE_DEVMODE)
    target_compile_options(_sparse_dot_topn_core PRIVATE ${SDTN_DEVMODE_OPTIONS})
endif()

# -- Options & Properties
set_property(TARGET _sparse_dot_topn_core PROPERTY CXX_STANDARD ${SDTN_CPP_STANDARD})
set_property(TARGET _sparse_dot_topn_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET _sparse_dot_topn_core PROPERTY POSITION_INDEPENDENT_CODE ON)

include(CheckCXXCompilerFlag)

function(check_cxx_support FLAG DEST)
    string(SUBSTRING ${FLAG} 1 -1 STRIPPED_FLAG)
    string(REGEX REPLACE "=" "_" STRIPPED_FLAG ${STRIPPED_FLAG})
    string(TOUPPER ${STRIPPED_FLAG} STRIPPED_FLAG)
    set(RES_VAR "${STRIPPED_FLAG}_SUPPORTED")
    check_cxx_compiler_flag("${FLAG}" ${RES_VAR})
    if(${RES_VAR})
        set(${DEST} "${${DEST}} ${FLAG}" PARENT_SCOPE)
    endif()
endfunction()

# -- Compiler Flags
if (SDTN_ENABLE_ARCH_FLAGS AND "${CMAKE_CXX_FLAGS}" STREQUAL "${CMAKE_CXX_FLAGS_DEFAULT}")
    set(SDTN_ARCHITECTURE_FLAGS "")
    if (APPLE AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64"))
        # see https://github.com/google/highway/issues/745
        check_cxx_support("-march=native" SDTN_ARCHITECTURE_FLAGS)
    else()
        include(FindSse)
        include(FindAvx)
        SDTN_CHECK_FOR_SSE()
        add_definitions(${SSE_DEFINITIONS})
        SDTN_CHECK_FOR_AVX()
        string(APPEND SDTN_ARCHITECTURE_FLAGS "${SSE_FLAGS} ${AVX_FLAGS}")
    endif()

    if (NOT ${SDTN_ARCHITECTURE_FLAGS} STREQUAL "")
        string(STRIP ${SDTN_ARCHITECTURE_FLAGS} SDTN_ARCHITECTURE_FLAGS)
        message(STATUS "sparse-dot-topn | Enabled arch flags: ${SDTN_ARCHITECTURE_FLAGS}")
        if (MSVC)
            separate_arguments(SDTN_ARCHITECTURE_FLAGS WINDOWS_COMMAND "${SDTN_ARCHITECTURE_FLAGS}")
        else()
            separate_arguments(SDTN_ARCHITECTURE_FLAGS UNIX_COMMAND "${SDTN_ARCHITECTURE_FLAGS}")
        endif()
        target_compile_options(_sparse_dot_topn_core PRIVATE $<$<CONFIG:RELEASE>:${SDTN_ARCHITECTURE_FLAGS}>)
    else()
        message(STATUS "sparse-dot-topn | Architecture flags enabled but no valid flags were found")
    endif()
endif()
