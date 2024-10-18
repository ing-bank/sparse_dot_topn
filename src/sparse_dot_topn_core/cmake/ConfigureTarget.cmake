target_include_directories(_sparse_dot_topn_core PUBLIC ${SDTN_INCLUDE_DIR})
if(OpenMP_CXX_FOUND)
  target_link_libraries(_sparse_dot_topn_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(_sparse_dot_topn_core PRIVATE SDTN_OMP_ENABLED=TRUE)
  if(APPLE)
    # Modified implementation from LightGBM written by JamesLamb
    # see https://github.com/microsoft/LightGBM/pull/6391
    # store path to libomp found at build time in a variable
    get_target_property(
        OpenMP_LIBRARY_LOCATION
        OpenMP::OpenMP_CXX
        INTERFACE_LINK_LIBRARIES
      )
    # get just the filename of that path
    # (to deal with the possibility that it might be 'libomp.dylib' or 'libgomp.dylib' or 'libiomp.dylib')
    get_filename_component(
        OpenMP_LIBRARY_NAME
        ${OpenMP_LIBRARY_LOCATION}
        NAME
      )
    # get directory of that path
    get_filename_component(
        OpenMP_LIBRARY_DIR
        ${OpenMP_LIBRARY_LOCATION}
        DIRECTORY
      )

    # Override the absolute path to OpenMP with a relative one using @rpath.
    #
    # This also ensures that if a libomp.dylib has already been loaded, it'll just use that.
    add_custom_command(
        TARGET _sparse_dot_topn_core
        POST_BUILD
          COMMAND
            install_name_tool
            -change
            ${OpenMP_LIBRARY_LOCATION}
            "@rpath/${OpenMP_LIBRARY_NAME}"
            $<TARGET_FILE:_sparse_dot_topn_core>
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMENT "Replacing hard-coded OpenMP install_name with '@rpath/${OpenMP_LIBRARY_NAME}'..."
      )
    if (SDTN_VENDOR_OPENMP)
      set(OpenMP_target_location ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/.dylibs/${OpenMP_LIBRARY_NAME})

      message(STATUS "Copying ${OpenMP_LIBRARY_NAME} to ${OpenMP_target_location}")
      add_custom_command(
          TARGET _sparse_dot_topn_core
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${OpenMP_LIBRARY_LOCATION}
          ${OpenMP_target_location}
      )
      # add RPATH entries to ensure the loader looks in the following, in the following order:
      #   - @loader_path/../.dylibs/     the dylib we're shipping alongside
      set_target_properties(
          _sparse_dot_topn_core
          PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@loader_path/../.dylibs/"
            INSTALL_RPATH_USE_LINK_PATH FALSE
        )
    else()
      # add RPATH entries to ensure the loader looks in the following, in the following order:
      #   - ${OpenMP_LIBRARY_DIR}        (wherever find_package(OpenMP) found OpenMP at build time)
      #   - /opt/homebrew/opt/libomp/lib (where 'brew install' / 'brew link' puts libomp.dylib)
      set_target_properties(
          _sparse_dot_topn_core
          PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${OpenMP_LIBRARY_DIR};/opt/homebrew/opt/libomp/lib"
            INSTALL_RPATH_USE_LINK_PATH FALSE
        )

    endif()
  endif()
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
