cmake_minimum_required(VERSION 3.15...3.26)

string(REPLACE "-" "_" SKBUILD_PROJECT_NAME ${SKBUILD_PROJECT_NAME})
string(REGEX REPLACE "\\.?[a-z]+[0-9]+" "" SKBUILD_PROJECT_VERSION ${SKBUILD_PROJECT_VERSION})

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/src/sparse_dot_topn_core/cmake")

include(GNUInstallDirs)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

if(SDTN_MBUILD)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/src")
endif()

set(SDTN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/sparse_dot_topn_core/include")
set(SDTN_SRC_PREF "${PROJECT_SOURCE_DIR}/src/sparse_dot_topn_core/src/")
set(SDTN_SRC_FILES
    ${SDTN_SRC_PREF}/extension.cpp
    ${SDTN_SRC_PREF}/sp_matmul_bindings.cpp
    ${SDTN_SRC_PREF}/sp_matmul_topn_bindings.cpp
    ${SDTN_SRC_PREF}/zip_sp_matmul_topn_bindings.cpp
)

include(FindDependencies)
include(ConfigureBuildType)
# -- target
nanobind_add_module(_sparse_dot_topn_core STABLE_ABI NB_STATIC LTO NOMINSIZE ${SDTN_SRC_FILES})

include(ConfigureTarget)
install(TARGETS _sparse_dot_topn_core LIBRARY DESTINATION "${PROJECT_NAME}/lib")
include(CleanUp)
