# -- Python
find_package(
  Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# -- Nanobind
find_package(nanobind CONFIG REQUIRED)


# -- OpenMP
if(NOT SDTN_DISABLE_OPENMP)
  find_package(OpenMP)
  if ((NOT OpenMP_FOUND) AND APPLE)
    include(SetHomebrew)
    set(OpenMP_ROOT ${HOMEBREW_PREFIX}/opt/libomp)
    find_package(OpenMP)
  endif()
  if (NOT OpenMP_FOUND AND SDTN_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
  endif()
endif()
