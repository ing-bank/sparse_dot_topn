# -- Python
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# -- Nanobind
find_package(nanobind CONFIG REQUIRED)

# -- OpenMP
if(SDTN_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
elseif(NOT SDTN_DISABLE_OPENMP)
  find_package(OpenMP)
endif()
