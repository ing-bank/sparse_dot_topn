if(SDTN_MBUILD)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/src")
endif()

install(TARGETS _sparse_dot_topn_core LIBRARY DESTINATION "${PROJECT_NAME}/lib")
