cmake -S . -G Ninja -B build \
    -DSKBUILD_PROJECT_NAME="sparse-dot-topn" \
    -DSKBUILD_PROJECT_VERSION="1.0.0" \
    -DSDTN_MBUILD=ON \
    -DSDTN_CPP_STANDARD=17 \
    -DSDTN_ENABLE_DEVMODE=ON \
    -DSDTN_ENABLE_OPENMP=OFF \
    -DSDTN_DISABLE_OPENMP=OFF \
    -DSDTN_ENABLE_ARCH_FLAGS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenMP_ROOT=$(brew --prefix)/opt/libomp \
    -Dnanobind_DIR=$(python3 -c "import nanobind; print(nanobind.cmake_dir())") \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build --target install --config Release --parallel 4
