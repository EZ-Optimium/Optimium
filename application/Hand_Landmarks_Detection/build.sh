#! /bin/bash

set -ex

ROOT=$(dirname $(realpath $0))

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DXNNPACK_PLATFORM_JIT=ON \
    -B build -S . -G Ninja

cmake --build build --target rpi-demo
