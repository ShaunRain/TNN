#!/bin/bash

CC=/root/cross-pi-gcc-8.3.0-1/bin/arm-linux-gnueabihf-gcc
CXX=/root/cross-pi-gcc-8.3.0-1/bin/arm-linux-gnueabihf-g++
TNN_LIB_PATH=../../scripts/build_armhf_linux

cd ../../scripts
./build_armhf_linux.sh
cd -

rm -r build
mkdir -p build
cd build
cmake .. \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH

make -j4
