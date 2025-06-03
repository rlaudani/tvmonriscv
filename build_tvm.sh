#!/bin/bash

# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

TVM_PATH=$DIR/tvm
BUILD_DIR=$TVM_PATH/build/debug/build
INSTALL_DIR=$TVM_PATH/build/debug

git clone --recursive https://github.com/apache/tvm.git --depth=1 --branch=v0.17.dev0 $TVM_PATH

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DUSE_LLVM="$DIR/llvm/build/release/bin/llvm-config --link-static" \
    -DUSE_GRAPH_EXECUTOR=ON \
    -DUSE_PROFILER=ON \
    -DUSE_MICRO=ON \
    -DUSE_UMA=ON \
    -DUSE_CPP_RPC=ON \
    -G Ninja \
    ../../..

cmake --build $BUILD_DIR -j `nproc`
cmake --install $BUILD_DIR

cd $DIR
