#!/bin/bash

# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
DIR=$DIR/../

PROFILING=false
for arg in "$@"
do
    if [ "$arg" == "--profiling" ]; then
        PROFILER_FLAG="-DPROFILER=ON"
        PROFILING=true
    fi
done

rm -rf ${DIR}/build
rm ${DIR}/models/*.so
mkdir -p build/release/build
cd build/release/build

export TVM_ROOT=$DIR/tvm
export CXX=g++
export TVM_NUM_THREADS=1

cmake -DTVM_ROOT=$TVM_ROOT \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../ \
    -DCMAKE_CXX_COMPILER=${CXX} \
    ${PROFILER_FLAG} \
    ../../../

make install

# Compile neural network
cd $DIR
export PYTHONPATH=${DIR}/tvm/python
export TVM_LIBRARY_PATH=${DIR}/tvm/build/debug/build

if $PROFILING; then
    python3 compile/compile_mnist_host_profiler.py
else
    python3 compile/compile_mnist_host.py
fi

# Native execution
export LD_LIBRARY_PATH=${DIR}/models:${DIR}/build/release/lib

for ((i=0; i<=10; i++))
do
    ./build/release/bin/nn_packed
done
