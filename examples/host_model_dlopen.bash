#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
DIR=$DIR/../

iterations=1
eval=false
while getopts "c:i:e" opt; do
  case $opt in
  c) config_file="$OPTARG" ;;
	i) iterations="$OPTARG" ;;
  e) evaluate=true ;;
	\?) echo "Invalid option: -$OPTARG" ;;
  esac
done

rm ${DIR}/models/*.so
rm -rf ${DIR}/build
mkdir -p build/release/build
cd build/release/build

export TVM_ROOT=$DIR/tvm
export CXX=g++
export TVM_NUM_THREADS=1

# TODO: Decide wether to use loop
# TODO: use config file from parameter
# Parse config file
config_file=${DIR}/config.json
MODEL_LIB_PATH=$(jq -r '.lib_path' $config_file)
RUNTIME_STATS_PATH=$(jq -r '.runtime_stats_path' $config_file)
EXEC_NAME=$(jq -r '.exe_name' $config_file)

# Remove old runtime stats file
rm ${DIR}/${RUNTIME_STATS_PATH}

#
cmake -DTVM_ROOT=$TVM_ROOT \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../ \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DMODEL_LIB_PATH=${MODEL_LIB_PATH} \
    -DRUNTIME_STATS_PATH=${RUNTIME_STATS_PATH} \
    -DEXEC_NAME=${EXEC_NAME} \
    ../../../

make install

# Compile model
cd $DIR
export PYTHONPATH=${DIR}/tvm/python
export TVM_LIBRARY_PATH=${DIR}/tvm/build/debug/build
python3 compile/compile_model_x86_profiler.py config.json

export LD_LIBRARY_PATH=${DIR}/models:${DIR}/build/release/lib

# Run model
for ((i=1; i<=iterations; i++))
do
    printf "Iteration %4d/%d\n" "$i" "$iterations"
    ./build/release/bin/${EXEC_NAME}
done

# Evaluate runtime stats
if [ "$evaluate" = true ]; then
    python3 eval/eval.py config.json
fi
