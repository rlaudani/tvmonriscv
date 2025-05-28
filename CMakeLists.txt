cmake_minimum_required(VERSION 3.10)

project(TVMStaticBuild)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED TVM_ROOT)
    message(FATAL_ERROR "TVM_ROOT must be set. Use -DTVM_ROOT=/path/to/tvm.")
endif()

include_directories(
    ${TVM_ROOT}
    ${TVM_ROOT}/include
    ${TVM_ROOT}/3rdparty/dmlc-core/include
    ${TVM_ROOT}/3rdparty/dlpack/include
)

set(SOURCES_RT
    execute/tvm_runtime_pack.cc
)

# Create packed exeutable
link_directories(${TVM_ROOT}/build)
add_library(tvm_runtime_pack SHARED ${SOURCES_RT})
target_compile_definitions(tvm_runtime_pack PUBLIC -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)
install(TARGETS tvm_runtime_pack DESTINATION lib)

# Compile functions.cc as shared library
set(PREV_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
add_library(profiling_functions SHARED compile/gprov/functions.cc)

# TODO: Add to code
target_compile_definitions(profiling_functions PRIVATE RUNTIME_STATS_PATH="${RUNTIME_STATS_PATH}")

install(TARGETS profiling_functions DESTINATION lib)

# Create packed exeutable
set(CMAKE_CXX_FLAGS "${PREV_CXX_FLAGS}")

# TODO: Change to execute_model.cc
# add_executable("${EXEC_NAME}" execute/execute_mnist.cc)
add_executable(${EXEC_NAME} execute/execute_model.cc)

# TODO: Add all variables
target_compile_definitions(${EXEC_NAME} PRIVATE MODEL_LIB_PATH="${MODEL_LIB_PATH}" N=1 C=3 H=224 W=224)

target_link_libraries(${EXEC_NAME} PRIVATE profiling_functions tvm_runtime_pack dl pthread)
install(TARGETS ${EXEC_NAME} DESTINATION bin)
