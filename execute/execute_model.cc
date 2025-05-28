#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>

#include <chrono>

#ifndef MODEL_LIB_PATH
    #error "MODEL_LIB_PATH is not defined."
#endif

#ifndef N
    #error "N is not defined."
#endif

#ifndef C
    #error "C is not defined."
#endif

#ifndef H
    #error "H is not defined."
#endif

#ifndef W
    #error "W is not defined."
#endif

void DeployGraphExecutor() {
    // Load in the library
    DLDevice dev{kDLCPU, 0};
    // TODO: load
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(MODEL_LIB_PATH);
    // Create the graph executor module
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    // Use the C++ API
    tvm::runtime::NDArray* x = new tvm::runtime::NDArray(tvm::runtime::NDArray::Empty({N, C, H, W}, DLDataType{kDLFloat, 32, 1}, dev));

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    reinterpret_cast<float*>((*x)->data)[n*(C*H*W) + c*(H*W) + h*W + w] = 0.5;
            }
        }
    }
    set_input("input0", (*x));
    

    auto start = std::chrono::high_resolution_clock::now();

    run();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    delete x;
}

int main(void) {
  DeployGraphExecutor();
  return 0;
}
