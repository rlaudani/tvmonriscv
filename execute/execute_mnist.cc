#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>


void DeployGraphExecutor() {
  LOG(INFO) << "Running graph executor...";
  // Load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("mnist_cnn_input_1x28x28x1_lib.so");
  // Create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  int batch_size = 1;
  tvm::runtime::NDArray* x = new tvm::runtime::NDArray(tvm::runtime::NDArray::Empty({batch_size, 28, 28, 1}, DLDataType{kDLFloat, 32, 1}, dev));
  tvm::runtime::NDArray* y = new tvm::runtime::NDArray(tvm::runtime::NDArray::Empty({batch_size, 10}, DLDataType{kDLFloat, 32, 1}, dev));

  std::cout << "Set input." << std::endl;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < 28; ++j) {
      for (int l = 0; l < 28; ++l) {
        reinterpret_cast<float*>((*x)->data)[i*28*28 + j*28 + l] = 0.5;
      }
    }
  }
  set_input("input_1", (*x));
  
  std::cout << "Run CNN." << std::endl;
  run();
  
  std::cout << "Get output." << std::endl;
  get_output(0, (*y));
  for (int i=0; i<batch_size * 10; ++i)
  {
    std::cout << "y[" << i << "]: " << reinterpret_cast<float*>((*y)->data)[i] << std::endl;
  }

  delete x;
  delete y;
}

// VGG16
// void DeployGraphExecutor() {
//   // Load in the library
//   DLDevice dev{kDLCPU, 0};
//   tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("vgg16_input_1x3x224x224_lib.so");
                                                                         

//   // Create the graph executor module
//   tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
//   tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
//   tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
//   tvm::runtime::PackedFunc run = gmod.GetFunction("run");

//   // Use the C++ API
//   int batch_size = 1;
//   tvm::runtime::NDArray* x = new tvm::runtime::NDArray(tvm::runtime::NDArray::Empty({batch_size, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev));
//   tvm::runtime::NDArray* y = new tvm::runtime::NDArray(tvm::runtime::NDArray::Empty({batch_size, 1000}, DLDataType{kDLFloat, 32, 1}, dev));

//   for (int i = 0; i < batch_size; ++i) {
//     for (int j = 0; j < 224; ++j) {
//       for (int k = 0; k < 224; ++k) {
//         for (int l = 0; l < 3; ++l) {
//           reinterpret_cast<float*>((*x)->data)[i*224*224*3 + j*224*3 + k*3 + l] = 0.5;
//         }
//       }
//     }
//   }
//   set_input("input_1", (*x));
  
//   // std::cout << "Run CNN." << std::endl;
//   run();
  
//   // std::cout << "Get output." << std::endl;
//   get_output(0, (*y));
//   for (int i=0; i<batch_size * 1000; ++i)
//   {
//     // std::cout << "y[" << i << "]: " << reinterpret_cast<float*>((*y)->data)[i] << std::endl;
//   }

//   delete x;
//   delete y;
// }

int main(void) {
  DeployGraphExecutor();
  return 0;
}
