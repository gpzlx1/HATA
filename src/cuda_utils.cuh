
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

#define CUDA_CALL(func, ...)                                            \
  {                                                                     \
    cudaError_t e = (func);                                             \
    if (e != cudaSuccess) {                                             \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e \
                << ") " << __FILE__ << ": line " << __LINE__            \
                << " at function " << STR(func) << std::endl;           \
      return e;                                                         \
    }                                                                   \
  }
