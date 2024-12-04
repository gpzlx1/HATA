#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include "operator.h"

namespace kvlib {

// cuda vector add
__global__ void vector_add(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

torch::Tensor AddCUDA(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  int n = a.size(0);
  int block_size = 1024;
  int grid_size = (n + block_size - 1) / block_size;
  vector_add<<<grid_size, block_size>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
  return c;
}

}  // namespace kvlib