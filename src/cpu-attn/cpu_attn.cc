#include "cpu_attn.h"

namespace kvlib {

void CpuAttention::MyGGML_print_tensor(const struct ggml_tensor* tensor) {
  printf(
      "tensor type: %d, shape: %ld,%ld,%ld,%ld, op: %d, flags: %d, name: "
      "%s\n",
      tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
      tensor->op, tensor->flags, tensor->name);
}

void CpuAttention::MyGGML_print_tensor_data(const struct ggml_tensor* tensor) {
  if (!tensor) {
    printf("Tensor is NULL.\n");
    return;
  }

  printf("Tensor Name: %s\n", tensor->name);
  printf("Dimensions: [");
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    printf("%ld%s", tensor->ne[i], (i < GGML_MAX_DIMS - 1) ? ", " : "");
  }
  printf("]\n");

  if (!tensor->data) {
    printf("Data pointer is NULL.\n");
    return;
  }

  printf("Data:\n");

  size_t total_elements = 1;
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    total_elements *= tensor->ne[i];
  }

  half* data_fp16 = (half*)tensor->data;
  for (size_t i = 0; i < total_elements; ++i) {
    printf("%f ", __half2float(data_fp16[i]));

    if ((i + 1) % tensor->ne[0] == 0) {
      printf("\n");
    }
  }
}

void CpuAttention::MyGGML_print_tensor_data_float(
    const struct ggml_tensor* tensor) {
  if (!tensor) {
    printf("Tensor is NULL.\n");
    return;
  }

  printf("Tensor Name: %s\n", tensor->name);
  printf("Dimensions: [");
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    printf("%ld%s", tensor->ne[i], (i < GGML_MAX_DIMS - 1) ? ", " : "");
  }
  printf("]\n");

  if (!tensor->data) {
    printf("Data pointer is NULL.\n");
    return;
  }

  printf("Data:\n");

  size_t total_elements = 1;
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    total_elements *= tensor->ne[i];
  }

  float* data_fp16 = (float*)tensor->data;
  for (size_t i = 0; i < total_elements; ++i) {
    printf("%f ", (data_fp16[i]));

    if ((i + 1) % tensor->ne[0] == 0) {
      printf("\n");
    }
  }
}

// CpuAttention::CpuAttention(size_t mem_size, int n_dims, const int64_t* ne)
//     : n_dims(n_dims), ne(ne)  // Initialize n_dims and ne
// {
//     struct ggml_init_params params = {
//         /* .mem_size   = */ mem_size,
//         /* .mem_buffer = */ NULL,
//         /* .no_alloc   = */ false,
//     };
//     ctx0 = ggml_init(params);
//     const int64_t query_ne[4] = {ne[0], 1, ne[2], ne[3]};
//     query_buffer = ggml_new_tensor(ctx0, GGML_TYPE_F16, n_dims, query_ne);
//     MyGGML_print_tensor(query_buffer);
// }

CpuAttention::CpuAttention(size_t mem_size, int n_dims,
                           const std::vector<int64_t>& ne)
    : n_dims(n_dims),
      ne(ne)  // Initialize n_dims and ne
{
  struct ggml_init_params params = {
      /* .mem_size   = */ mem_size,
      /* .mem_buffer = */ NULL,
      /* .no_alloc   = */ true,
  };
  ctx0 = ggml_init(params);
  const int64_t query_ne[4] = {ne[0], 1, ne[2], ne[3]};
  query_buffer = ggml_new_tensor(ctx0, GGML_TYPE_F32, n_dims, query_ne);
  // MyGGML_print_tensor(query_buffer);
}

CpuAttention::~CpuAttention() {
  // Add destructor to clean up resources if necessary
  ggml_free(ctx0);
}

void CpuAttention::FillKeyValye(torch::Tensor keys, torch::Tensor values) {
  const int64_t kv_ne[4] = {ne[0], ne[1], ne[2], ne[3]};
  key_buffer = ggml_new_tensor(ctx0, GGML_TYPE_F16, n_dims, kv_ne);
  value_buffer = ggml_new_tensor(ctx0, GGML_TYPE_F16, n_dims, kv_ne);
  // MyGGML_print_tensor(key_buffer);
  // MyGGML_print_tensor(value_buffer);
  key_buffer->data = keys.data_ptr();
  value_buffer->data = values.data_ptr();
  // MyGGML_print_tensor_data(key_buffer);
  // MyGGML_print_tensor_data(value_buffer);
  return;
}

void CpuAttention::AppendKeyValue(torch::Tensor keys, torch::Tensor values) {
  const int64_t kv_ne[4] = {ne[0], 1, ne[2], ne[3]};
  struct ggml_tensor* new_key_buffer =
      ggml_new_tensor(ctx0, GGML_TYPE_F16, n_dims, kv_ne);
  struct ggml_tensor* new_value_buffer =
      ggml_new_tensor(ctx0, GGML_TYPE_F16, n_dims, kv_ne);
  new_key_buffer->data = keys.data_ptr();
  new_value_buffer->data = values.data_ptr();
  // TODO: add append logic here
  // MyGGML_print_tensor_data(key_buffer);
  // MyGGML_print_tensor_data(value_buffer);
  return;
}

//   struct ggml_tensor* ConvertFromTorchTensor(torch::Tensor t) {}

void CpuAttention::Attention(torch::Tensor query, torch::Tensor result) {
  query_buffer->data = query.data_ptr();
  struct ggml_tensor* dst =
      ggml_flash_attn_ext(ctx0, query_buffer, key_buffer, value_buffer, nullptr,
                          1.0f / sqrt(ne[0]), 0, 0);
  dst->data = result.data_ptr<float>();

  ggml_cgraph* gf = ggml_new_graph(ctx0);
  ggml_build_forward_expand(gf, dst);
  std::vector<uint8_t> buf;
  struct ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
  if (plan.work_size > 0) {
    buf.resize(plan.work_size);
    plan.work_data = buf.data();
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  ggml_graph_compute(gf, &plan);
  auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  std::cout << "Computation took: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   elapsed_time)
                   .count()
            << " us" << std::endl;
}

void CpuAttention::SparseAttention(torch::Tensor query, torch::Tensor result,
                                   torch::Tensor index) {
  query_buffer->data = query.data_ptr();
  const int64_t index_ne[4] = {index.sizes()[3], index.sizes()[2],
                               index.sizes()[1], index.sizes()[0]};
  struct ggml_tensor* index_tensor =
      ggml_new_tensor(ctx0, GGML_TYPE_I64, n_dims, index_ne);
  index_tensor->data = index.data_ptr();
  struct ggml_tensor* dst = ggml_sparse_flash_attn_ext(
      ctx0, query_buffer, key_buffer, value_buffer, index_tensor, nullptr,
      1.0f / sqrt(ne[0]), 0, 0);
  dst->data = result.data_ptr<float>();

  ggml_cgraph* gf = ggml_new_graph(ctx0);
  ggml_build_forward_expand(gf, dst);
  std::vector<uint8_t> buf;
  struct ggml_cplan plan = ggml_graph_plan(gf, 32, nullptr);
  if (plan.work_size > 0) {
    buf.resize(plan.work_size);
    plan.work_data = buf.data();
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  ggml_graph_compute(gf, &plan);
  auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  std::cout << "Computation took: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   elapsed_time)
                   .count()
            << " us" << std::endl;
}

void CpuAttention::SparseAttentionWithMeta(torch::Tensor query,
                                           torch::Tensor result,
                                           torch::Tensor index) {
  query_buffer->data = query.data_ptr();
  const int64_t index_ne[4] = {index.sizes()[3], index.sizes()[2],
                               index.sizes()[1], index.sizes()[0]};
  struct ggml_tensor* index_tensor =
      ggml_new_tensor(ctx0, GGML_TYPE_I64, n_dims, index_ne);
  index_tensor->data = index.data_ptr();
  struct ggml_tensor* dst = ggml_sparse_flash_attn_with_meta_ext(
      ctx0, query_buffer, key_buffer, value_buffer, index_tensor, nullptr,
      1.0f / sqrt(ne[0]), 0, 0);
  dst->data = result.data_ptr<float>();

  ggml_cgraph* gf = ggml_new_graph(ctx0);
  ggml_build_forward_expand(gf, dst);
  std::vector<uint8_t> buf;
  struct ggml_cplan plan = ggml_graph_plan(gf, 32, nullptr);
  if (plan.work_size > 0) {
    buf.resize(plan.work_size);
    plan.work_data = buf.data();
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  ggml_graph_compute(gf, &plan);
  auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  std::cout << "Computation took: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   elapsed_time)
                   .count()
            << " us" << std::endl;
}

}  // namespace kvlib