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

CpuAttention::CpuAttention(size_t mem_size, int num_threads)
    : num_threads(num_threads) {
  struct ggml_init_params params = {
      /* .mem_size   = */ mem_size,
      /* .mem_buffer = */ NULL,
      /* .no_alloc   = */ true,
  };
  ggml_ctx = ggml_init(params);
}

CpuAttention::~CpuAttention() {
  // Add destructor to clean up resources if necessary
  ggml_free(ggml_ctx);
}

torch::Tensor CpuAttention::Attention(torch::Tensor query, torch::Tensor key,
                                      torch::Tensor value, float scale) {
  const int32_t bsz = query.size(0);
  const int32_t num_heads = query.size(1);
  const int32_t num_kv_heads = key.size(1);
  const int32_t seqlen_q = query.size(2);
  const int32_t seqlen_kv = key.size(2);
  const int32_t head_dim = query.size(3);

  auto result =
      torch::zeros({bsz, seqlen_q, num_heads, head_dim}, query.options());

  const int64_t query_ne[4] = {head_dim, seqlen_q, num_heads, bsz};
  const int64_t key_ne[4] = {head_dim, seqlen_kv, num_kv_heads, bsz};

  auto query_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, 4, query_ne);
  auto key_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F16, 4, key_ne);
  auto value_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F16, 4, key_ne);

  query_buffer->data = query.data_ptr<float>();
  key_buffer->data = (half*)key.data_ptr<at::Half>();
  value_buffer->data = (half*)value.data_ptr<at::Half>();

  struct ggml_tensor* dst = ggml_flash_attn_ext(
      ggml_ctx, query_buffer, key_buffer, value_buffer, nullptr, scale, 0, 0);
  dst->data = result.data_ptr<float>();

  ggml_cgraph* gf = ggml_new_graph(ggml_ctx);
  ggml_build_forward_expand(gf, dst);
  std::vector<uint8_t> buf;
  struct ggml_cplan plan = ggml_graph_plan(gf, num_threads, nullptr);
  if (plan.work_size > 0) {
    buf.resize(plan.work_size);
    plan.work_data = buf.data();
  }

  ggml_graph_compute(gf, &plan);

  return result;
}

torch::Tensor CpuAttention::SparseAttention(torch::Tensor query,
                                            torch::Tensor key,
                                            torch::Tensor value,
                                            torch::Tensor index, float scale) {
  const int32_t bsz = query.size(0);
  const int32_t num_heads = query.size(1);
  const int32_t num_kv_heads = key.size(1);
  const int32_t seqlen_q = query.size(2);
  const int32_t seqlen_kv = key.size(2);
  const int32_t seqlen_gather = index.size(2);
  const int32_t head_dim = query.size(3);

  auto result =
      torch::zeros({bsz, seqlen_q, num_heads, head_dim}, query.options());

  const int64_t query_ne[4] = {head_dim, seqlen_q, num_heads, bsz};
  const int64_t key_ne[4] = {head_dim, seqlen_kv, num_kv_heads, bsz};
  const int64_t index_ne[4] = {1, seqlen_gather, num_kv_heads, bsz};

  auto index_tensor = ggml_new_tensor(ggml_ctx, GGML_TYPE_I32, 4, index_ne);
  auto query_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, 4, query_ne);
  auto key_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F16, 4, key_ne);
  auto value_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F16, 4, key_ne);

  query_buffer->data = query.data_ptr<float>();
  key_buffer->data = (half*)key.data_ptr<at::Half>();
  value_buffer->data = (half*)value.data_ptr<at::Half>();
  index_tensor->data = index.data_ptr<int32_t>();

  struct ggml_tensor* dst = ggml_sparse_flash_attn_ext(
      ggml_ctx, query_buffer, key_buffer, value_buffer, index_tensor, nullptr,
      scale, 0, 0);
  dst->data = result.data_ptr<float>();

  ggml_cgraph* gf = ggml_new_graph(ggml_ctx);
  ggml_build_forward_expand(gf, dst);
  std::vector<uint8_t> buf;
  struct ggml_cplan plan = ggml_graph_plan(gf, num_threads, nullptr);
  if (plan.work_size > 0) {
    buf.resize(plan.work_size);
    plan.work_data = buf.data();
  }

  ggml_graph_compute(gf, &plan);

  return result;
}

std::vector<torch::Tensor> CpuAttention::SparseAttentionWithMeta(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor index, float scale) {
  const int32_t bsz = query.size(0);
  const int32_t num_heads = query.size(1);
  const int32_t num_kv_heads = key.size(1);
  const int32_t seqlen_q = query.size(2);
  const int32_t seqlen_kv = key.size(2);
  const int32_t seqlen_gather = index.size(2);
  const int32_t head_dim = query.size(3);

  auto result = torch::zeros({bsz * seqlen_q * num_heads * (head_dim + 1)},
                             query.options());

  const int64_t query_ne[4] = {head_dim, seqlen_q, num_heads, bsz};
  const int64_t key_ne[4] = {head_dim, seqlen_kv, num_kv_heads, bsz};
  const int64_t index_ne[4] = {1, seqlen_gather, num_kv_heads, bsz};

  auto index_tensor = ggml_new_tensor(ggml_ctx, GGML_TYPE_I64, 4, index_ne);
  auto query_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, 4, query_ne);
  auto key_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F16, 4, key_ne);
  auto value_buffer = ggml_new_tensor(ggml_ctx, GGML_TYPE_F16, 4, key_ne);

  query_buffer->data = query.data_ptr<float>();
  key_buffer->data = (half*)key.data_ptr<at::Half>();
  value_buffer->data = (half*)value.data_ptr<at::Half>();
  index_tensor->data = index.data_ptr<int64_t>();

  struct ggml_tensor* dst = ggml_sparse_flash_attn_with_meta_ext(
      ggml_ctx, query_buffer, key_buffer, value_buffer, index_tensor, nullptr,
      scale, 0, 0);
  dst->data = result.data_ptr<float>();

  ggml_cgraph* gf = ggml_new_graph(ggml_ctx);
  ggml_build_forward_expand(gf, dst);
  std::vector<uint8_t> buf;
  struct ggml_cplan plan = ggml_graph_plan(gf, num_threads, nullptr);
  if (plan.work_size > 0) {
    buf.resize(plan.work_size);
    plan.work_data = buf.data();
  }

  ggml_graph_compute(gf, &plan);

  torch::Tensor lse =
      result
          .index({torch::indexing::Slice(bsz * num_heads * seqlen_q * head_dim,
                                         torch::indexing::None)})
          .reshape({bsz, seqlen_q, num_heads, 1});
  torch::Tensor attn =
      result
          .index({torch::indexing::Slice(
              torch::indexing::None, bsz * num_heads * seqlen_q * head_dim)})
          .reshape({bsz, seqlen_q, num_heads, head_dim});

  return {attn, lse};
}

}  // namespace kvlib