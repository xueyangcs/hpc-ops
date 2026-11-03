// Copyright (C) 2026 Tencent.

#include "src/topk/topk.h"
#include "src/topk/topk_filtered.cuh"

namespace hpc::topk {

bool topk_filtered_async(int* topk_indices, const float* logits, const int* ke, int topk,
                         const int* num_valid_rows, int m, int n, int row_stride, int out_stride,
                         void* counters, size_t counters_bytes, void* workspace,
                         size_t workspace_bytes, cudaStream_t stream) {
  return filtered::launch_filtered(topk_indices, logits, ke, topk, num_valid_rows, m, n, row_stride,
                                   out_stride, counters, counters_bytes, workspace, workspace_bytes,
                                   stream);
}

size_t topk_filtered_counters_bytes(int num_rows) { return filtered::counters_bytes_for(num_rows); }

size_t topk_filtered_workspace_bytes(int num_rows, int max_kv_len) {
  return filtered::workspace_bytes(num_rows, max_kv_len);
}

size_t topk_filtered_min_workspace_bytes(int max_kv_len) {
  return filtered::one_cta_workspace_bytes(max_kv_len);
}

size_t topk_filtered_peak_workspace_bytes(int max_kv_len) {
  return filtered::peak_workspace_bytes(max_kv_len);
}

}  // namespace hpc::topk
