#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>


void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  // We use the stride instead of numel in case the cache is padded for memory
  // alignment reasons, we assume the blocks data (inclusive of any padding)
  // is contiguous in memory
  const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpy(dst_ptr + dst_offset, src_ptr + src_offset,
               block_size_in_bytes, memcpy_type);
  }
}

void swap_blocks_multi_layer2(
    const std::vector<torch::Tensor>& src_kv_caches,
    const std::vector<torch::Tensor>& dst_kv_caches,
    const torch::Tensor& block_mapping) {

  TORCH_CHECK(src_kv_caches.size() == dst_kv_caches.size(),
              "src and dst must have the same number of layers");
  TORCH_CHECK(block_mapping.device().is_cpu(),
              "block_mapping must be on CPU");

  // Assume all tensors are on same device
  torch::Device src_device = src_kv_caches[0].device();
  torch::Device dst_device = dst_kv_caches[0].device();

  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  const int64_t num_blocks = block_mapping.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Loop over layers
  for (size_t layer = 0; layer < src_kv_caches.size(); ++layer) {
    auto src = src_kv_caches[layer];
    auto dst = dst_kv_caches[layer];

    TORCH_CHECK(src.dim() >= 2 && dst.dim() >= 2,
                "Expected each KV cache tensor to have at least 2 dims "
                "(kv, blocks, ...)");
    TORCH_CHECK(src.size(0) == 2 && dst.size(0) == 2,
                "First dimension must be size=2 (key, value)");

    // For both key/value (dim 0 split)
    for (int kv = 0; kv < 2; ++kv) {
      auto src_view = src.select(0, kv);
      auto dst_view = dst.select(0, kv);

      char* src_ptr = static_cast<char*>(src_view.data_ptr());
      char* dst_ptr = static_cast<char*>(dst_view.data_ptr());

      // Size per "block" in bytes.
      // We assume blocks are indexed along dim=0 of src_view.
      int64_t block_size_in_bytes =
        src_view.element_size() * src_view.stride(0);

      for (int64_t i = 0; i < num_blocks; i++) {
        int64_t src_block_number = block_mapping[i][0].item<int64_t>();
        int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
        int64_t src_offset = src_block_number * block_size_in_bytes;
        int64_t dst_offset = dst_block_number * block_size_in_bytes;

        cudaMemcpyAsync(dst_ptr + dst_offset,
                        src_ptr + src_offset,
                        block_size_in_bytes,
                        memcpy_type,
                        stream);
      }
    }
  }

  // Synchronize once after all copies queued
  cudaStreamSynchronize(stream);
}

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// CUDA kernel for copying blocks
__global__ void copy_blocks_kernel(
    const char* src,
    char* dst,
    const int64_t* block_mapping, // shape: [num_blocks, 2]
    int64_t num_blocks,
    int64_t block_size_in_bytes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_bytes = num_blocks * block_size_in_bytes;

  for (int i = idx; i < total_bytes; i += blockDim.x * gridDim.x) {
    int block_idx = i / block_size_in_bytes;
    int byte_in_block = i % block_size_in_bytes;

    int64_t src_block_num = block_mapping[2 * block_idx + 0];
    int64_t dst_block_num = block_mapping[2 * block_idx + 1];

    // Calculate src and dst offsets
    int64_t src_offset = src_block_num * block_size_in_bytes + byte_in_block;
    int64_t dst_offset = dst_block_num * block_size_in_bytes + byte_in_block;

    dst[dst_offset] = src[src_offset];
  }
}

// This function copies data between CUDA and pinned host memory using the custom kernel.
void swap_blocks_multi_layer_kfir(
    const std::vector<torch::Tensor>& src_kv_caches,
    const std::vector<torch::Tensor>& dst_kv_caches,
    const torch::Tensor& block_mapping) {
  TORCH_CHECK(src_kv_caches.size() == dst_kv_caches.size(),
              "src and dst must have the same number of layers");
  TORCH_CHECK(block_mapping.device().is_cpu(),
              "block_mapping must be on CPU");
  TORCH_CHECK(block_mapping.dim() == 2 && block_mapping.size(1) == 2,
              "block_mapping must have shape [num_blocks, 2]");

  // Assume all tensors are on same device type - either CUDA or pinned CPU
  torch::Device src_device = src_kv_caches[0].device();
  torch::Device dst_device = dst_kv_caches[0].device();

  bool src_cuda = src_device.is_cuda();
  bool dst_cuda = dst_device.is_cuda();
  TORCH_CHECK(src_cuda != dst_cuda, "Must copy between CUDA and host (not CUDA->CUDA or host->host)");

  const int64_t num_blocks = block_mapping.size(0);
  const int64_t block_size_in_bytes =
      src_kv_caches[0][0].element_size() * src_kv_caches[0][0].stride(0);

  // Flatten block_mapping to int64_t* on CPU
  const int64_t* cpu_block_mapping = block_mapping.data_ptr<int64_t>();

  int64_t* d_block_mapping;
  cudaMalloc(&d_block_mapping, num_blocks * 2 * sizeof(int64_t));
  cudaMemcpy(d_block_mapping, cpu_block_mapping, num_blocks * 2 * sizeof(int64_t), cudaMemcpyHostToDevice);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (size_t layer = 0; layer < src_kv_caches.size(); ++layer) {
    for (int kv = 0; kv < 2; ++kv) {
      auto src = src_kv_caches[layer].select(0, kv);
      auto dst = dst_kv_caches[layer].select(0, kv);

      // source pointer
      void* src_ptr_raw = src.data_ptr();
      void* dst_ptr_raw = dst.data_ptr();

      char* src_ptr;
      char* dst_ptr;

      if (src_cuda && !dst_cuda) {
        // src: cuda, dst: pinned host
        cudaHostGetDevicePointer((void**)&dst_ptr, dst_ptr_raw, 0);
        src_ptr = static_cast<char*>(src_ptr_raw);
      } else if (!src_cuda && dst_cuda) {
        // src: pinned host, dst: cuda
        cudaHostGetDevicePointer((void**)&src_ptr, src_ptr_raw, 0);
        dst_ptr = static_cast<char*>(dst_ptr_raw);
      } else {
        TORCH_CHECK(false, "Invalid device combination");
      }

      int threads = 256;
      int blocks = (num_blocks * block_size_in_bytes + threads - 1) / threads;

      copy_blocks_kernel<<<blocks, threads, 0, stream>>>(
          src_ptr, dst_ptr, d_block_mapping, num_blocks, block_size_in_bytes);
    }
  }
  cudaFree(d_block_mapping);
}
