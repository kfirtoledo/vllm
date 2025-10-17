#!/bin/bash

POD_NAME="ubuntu"
REMOTE_BASE="/workspace/vllm"

# List of files to copy (relative to repo root)
FILES=(

  #Shared storage files
  #"vllm/v1/offloading/worker/shared_storage.py"
  #"vllm/v1/offloading/shared_storage.py"
  #"tests/v1/offloading/test_shared_storage.py"
  # "vllm/v1/offloading/shared_storage_manager.py"
  # "vllm/v1/offloading/mediums.py"

  # offloading
  # "vllm/v1/offloading/worker/worker.py"
  # "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py"
  #"vllm/distributed/kv_transfer/kv_connector/factory.py"

  # CPU
  #"vllm/v1/offloading/worker/cpu.py"

  # LMCACHE
  #vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py

  # CUDA
  #"csrc/setup_torch.py"
  "csrc/storage_offload_ext.cu"
  #"csrc/setup.py"
  # "csrc/cache.h"
  # "csrc/cache_kernels.cu"
  # "csrc/torch_bindings.cpp"
  # "vllm/_custom_ops.py"
  # "vllm/attention/backends/abstract.py"
  # "vllm/v1/attention/backends/flash_attn.py"



  # "vllm/v1/executor/multiproc_executor.py"
  #"vllm/v1/offloading/cpu.py"
  # "vllm/v1/offloading/spec.py"
  # "vllm/v1/offloading/factory.py"
  # "vllm/attention/layer.py"
  #"vllm/distributed/kv_transfer/kv_connector/utils.py"
)

for file in "${FILES[@]}"; do
  CMD="kubectl cp ${file} ${POD_NAME}:${REMOTE_BASE}/${file}"
  echo "$CMD"
  eval "$CMD"
done
