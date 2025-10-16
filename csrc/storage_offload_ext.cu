#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <future>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <optional>

namespace py = pybind11;

// -------------------------------------
// Thread-local pinned buffer
// -------------------------------------
static thread_local void* t_pinned_ptr = nullptr;
static thread_local size_t t_pinned_size = 0;

static std::pair<void*, size_t> get_thread_local_pinned(size_t required_bytes) {
    if (!t_pinned_ptr || t_pinned_size < required_bytes) {
        if (t_pinned_ptr) {
            cudaFreeHost(t_pinned_ptr);
            t_pinned_ptr = nullptr;
            t_pinned_size = 0;
        }

        size_t alloc_size = std::max(required_bytes, (size_t)16 * 1024 * 1024); // at least 16 MB
        cudaError_t err = cudaMallocHost(&t_pinned_ptr, alloc_size);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] cudaMallocHost failed: "
                      << cudaGetErrorString(err) << "\n";
            t_pinned_ptr = nullptr;
            t_pinned_size = 0;
        } else {
            t_pinned_size = alloc_size;
            std::cout << "[INFO] Thread " << std::this_thread::get_id()
                      << " allocated pinned buffer "
                      << (alloc_size / (1024 * 1024)) << " MB\n";
        }
    }
    return {t_pinned_ptr, t_pinned_size};
}


// Tracks async job progress and results
struct JobState {
    // Futures for each async task in the job
    std::vector<std::shared_future<bool>> futures;
    // Number of tasks completed so far
    std::atomic<int> completed_tasks{0};
    // Total number of tasks scheduled for this job
    std::atomic<int> total_tasks{0};
    // Flag indicating if all tasks succeeded
    std::atomic<bool> all_success{true};
};
// --------------------------------------
// Global resources - TODO should removed all Global variables
// --------------------------------------
// CUDA streams assigned to each worker thread
static std::vector<at::cuda::CUDAStream> g_streams;

// Thread-local CUDA stream used by the current worker
static thread_local std::optional<c10::cuda::CUDAStream> thread_stream;
// Thread-local stream index for mapping worker threads to streams
static thread_local size_t thread_stream_idx = 0;

// Mutex protecting access to the jobs map
static std::mutex jobs_mutex;
// Global map of job_id → JobState, tracking async job progress
static std::map<int, std::unique_ptr<JobState>> jobs;

// -------------------------------
// I/O thread pool for scheduling
// -------------------------------
class IOThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};

public:
    IOThreadPool(size_t threads, size_t pinned_buffer_mb = 16) {
        // Initialize PyTorch threading globally (main thread only)
        at::init_num_threads();
        at::set_num_threads(1);

        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i, pinned_buffer_mb] {
                // Each I/O thread preallocates its own pinned buffer
                size_t alloc_bytes = pinned_buffer_mb * 1024 * 1024;
                auto [ptr, size] = get_thread_local_pinned(alloc_bytes);

                if (ptr) {
                    std::cout << "[INFO] IO thread " << i
                              << " preallocated pinned buffer "
                              << (size / (1024 * 1024)) << " MB\n";
                } else {
                    std::cerr << "[WARN] IO thread " << i
                              << " failed to allocate pinned buffer\n";
                }

                // Set thread affinity to prevent migration overhead
                thread_stream_idx = i;
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~IOThreadPool() {
        stop = true;
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
        using return_type = typename std::result_of<F()>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::forward<F>(f)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }
};

// Global IO thread pool for scheduling async PUT/GET tasks
static std::unique_ptr<IOThreadPool> g_io_pool;

// -------------------------------
// Initialize resources with pre-allocation
// -------------------------------

// Initialize IO threads, CUDA streams, and pinned memory pool
void init_performance_resources(size_t io_threads = 0, size_t pinned_buffer_size_mb = 256 ,size_t max_pinned_memory_gb = 10) {
    if (!g_io_pool) {
        if (io_threads == 0) {
            io_threads = std::max(4u, std::thread::hardware_concurrency() / 2);
        }
        std::cout << "[INFO] Initializing IOThreadPool with "
                  << io_threads << " threads, "
                  << pinned_buffer_size_mb << " MB pinned buffer per thread, "
                  << max_pinned_memory_gb << " GB max pinned memory\n"
                  << std::thread::hardware_concurrency() << " hardware_concurrency\n";

        g_io_pool = std::make_unique<IOThreadPool>(io_threads, pinned_buffer_size_mb);

        // Create dedicated streams for each thread
        g_streams.clear();
        g_streams.reserve(io_threads);
        for (size_t i = 0; i < io_threads; i++) {
            g_streams.push_back(at::cuda::getStreamFromPool(/*isHighPriority=*/false));
        }

        // Pre-allocate pinned memory pool
        size_t buffer_size = pinned_buffer_size_mb * 1024 * 1024;
        size_t max_buffers = std::min(io_threads, (max_pinned_memory_gb * 1024 * 1024 * 1024) / buffer_size);

        if (max_buffers == 0) {
           throw std::runtime_error("PinnedMemoryPool size is zero. Increase max_pinned_memory_gb or decrease pinned_buffer_size_mb.");
        }
        // Warm up CUDA context and streams
        for (auto& stream : g_streams) {
            cudaStreamSynchronize(stream.stream());
        }
    }
}

// -------------------------------
// Optimized GPU → CPU copy with zero-copy when possible
// -------------------------------
// Copy selected GPU tensor blocks into pinned CPU buffer asynchronously
torch::Tensor copy_gpu_tensors_to_buffer_async(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<int64_t>& block_ids_list,
    const c10::cuda::CUDAStream& stream) {

    // Calculate total size first
    size_t total_elements = 0;
    auto dtype = src_tensors[0].dtype();
    size_t element_size = src_tensors[0].element_size();

    for (int64_t block_id : block_ids_list) {
        for (const auto &tensor : src_tensors) {
            auto block = tensor.index({torch::indexing::Slice(), block_id});
            total_elements += block.numel();
        }
    }


    size_t required_bytes = total_elements * element_size;
    auto [pinned_ptr, pinned_size] = get_thread_local_pinned(required_bytes);

    torch::Tensor result_cpu;

    if (pinned_ptr && pinned_size >= required_bytes) {
        // Use pinned memory - much faster transfer
        std::vector<torch::Tensor> blocks;
        blocks.reserve(src_tensors.size() * block_ids_list.size());

        for (int64_t block_id : block_ids_list) {
            for (const auto &tensor : src_tensors) {
                auto block = tensor.index({torch::indexing::Slice(), block_id});
                blocks.push_back(block.contiguous());
            }
        }

        // Concatenate all selected [K,V] block slices into one big tensor along dim=0 → [2 * num_layers * num_blocks, H, B, D]
        auto flat_gpu = torch::cat(blocks, 0);

        // Create memory view of CPU tensor using pinned memory
        result_cpu = torch::from_blob(
            pinned_ptr,
            {static_cast<long>(total_elements)},
            torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true)
        );

        // Async copy GPU -> pinned CPU memory
        cudaMemcpyAsync(
            pinned_ptr,
            flat_gpu.data_ptr(),
            required_bytes,
            cudaMemcpyDeviceToHost,
            stream.stream()
        );

    } else {
        // Fallback to regular memory

        std::vector<torch::Tensor> blocks;
        blocks.reserve(src_tensors.size() * block_ids_list.size());

        for (int64_t block_id : block_ids_list) {
            for (const auto &tensor : src_tensors) {
                auto block = tensor.index({torch::indexing::Slice(), block_id});
                blocks.push_back(block);
            }
        }

        auto flat = torch::cat(blocks, 0);
        result_cpu = flat.contiguous().to(torch::kCPU, /*non_blocking=*/true);
    }

    if (result_cpu.dtype() == torch::kBFloat16) {
        result_cpu = result_cpu.view(torch::kUInt16);
    }

    return result_cpu;
}

// -------------------------------
// Status and cleanup
// -------------------------------
// Return finished jobs and their success status
std::vector<std::pair<int, bool>> get_finished_ext() {
    std::lock_guard<std::mutex> lock(jobs_mutex);

    std::vector<std::pair<int, bool>> results;
    std::vector<int> to_erase;

    // Iterate over all active jobs.
    for (auto &kv : jobs) {
        int job_id = kv.first;
        auto &job_state = kv.second;

        // Check if the job has completed all its tasks.
        if (job_state->completed_tasks.load() == job_state->total_tasks.load()) {
            bool all_ok = job_state->all_success.load();
            results.emplace_back(job_id, all_ok);
            to_erase.push_back(job_id);
        }
    }

    // Remove all finished jobs from the map.
    for (int jid : to_erase) {
        jobs.erase(jid);
    }
    return results;
}

// Release IO threads, CUDA streams, and pinned pool
void cleanup_performance_resources() {
    g_io_pool.reset();
    g_streams.clear();

    if (t_pinned_ptr) {
        cudaFreeHost(t_pinned_ptr);
        t_pinned_ptr = nullptr;
        t_pinned_size = 0;
    }
}

//----------------------------------------------------------------------
// GPU → Storage (PUT)
// ----------------------------------------------------------------------

// File write helpers - optimized for large writes
bool flush_one_to_disk_fast(const std::string &target_path,
                            const torch::Tensor &host_buf) {
    const void* data_ptr = host_buf.data_ptr();
    size_t nbytes = host_buf.nbytes();
    std::string tmp_path = target_path + ".tmp";

    // Use larger buffer for faster I/O
    const size_t WRITE_BUFFER_SIZE = 1024 * 1024; // 1MB buffer

    std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary);
    if (!ofs) {
        std::cerr << "[ERROR] Failed to open temporary file for writing: " << tmp_path << "\n";
        return false;
    }

    // Set larger buffer
    std::vector<char> buffer(WRITE_BUFFER_SIZE);
    ofs.rdbuf()->pubsetbuf(buffer.data(), WRITE_BUFFER_SIZE);

    ofs.write(reinterpret_cast<const char*>(data_ptr), nbytes);
    ofs.close();
    // std::cout << "[INFO] Written " << nbytes << " bytes to temporary file: " << tmp_path << "\n";
    return (std::rename(tmp_path.c_str(), target_path.c_str()) == 0);
}


// Async GPU → Storage transfer (PUT)
bool transfer_async_put_ext(int job_id,
                        std::vector<std::string> target_files,
                        std::vector<torch::Tensor> src_tensors,
                        std::vector<std::vector<int64_t>> all_block_ids) {
    init_performance_resources();

    std::lock_guard<std::mutex> lock(jobs_mutex);
    auto job_state = std::make_unique<JobState>();
    job_state->total_tasks = target_files.size();

    for (size_t i = 0; i < target_files.size(); i++) {
        std::string target = target_files[i];
        auto src = src_tensors;
        auto bids = all_block_ids[i];

        auto future = g_io_pool->enqueue([=, job_state = job_state.get()]() -> bool {

            // Get dedicated stream for this thread
            if (!thread_stream.has_value()) {
                thread_stream = g_streams[thread_stream_idx % g_streams.size()];
            }

            auto current_stream = at::cuda::getCurrentCUDAStream();
            at::cuda::setCurrentCUDAStream(*thread_stream);

            try {
                // Stage 1: Start async GPU → CPU copy (non-blocking)
                auto host_buf = copy_gpu_tensors_to_buffer_async(src, bids, *thread_stream);

                // Stage 2: Only synchronize the specific stream we need
                cudaStreamSynchronize(thread_stream->stream());

                // Stage 3: Write to disk (this is now the only blocking operation)
                bool ok = flush_one_to_disk_fast(target, host_buf);
                if (!ok) {
                    std::cerr << "[ERROR] PUT failed during file write: " << target << "\n";
                }
                //  else {
                //     std::cout << "[INFO] Put to file completed: " << target << "\n";
                // }

                at::cuda::setCurrentCUDAStream(current_stream);

                job_state->completed_tasks.fetch_add(1);
                if (!ok) job_state->all_success = false;
                return ok;

            } catch (...) {
                at::cuda::setCurrentCUDAStream(current_stream);
                job_state->completed_tasks.fetch_add(1);
                job_state->all_success = false;
                std::cerr << "[ERROR] PUT failed for " << target << "\n";
                return false;
            }
        });

        job_state->futures.push_back(future.share());
    }

    jobs[job_id] = std::move(job_state);
    return true;
}



// ----------------------------------------------------------------------
// Storage -> GPU (GET)
// ----------------------------------------------------------------------
// Swap KV blocks between source and destination tensors (multi-layer)
void swap_blocks_multi_layer(
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


// Read a file into a pinned CPU tensor from the pool
torch::Tensor read_file_to_pinned_tensor(const std::string& path) {

    // Open file
    std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::cerr << "[ERROR] Failed to open file: " << path << "\n";
        throw std::runtime_error("Failed to open file: " + path);
    }

    // Determine file size
    size_t file_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    // Get pinned buffer from pool
    auto [pinned_ptr, pinned_size] = get_thread_local_pinned(file_size);
    if (!pinned_ptr || pinned_size < file_size) {
        std::cerr << "[ERROR] Pinned buffer too small for file: " << path << "\n"
                  << "[INFO] Required size: " << file_size << " bytes, Available size: " << pinned_size << " bytes\n"
                  << "pinned_ptr: " << pinned_ptr << "\n";
        throw std::runtime_error("Pinned buffer too small for file: " + path);
    }

    // Read into pinned memory
    ifs.read(reinterpret_cast<char*>(pinned_ptr), file_size);
    ifs.close();
    //std::cout << "[INFO] 1/2 Read file into pinned tensor: " << path << " (" << file_size << " bytes)\n";
    // Wrap pinned buffer into a Torch tensor (CPU, pinned)
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)  // raw bytes
        .device(torch::kCPU)
        .pinned_memory(true);

    // Wrap without copy, need free pinned_ptr.
    auto tensor = torch::from_blob(
        pinned_ptr,
        {static_cast<long>(file_size)},
        [pinned_ptr](void* /*unused*/) {},
        options
    );
    // std::cout << "[INFO] 2/2 Read file into pinned tensor: " << path << " (" << file_size << " bytes)\n";
    return tensor;
}

// Copy buffer to GPU tensors and swap blocks into place
bool copy_and_swap_gpu_tensors_ext(
    torch::Tensor buf,                           // raw CPU buffer (uint8 tensor from mmap or read)
    const std::vector<int64_t>& block_ids_list,  // block IDs to load
    const std::vector<torch::Tensor>& dst_tensors,
    int gpu_blocks_per_file) {

    // 1. Shape of one KV block [2, H, B, D]
    auto sliced = dst_tensors[0].index({torch::indexing::Slice(), 0});
    auto block_shape = sliced.sizes();  // (2, B, D)

    int64_t elems_per_block = 1;
    for (auto s : block_shape) elems_per_block *= s;

    int num_layers = dst_tensors.size();

    int num_blocks = gpu_blocks_per_file;

    int64_t expected_size = elems_per_block * num_layers * num_blocks;
    // std::cout << "[DEBUG] expected_size (elements): " << expected_size
    //           << " -> bytes: " << expected_size * dst_tensors[0].element_size() << std::endl;

    // 2. Reinterpret raw buffer
    auto dtype = dst_tensors[0].dtype();
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

    // 3. Sanity check
    bool full_block = (block_ids_list.size() % gpu_blocks_per_file == 0);
    auto data_ptr = buf.data_ptr<uint8_t>();
    int64_t num_bytes = buf.numel();
    int64_t num_elems = num_bytes / dst_tensors[0].element_size();

    // Create a memory overview of CPU tensor over existing data_ptr.
    torch::Tensor flat_tensor = torch::from_blob(
        data_ptr,
        {num_elems},
        torch::TensorOptions().dtype(dst_tensors[0].dtype()).device(torch::kCPU)
    );


    int64_t numel = flat_tensor.numel();
    int64_t got_bytes = numel * flat_tensor.element_size();
    int64_t expected_bytes = expected_size * dst_tensors[0].element_size();

    if (numel != expected_size) {
        std::cerr << "[ERROR] File size mismatch: got " << numel
                  << " elems (" << got_bytes << " bytes), expected "
                  << expected_size << " elems (" << expected_bytes << " bytes)\n";
        throw std::runtime_error(
            "File size mismatch: got " + std::to_string(numel) + " elems (" +
            std::to_string(got_bytes) + " bytes), expected " +
            std::to_string(expected_size) + " elems (" +
            std::to_string(expected_bytes) + " bytes)");
    }
    // 4. Reshape into [num_blocks, num_layers, 2, H, B, D]
    // Expect [num_blocks, num_layers, 2, B, D]
    std::vector<int64_t> new_shape = {num_blocks, num_layers};
    new_shape.insert(new_shape.end(), block_shape.begin(), block_shape.end());
    // block_shape = (2, 64, 16, 128) -> [2, 64, 16, 128]
    // new_shape = [1, 80, 2, 64, 16, 128]

    auto torch_arr = flat_tensor.view(new_shape);
    // std::cout << "[DEBUG] torch_arr reshaped to: " << torch_arr.sizes() << std::endl;

    // 5. Handle bf16 special case
    if (dst_tensors[0].dtype() == torch::kBFloat16 && torch_arr.dtype() != torch::kBFloat16) {
        torch_arr = torch_arr.view(torch::kBFloat16);
    }

    // 6. Offset calculation
    int offset = 0;
    int num_read_blocks = static_cast<int>(block_ids_list.size());

    if (!full_block) {
        offset = num_blocks - block_ids_list.size();
    }


    // 7. Build src_tensors for each layer
    // Assert that slice length matches block_ids_list
    TORCH_CHECK((num_blocks - offset) == (int)block_ids_list.size(),
                "Mismatch: slice size != block_ids_list size");

    // Build list of per-layer tensors, each shaped [2, num_blocks, H, B, D]
    std::vector<torch::Tensor> src_tensors;
    src_tensors.reserve(num_layers);

    for (int i = 0; i < num_layers; i++) {
        std::vector<torch::Tensor> blocks_for_layer;
        blocks_for_layer.reserve(block_ids_list.size());

        for (int j = 0; j < (int)block_ids_list.size(); j++) {
            int global_b = block_ids_list[j];   // global block id
            int local_b  = global_b % num_blocks; // offset inside this file

            auto block_tensor = torch_arr[local_b][i];  // one block for this layer [2, H, B, D]
            blocks_for_layer.push_back(block_tensor);
        }
        // Stack all blocks along dim=1 → [2, num_blocks, H, B, D]
        auto layer_tensor = torch::stack(blocks_for_layer, 1).contiguous();
        src_tensors.push_back(layer_tensor);
    }


    // 8. Build mapping tensor (src index → dst block_id)
    std::vector<int64_t> mapping_vec;
    mapping_vec.reserve(num_read_blocks * 2);
    for (int i = 0; i < num_read_blocks; i++) {
        mapping_vec.push_back(i);                   // src index in stacked tensor
        mapping_vec.push_back(block_ids_list[i]);   // target block_id
    }
    auto src_to_dst = torch::from_blob(
        mapping_vec.data(),
        {(int64_t)num_read_blocks, 2},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
    ).clone();

    // 9. Call backend directly
    swap_blocks_multi_layer(src_tensors, dst_tensors, src_to_dst);

    return true;
}

// Async Storage → GPU transfer (GET)
bool transfer_async_get_ext(
    int job_id,
    std::vector<std::string> source_files,
    std::vector<std::vector<int64_t>> all_block_ids,
    std::vector<torch::Tensor> dst_tensors,
    int gpu_blocks_per_file)
{
    init_performance_resources();

    std::lock_guard<std::mutex> lock(jobs_mutex);
    auto job_state = std::make_unique<JobState>();
    job_state->total_tasks = source_files.size();

    for (size_t i = 0; i < source_files.size(); i++) {
        std::string src_file = source_files[i];
        auto block_ids = all_block_ids[i];
        auto future = g_io_pool->enqueue([=, job_state = job_state.get()]() -> bool {

            // Get dedicated stream for this thread
            if (!thread_stream.has_value()) {
                thread_stream = g_streams[thread_stream_idx % g_streams.size()];
            }

            auto current_stream = at::cuda::getCurrentCUDAStream();
            at::cuda::setCurrentCUDAStream(*thread_stream);

            bool stage1_ok = false;
            torch::Tensor host_buf;

            // -------------------------
            // Stage 1: File → pinned CPU
            // -------------------------
            try {
                host_buf = read_file_to_pinned_tensor(src_file);
                stage1_ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Stage1 read_file_to_pinned_tensor failed for "
                        << src_file << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[ERROR] Stage1 unknown failure for " << src_file << std::endl;
            }

            // -------------------------
            // Stage 2: CPU → GPU copy
            // -------------------------
            bool ok = false;
            if (stage1_ok) {
                try {
                    ok = copy_and_swap_gpu_tensors_ext(
                        host_buf, block_ids, dst_tensors, gpu_blocks_per_file);

                    cudaError_t err = cudaStreamSynchronize(thread_stream->stream());
                    if (err != cudaSuccess) {
                        std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                                << cudaGetErrorString(err) << std::endl;
                        ok = false;
                    }

                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] Stage2 copy_and_swap failed for " << src_file
                            << ": " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[ERROR] Stage2 unknown failure for " << src_file << std::endl;
                }
            }

            // -------------------------
            // Final cleanup & accounting
            // -------------------------
            at::cuda::setCurrentCUDAStream(current_stream);
            job_state->completed_tasks.fetch_add(1);
            if (!ok) job_state->all_success = false;
            return ok;
        });

        job_state->futures.push_back(future.share());
    }

    jobs[job_id] = std::move(job_state);
    return true;
}

// -------------------------------
// PYBIND11 module
// -------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_performance_resources", &init_performance_resources,
          py::arg("io_threads") = 0,
          py::arg("pinned_buffer_size_mb") = 256,
          py::arg("max_pinned_memory_gb") = 50);
    m.def("cleanup_performance_resources", &cleanup_performance_resources);
    m.def("get_finished_ext", &get_finished_ext);

    m.def("transfer_async_put_ext", &transfer_async_put_ext,
          "Async transfer  GPU-> Storage",
          py::arg("job_id"),
          py::arg("target_files"),
          py::arg("all_src_tensors"),
          py::arg("all_block_ids"));

    m.def("transfer_async_get_ext", &transfer_async_get_ext,
      "Async GET transfer from Storage → GPU",
      py::arg("job_id"),
      py::arg("source_files"),
      py::arg("all_block_ids"),
      py::arg("dst_tensors"),
      py::arg("gpu_blocks_per_file"));
    m.def("swap_blocks_multi_layer", &swap_blocks_multi_layer);
}