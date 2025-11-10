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
#include <sys/syscall.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <filesystem>
#include <numa.h>

#include "thread_pool.cpp"
#include "debug_utils.cpp"

namespace fs = std::filesystem;
namespace py = pybind11;

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
std::vector<at::cuda::CUDAStream> g_streams;

// Thread-local CUDA stream used by the current worker
thread_local std::optional<c10::cuda::CUDAStream> thread_stream;
// Thread-local stream index for mapping worker threads to streams
thread_local size_t thread_stream_idx = 0;

// Mutex protecting access to the jobs map
static std::mutex jobs_mutex;
// Global map of job_id → JobState, tracking async job progress
std::map<int, std::unique_ptr<JobState>> jobs;


thread_local void* t_pinned_ptr = nullptr;
thread_local size_t t_pinned_size = 0;

// Global IO thread pool for scheduling async PUT/GET tasks
static std::unique_ptr<ThreadPool> g_io_pool;

// -------------------------------
// Initialize resources with pre-allocation
// -------------------------------

// Initialize IO threads, CUDA streams, and pinned memory pool
void init_performance_resources(int io_threads = 0, size_t pinned_buffer_size_mb = 256, size_t max_pinned_memory_gb = 10, int tp_rank = 1) {
    if (!g_io_pool) {
        if (io_threads == 0) {
            io_threads = std::max(4u, std::thread::hardware_concurrency() / 2);
        }

        // Get current device (should be set by vLLM before calling this)
        int device_id;
        cudaGetDevice(&device_id);

        std::cout << "[INFO] Initializing ThreadPool with "
                  << io_threads << " threads on device " << device_id
                  << ", " << pinned_buffer_size_mb << " MB pinned buffer per thread, "
                  << max_pinned_memory_gb << " GB max pinned memory\n";

        // Enable GPU access to mapped host memory (needed only for cudaHostAllocMapped before any CUDA context)
        cudaSetDeviceFlags(cudaDeviceMapHost);
        int gpu_numa = get_gpu_numa_node(device_id);
        numa_set_preferred(gpu_numa);
        // Pre-allocate pinned buffers before launching threads
        preallocate_pinned_buffers(io_threads, pinned_buffer_size_mb);

        // Pass device_id to thread pool
        g_io_pool = std::make_unique<ThreadPool>(io_threads, pinned_buffer_size_mb,
                                                    tp_rank, device_id);

        // Create dedicated streams for each thread on the current device
        g_streams.clear();
        g_streams.reserve(io_threads);
        for (size_t i = 0; i < io_threads; i++) {
            g_streams.push_back(at::cuda::getStreamFromPool(/*isHighPriority=*/false, device_id));
        }

        // Warm up CUDA context and streams
        for (auto& stream : g_streams) {
            cudaStreamSynchronize(stream.stream());
        }
    }
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

// Release IO threads, CUDA streams, and pinned buffer
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
__global__ void copy_blocks_kernel(
    const uint8_t* __restrict__ src_base,           // Source (CPU for GET, GPU for PUT)
    uint8_t* __restrict__ dst_base,                 // Destination (GPU for GET, CPU for PUT)
    const int64_t* __restrict__ block_ids,          // Global block IDs to copy
    const int64_t* __restrict__ src_block_offsets,  // Per-block source offsets (within file or buffer)
    const int num_blocks,                           // Number of blocks to copy
    const int layer,                                // Layer index
    const int64_t num_blocks_tot,                   // Total blocks per tensor (used for offset math)
    const size_t bytes_per_plane,                   // Bytes per K or V plane
    const size_t bytes_per_block,                   // Total bytes per block (K+V)
    const bool is_put) {                            // Add direction flag

    const int bi = blockIdx.x;          // block index
    const int k_or_v = blockIdx.y;      // 0=K, 1=V
    const int tid = threadIdx.x;

    if (bi >= num_blocks) return;

    const int64_t gblock = block_ids[bi];
    const size_t src_block_base = src_block_offsets[bi];

    size_t src_offset, dst_offset;

    if (is_put) {
        // PUT: GPU→CPU Source is GPU, destination is CPU
        src_offset = static_cast<size_t>(gblock + k_or_v * num_blocks_tot) * bytes_per_plane;
        dst_offset = src_block_base +
                     static_cast<size_t>(layer) * bytes_per_block +
                     (k_or_v == 1 ? bytes_per_plane : 0);
    } else {
        // GET: CPU→GPU - Source is CPU, destination is GPU
        src_offset = src_block_base +
                     static_cast<size_t>(layer) * bytes_per_block +
                     (k_or_v == 1 ? bytes_per_plane : 0);
        dst_offset = static_cast<size_t>(gblock + k_or_v * num_blocks_tot) * bytes_per_plane;
    }

    const uint8_t* src = src_base + src_offset;
    uint8_t* dst = dst_base + dst_offset;

    for (size_t i = tid; i < bytes_per_plane; i += blockDim.x) {
        dst[i] = src[i];  // Copy cooperatively across threads
    }
}


// Copy selected GPU tensor blocks into pinned CPU buffer asynchronously
torch::Tensor copy_gpu_tensors_to_buffer(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<int64_t>& block_ids_list,
    const c10::cuda::CUDAStream& stream) {

    TORCH_CHECK(!src_tensors.empty(), "Source tensors list is empty");
    const auto& ref = src_tensors[0];
    TORCH_CHECK(ref.is_contiguous(), "src_tensors must be contiguous");

    const auto shape = ref.sizes();  // [2, num_blocks_total, H, B, D]
    TORCH_CHECK(shape.size() == 5, "Expected shape [2, num_blocks, H, B, D]");

    const int64_t num_blocks_tot = shape[1];
    const int64_t H = shape[2];  //H: number of attention heads
    const int64_t B = shape[3]; // B: number of tokens per block (block size)
    const int64_t D = shape[4]; // D: head dimension
    const int num_layers = static_cast<int>(src_tensors.size());

    const size_t elem_size = ref.element_size();
    const size_t bytes_per_plane = static_cast<size_t>(H * B * D) * elem_size;
    const size_t bytes_per_block = 2 * bytes_per_plane;  // [K,V]

    // Calculate total size needed
    const size_t total_bytes = block_ids_list.size() * num_layers * bytes_per_block;

    auto [pinned_ptr, pinned_size] = get_thread_local_pinned(total_bytes);
    TORCH_CHECK(pinned_size >= total_bytes,
        "Pinned buffer too small: need ", total_bytes, " got ", pinned_size);

    auto dtype = ref.dtype();
    const int64_t total_elements = static_cast<int64_t>(total_bytes / elem_size);

    // Create output tensor view
    torch::Tensor result_cpu = torch::from_blob(
        pinned_ptr, {total_elements},
        torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true)
    );

    auto* cpu_base = static_cast<uint8_t*>(pinned_ptr);

    const char* env = std::getenv("USE_KERNEL_COPY_WRITE");
    bool use_kernel_copy_write = (env && std::string(env) == "1");
    if (use_kernel_copy_write) {
        // Prepare block IDs and CPU offsets
        std::vector<int64_t> cpu_offsets(block_ids_list.size());
        for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
            cpu_offsets[bi] = bi * num_layers * bytes_per_block;
        }

        // Create tensor view of block IDs and transfer to GPU memory
        torch::Tensor block_ids_tensor = torch::from_blob(
            const_cast<int64_t*>(block_ids_list.data()),
            {static_cast<int64_t>(block_ids_list.size())},
            torch::dtype(torch::kInt64)
        ).to(torch::kCUDA, /*non_blocking=*/true);

        // Create tensor view of CPU offsets and transfer to GPU memory
        torch::Tensor cpu_offsets_tensor = torch::from_blob(
            cpu_offsets.data(),
            {static_cast<int64_t>(cpu_offsets.size())},
            torch::dtype(torch::kInt64)
        ).to(torch::kCUDA, /*non_blocking=*/true);

        // Map pinned CPU memory to device pointer (required for GPU kernel to write to host memory - zero-copy)
        uint8_t* cpu_base_dev = nullptr;
        cudaError_t map_err = cudaHostGetDevicePointer(&cpu_base_dev, cpu_base, 0);
        TORCH_CHECK(map_err == cudaSuccess,
            "cudaHostGetDevicePointer failed: ", cudaGetErrorString(map_err));

        // Launch one kernel per layer
        dim3 grid(block_ids_list.size(), 2);  // (blocks, K/V)
        dim3 block(512); // TODO check optimal thread count (256 or 512)

        // Launch copy kernel for each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src_tensors[layer].data_ptr());
            copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
                src_ptr,                                    // Source: GPU tensor
                cpu_base_dev,                               // Destination: CPU pinned memory
                block_ids_tensor.data_ptr<int64_t>(),
                cpu_offsets_tensor.data_ptr<int64_t>(),
                block_ids_list.size(),
                layer,
                num_blocks_tot,
                bytes_per_plane,
                bytes_per_block,
                true  // is_put = true
            );
        }

        // Check for kernel launch errors
        cudaError_t launch_err = cudaGetLastError();
        TORCH_CHECK(launch_err == cudaSuccess,
            "Kernel launch failed: ", cudaGetErrorString(launch_err));

    } else {  // Default behavior - memcpyAsync path
        // Direct pointer arithmetic - NO INDEXING operations
        for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
            const int64_t gblock = block_ids_list[bi];
            const size_t cpu_block_base = bi * num_layers * bytes_per_block;

            for (int layer = 0; layer < num_layers; ++layer) {
                const auto& layer_tensor = src_tensors[layer];
                auto* src_base = reinterpret_cast<const uint8_t*>(layer_tensor.data_ptr());

                // Compute GPU source offsets for K and V
                const size_t gpu_K_off = static_cast<size_t>(gblock) * bytes_per_plane;
                const size_t gpu_V_off = (static_cast<size_t>(num_blocks_tot) + gblock) * bytes_per_plane;

                // Compute CPU destination offsets for K and V
                const size_t cpu_K_off = cpu_block_base + static_cast<size_t>(layer) * bytes_per_block;
                const size_t cpu_V_off = cpu_K_off + bytes_per_plane;

                const void* src_K = src_base + gpu_K_off;
                const void* src_V = src_base + gpu_V_off;
                void* dst_K = cpu_base + cpu_K_off;
                void* dst_V = cpu_base + cpu_V_off;

                cudaError_t err1 = cudaMemcpyAsync(
                    dst_K, src_K, bytes_per_plane,
                    cudaMemcpyDeviceToHost, stream.stream());

                cudaError_t err2 = cudaMemcpyAsync(
                    dst_V, src_V, bytes_per_plane,
                    cudaMemcpyDeviceToHost, stream.stream());

                if (err1 != cudaSuccess || err2 != cudaSuccess) {
                    std::cerr << "[ERROR] cudaMemcpyAsync failed for block=" << gblock
                              << " layer=" << layer
                              << " err1=" << cudaGetErrorString(err1)
                              << " err2=" << cudaGetErrorString(err2)
                              << std::endl;
                    TORCH_CHECK(false, "cudaMemcpyAsync failed");
                }
            }
        }
    }

    // Reinterpret bfloat16 tensor as uint16_t for safe raw byte access (I/O or memcpy)
    if (result_cpu.dtype() == torch::kBFloat16) {
        result_cpu = result_cpu.view(torch::kUInt16);
    }

    return result_cpu;
}

// File write helpers - optimized for large writes
bool write_file_to_disk(const std::string &target_path,
                        const torch::Tensor &host_buf) {
    // Write to temporary file first
    const void* data_ptr = host_buf.data_ptr();
    // Get total number of bytes to write
    size_t nbytes = host_buf.nbytes();

    // Extract and create parent directory
    fs::path file_path(target_path);
    fs::path parent_dir = file_path.parent_path();
    try {
        fs::create_directories(parent_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Failed to create directories: "
                  << e.what() << "\n";
        return false;
    }

    // Write first to a temporary file to ensure atomic rename later
    std::string tmp_path = target_path + ".tmp";

    // Define a larger buffer (1MB) to reduce syscall overhead and speed up I/O
    const size_t WRITE_BUFFER_SIZE = 1 * 1024 * 1024; // 1MB buffer

    std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary);
    if (!ofs) {
        std::cerr << "[ERROR] Failed to open temporary file for writing: " << tmp_path << " - " << std::strerror(errno) << "\n";
        return false;
    }

    // Allocate custom I/O buffer for this stream (replaces small default buffer)
    std::vector<char> buffer(WRITE_BUFFER_SIZE);
    // Apply the custom buffer to the file stream
    ofs.rdbuf()->pubsetbuf(buffer.data(), WRITE_BUFFER_SIZE);

    ofs.write(reinterpret_cast<const char*>(data_ptr), nbytes);
    if (!ofs) {
        std::cerr << "[ERROR] Failed to write to temporary file: " << tmp_path << " - " << std::strerror(errno) << "\n";
        return false;
    }
    ofs.close();
    // Atomically rename temp file to final target name after successful write
    if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
        std::cerr << "[ERROR] " << "Failed to rename " + tmp_path + " to " + target_path + " - " + std::strerror(errno) << "\n";
        return false;
    }

    return true;
}

// Async GPU → Storage transfer (PUT)
bool transfer_async_put_ext(int job_id,
                        std::vector<std::string> target_files,
                        std::vector<torch::Tensor> src_tensors,
                        std::vector<std::vector<int64_t>> all_block_ids) {

    // Create job state object that will track progress and futures for this job.
    auto job_state = std::make_unique<JobState>();
    job_state->total_tasks = target_files.size();

    // Store shared_ptr to tensors to avoid repeated refcount changes
    auto shared_src_tensors= std::make_shared<std::vector<torch::Tensor>>(std::move(src_tensors));

    // For each target file, enqueue one async task in the I/O thread pool.
    for (size_t i = 0; i < target_files.size(); i++) {
        std::string target = target_files[i];
        auto bids = all_block_ids[i];

        auto future = g_io_pool->enqueue([target, bids, shared_src_tensors, job_state = job_state.get()]() -> bool {

            // Check if target file already exists - skip write if it does
            if (std::ifstream(target).good()) {
                job_state->completed_tasks.fetch_add(1);
                return true;  // File exists
            }
            // Ensure correct device is set (thread-local)
            int device_id;
            cudaGetDevice(&device_id);

            // Each thread gets a dedicated CUDA stream for async GPU ops.
            if (!thread_stream.has_value()) {
                //thread_stream = g_streams[thread_stream_idx % g_streams.size()];
                thread_stream = at::cuda::getStreamFromPool(/* isHighPriority = */ false); // use best-effort stream for writes
            }

            // Save current CUDA stream so we can restore it later.
            auto current_stream = at::cuda::getCurrentCUDAStream();
            at::cuda::setCurrentCUDAStream(*thread_stream);

            try {
                // Use reference to avoid copy - dereference shared_ptr
                const auto& src = *shared_src_tensors;

                // Stage 1: Asynchronously copy tensors from GPU to pinned CPU buffer.
                auto host_buf = TIME_EXPR("write phase 1: copy_gpu_tensors_to_buffer", copy_gpu_tensors_to_buffer(src, bids, *thread_stream),"file: " + target);

                cudaError_t err = cudaStreamSynchronize(thread_stream->stream());
                if (err != cudaSuccess) {
                    std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                            << cudaGetErrorString(err) << std::endl;
                }
                // Stage 2: Write the pinned buffer to disk (blocking operation).
                bool ok = TIME_EXPR("write phase 2: write_file_to_disk", write_file_to_disk(target, host_buf),
                  ("file:" + target + " size:" + std::to_string(host_buf.nbytes()))
                );

                if (!ok)
                    std::cerr << "[ERROR] PUT failed during file write: " << target << "\n";

                // Restore original CUDA stream for safety.
                at::cuda::setCurrentCUDAStream(current_stream);

                // Atomically mark task completion.
                job_state->completed_tasks.fetch_add(1);

                // if (!ok) job_state->all_success = false; // TODO- silent ignore write failures for now offloading connector not able to handle failures
                return ok;

            } catch (...) {
                at::cuda::setCurrentCUDAStream(current_stream);
                job_state->completed_tasks.fetch_add(1);
                job_state->all_success = false;
                std::cerr << "[ERROR] PUT failed for " << target << "\n";
                return false;
            }
        });
        // Convert std::future → std::shared_future- is copyable and can be waited on by multiple threads.
        job_state->futures.push_back(future.share());
    }

    std::lock_guard<std::mutex> lock(jobs_mutex); // protect jobs map
    jobs[job_id] = std::move(job_state);

    return true;
}
// ----------------------------------------------------------------------
// Storage -> GPU (GET)
// ----------------------------------------------------------------------

// Read a file into a pinned CPU tensor from the pool
torch::Tensor read_file_from_disk(const std::string& path) {

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
    return tensor;
}


bool copy_buffer_to_gpu_tensors(
    torch::Tensor cpu_buf,
    const std::vector<int64_t>& block_ids_list,
    const std::vector<torch::Tensor>& dst_tensors,
    int num_blocks_in_file,
    const c10::cuda::CUDAStream& stream) {

    TORCH_CHECK(!dst_tensors.empty(), "Destination tensors list is empty");
    const auto& ref = dst_tensors[0];
    TORCH_CHECK(ref.is_contiguous(), "dst_tensors must be contiguous");
    TORCH_CHECK(cpu_buf.is_contiguous(), "cpu buffer must be contiguous");

    // CRITICAL: Verify cpu_buf is pinned memory
    TORCH_CHECK(cpu_buf.is_pinned(),
        "cpu_buf must be pinned memory for kernel-based copy");

    const auto shape = ref.sizes();  // [2, num_blocks_total, H, B, D]
    TORCH_CHECK(shape.size() == 5, "Expected shape [2, num_blocks, H, B, D]");

    const int64_t num_blocks_tot = shape[1];
    const int64_t H = shape[2];  // H: number of attention heads
    const int64_t B = shape[3];  // B: number of tokens per block (block size)
    const int64_t D = shape[4];  // D: head dimension
    const int num_layers = static_cast<int>(dst_tensors.size());

    const size_t elem_size = ref.element_size(); // Size (in bytes) of a single element

    const size_t bytes_per_plane = static_cast<size_t>(H * B * D) * elem_size;
    const size_t bytes_per_block = 2 * bytes_per_plane;  // [K,V] in CPU buf

    auto* cpu_base = cpu_buf.data_ptr<uint8_t>();

    const char* env = std::getenv("USE_KERNEL_COPY_READ");
    bool use_kernel_copy_read = (!env || std::string(env) != "0");

    if (use_kernel_copy_read) {  // Default behavior - batched kernel
        // Calculate CPU buffer offset for each block (maps global block ID to local file offset)
        std::vector<int64_t> cpu_offsets(block_ids_list.size());
        for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
            const int64_t gblock = block_ids_list[bi];
            const int64_t lblock = (num_blocks_in_file > 0) ? (gblock % num_blocks_in_file) : 0;
            cpu_offsets[bi] = static_cast<int64_t>(lblock) * num_layers * bytes_per_block;
        }

        // Wrap block IDs in tensor and copy to GPU for kernel access
        torch::Tensor block_ids_tensor = torch::from_blob(
            const_cast<int64_t*>(block_ids_list.data()),
            {static_cast<int64_t>(block_ids_list.size())},
            torch::dtype(torch::kInt64)
        ).to(torch::kCUDA, /*non_blocking=*/true);

        // Wrap CPU offsets in tensor and copy to GPU for kernel access
        torch::Tensor cpu_offsets_tensor = torch::from_blob(
            cpu_offsets.data(),
            {static_cast<int64_t>(cpu_offsets.size())},
            torch::dtype(torch::kInt64)
        ).to(torch::kCUDA, /*non_blocking=*/true);

        // Launch one kernel per layer
        dim3 grid(block_ids_list.size(), 2);  // (blocks, K/V)
        dim3 block(512); //TODO chechk optimal thread count (256 or 512)

        for (int layer = 0; layer < num_layers; ++layer) {
            uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_tensors[layer].data_ptr());

            copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
                cpu_base,
                dst_ptr,
                block_ids_tensor.data_ptr<int64_t>(),
                cpu_offsets_tensor.data_ptr<int64_t>(),
                block_ids_list.size(),
                layer,
                num_blocks_tot,
                bytes_per_plane,
                bytes_per_block,
                false // is_put = false
            );
        }

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            std::cerr << "[ERROR] Kernel launch failed: "
                    << cudaGetErrorString(launch_err) << std::endl;
            return false;
        }

    } else {  // Standard cudaMemcpyAsync path
        for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
            const int64_t gblock = block_ids_list[bi];
            const int64_t lblock = (num_blocks_in_file > 0) ? (gblock % num_blocks_in_file) : 0;
            const size_t cpu_block_base = static_cast<size_t>(lblock) * num_layers * bytes_per_block;

            for (int layer = 0; layer < num_layers; ++layer) {
                const auto& layer_tensor = dst_tensors[layer];
                auto* dst_base = reinterpret_cast<uint8_t*>(layer_tensor.data_ptr());

                // Compute CPU source offsets for K and V
                const size_t cpu_K_off = cpu_block_base + static_cast<size_t>(layer) * bytes_per_block;
                const size_t cpu_V_off = cpu_K_off + bytes_per_plane;

                // Compute GPU destination offsets for K and V
                const size_t gpu_K_off = static_cast<size_t>(gblock) * bytes_per_plane;
                const size_t gpu_V_off = (static_cast<size_t>(num_blocks_tot) + gblock) * bytes_per_plane;

                void* dst_K = dst_base + gpu_K_off;
                void* dst_V = dst_base + gpu_V_off;
                void* src_K = cpu_base + cpu_K_off;
                void* src_V = cpu_base + cpu_V_off;

                cudaError_t err1 = cudaMemcpyAsync(
                    dst_K, src_K, bytes_per_plane,
                    cudaMemcpyHostToDevice, stream.stream());

                cudaError_t err2 = cudaMemcpyAsync(
                    dst_V, src_V, bytes_per_plane,
                    cudaMemcpyHostToDevice, stream.stream());

                if (err1 != cudaSuccess || err2 != cudaSuccess) {
                    std::cerr << "[ERROR] cudaMemcpyAsync failed for block=" << gblock
                              << " layer=" << layer
                              << " err1=" << cudaGetErrorString(err1)
                              << " err2=" << cudaGetErrorString(err2)
                              << std::endl;
                    return false;
                }
            }
        }
    }

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
    // Create job state object to track progress and futures for this job.
    auto job_state = std::make_unique<JobState>();
    job_state->total_tasks = source_files.size();

    // For each source file, enqueue one async task in the I/O thread pool.
    for (size_t i = 0; i < source_files.size(); i++) {
        std::string src_file = source_files[i];
        auto block_ids = all_block_ids[i];
        auto future = g_io_pool->enqueue([=, job_state = job_state.get()]() -> bool {

            // Get dedicated stream for this thread
            if (!thread_stream.has_value()) {
                thread_stream = g_streams[thread_stream_idx % g_streams.size()];
                //thread_stream = at::cuda::getStreamFromPool(/* isHighPriority = */ true); // use high-priority stream for reads
            }

            // Save current CUDA stream so we can restore it later.
            auto current_stream = at::cuda::getCurrentCUDAStream();
            at::cuda::setCurrentCUDAStream(*thread_stream);

            // -------------------------
            // Stage 1: File → pinned CPU
            // -------------------------
            bool stage1_ok = false;
            torch::Tensor host_buf;
            try {
                // Read data from disk into pinned memory tensor.
                host_buf = TIME_EXPR("read phase 1: read_file_from_disk", read_file_from_disk(src_file),
                    ("file:" + src_file)
                );
                stage1_ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Stage1 read_file_from_disk failed for "
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
                    // Perform asynchronous GPU copy and tensor swap.
                    ok = TIME_EXPR("read phase 2: copy_buffer_to_gpu_tensors", copy_buffer_to_gpu_tensors(
                        host_buf, block_ids, dst_tensors, gpu_blocks_per_file, *thread_stream), "file: " + src_file);
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
            // Synchronize only this thread's CUDA stream.
            at::cuda::setCurrentCUDAStream(current_stream);
            job_state->completed_tasks.fetch_add(1);
            if (!ok) job_state->all_success = false;
            return ok;
        });
        // Convert std::future → std::shared_future- is copyable and can be waited on by multiple threads.

        job_state->futures.push_back(future.share());
    }

    std::lock_guard<std::mutex> lock(jobs_mutex);
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
          py::arg("max_pinned_memory_gb") = 50,
          py::arg("tp_rank") = 1);

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
}
