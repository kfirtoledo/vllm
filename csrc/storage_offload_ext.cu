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

namespace py = pybind11;
// -------------------------------------
// Debugging and timing macros
// -------------------------------------

#define DEBUG_PRINT(msg)                                                     \
    do {                                                                     \
        const char* env = std::getenv("STORAGE_CONNECTOR_DEBUG");            \
        if (env && std::string(env) != "0")                                 \
            std::cout << "[DEBUG] " << msg << std::endl;                     \
    } while (0)

// Timing macro - measures only if STORAGE_CONNECTOR_DEBUG is not "0"
#define TIME_EXPR(label, expr, info_str) ([&]() {                                  \
    const char* env = std::getenv("STORAGE_CONNECTOR_DEBUG");                      \
    if (!(env && std::string(env) == "1")) {                                       \
        return (expr);                                                             \
    }                                                                              \
    auto __t0 = std::chrono::high_resolution_clock::now();                         \
    auto __ret = (expr);                                                           \
    auto __t1 = std::chrono::high_resolution_clock::now();                         \
    double __ms = std::chrono::duration<double, std::milli>(__t1 - __t0).count();  \
    std::cout << "[DEBUG][TIME] " << label << " took " << __ms << " ms | "         \
              << info_str << std::endl;                                            \
    return __ret;                                                                  \
})()

// -------------------------------------
// Thread-local pinned buffer
// -------------------------------------
struct PinnedBufferInfo {
    void* ptr = nullptr;
    size_t size = 0;
};

static thread_local void* t_pinned_ptr = nullptr;
static thread_local size_t t_pinned_size = 0;
static std::vector<PinnedBufferInfo> g_pinned_buffers;

static std::pair<void*, size_t> get_thread_local_pinned(size_t required_bytes) {
    if (!t_pinned_ptr || t_pinned_size < required_bytes) {
        if (t_pinned_ptr) {
            cudaFreeHost(t_pinned_ptr);
            t_pinned_ptr = nullptr;
            t_pinned_size = 0;
        }

        size_t alloc_size = std::max(required_bytes, (size_t)16 * 1024 * 1024);
        cudaError_t err = cudaHostAlloc(
            &t_pinned_ptr,
            alloc_size,
            cudaHostAllocMapped | cudaHostAllocPortable
        );

        if (err != cudaSuccess) {
            std::cerr << "[ERROR] cudaHostAlloc failed: "
                      << cudaGetErrorString(err) << "\n";
            t_pinned_ptr = nullptr;
            t_pinned_size = 0;
        } else {
            t_pinned_size = alloc_size;
            DEBUG_PRINT("[INFO] Thread " << std::this_thread::get_id()
                      << " allocated pinned buffer "
                      << (alloc_size / (1024 * 1024)) << " MB");
        }
    }
    return {t_pinned_ptr, t_pinned_size};
}

void preallocate_pinned_buffers(size_t io_threads, size_t pinned_buffer_size_mb) {
    g_pinned_buffers.resize(io_threads);
    size_t alloc_bytes = pinned_buffer_size_mb * 1024 * 1024;

    std::vector<std::thread> workers;
    workers.reserve(io_threads);

    for (size_t i = 0; i < io_threads; ++i) {
        workers.emplace_back([i, alloc_bytes]() {
            auto [ptr, size] = get_thread_local_pinned(alloc_bytes);
            if (!ptr) {
                std::cerr << "[ERROR] Failed to preallocate pinned buffer for thread "
                          << i << std::endl;
                g_pinned_buffers[i] = {nullptr, 0};
            } else {
                g_pinned_buffers[i] = {ptr, size};
            }
        });
    }

    // Wait for all threads to complete initialization
    for (auto& t : workers) t.join();

    std::cout << "[INFO] Pre-allocated pinned buffer "
              << (alloc_bytes / (1024 * 1024))
              << " MB for " << io_threads << " threads" << std::endl;
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


// Optimized memcpy kernel using vectorized loads
template<typename T = uint4>
__global__ void memcpy_kernel(const char* __restrict__ src,
                              char* __restrict__ dst,
                              size_t num_bytes) {
    // Vector size (default: 16 bytes if T=uint4)
    constexpr size_t vec_size = sizeof(T);
    // Number of full vector elements we can copy
    const size_t total_vecs = num_bytes / vec_size;
    // Global stride: total number of threads across all blocks in the grid (gridDim.x = number of blocks, blockDim.x = threads per block)
    // Each thread will advance by this stride in the loop to cover all data
    const size_t stride = gridDim.x * blockDim.x;
    // Reinterpret byte pointers as vector types to enable wide 16-byte loads/stores
    const T* src_vec = reinterpret_cast<const T*>(src);
    T* dst_vec = reinterpret_cast<T*>(dst);

    // Each thread copies multiple vector elements starting from its global index, jumping by 'stride' each iteration
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_vecs;
         idx += stride) {
        dst_vec[idx] = src_vec[idx];
    }

    // Remaining tail bytes
    size_t tail_start = total_vecs * vec_size;
    for (size_t i = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
         i < num_bytes;
         i += stride) {
        dst[i] = src[i];
    }
}

// Launch helper function
inline void launch_memcpy_kernel(
    const void* src,
    void* dst,
    size_t bytes,
    const c10::cuda::CUDAStream& stream) {

    if (bytes == 0) return;

    constexpr int threads = 256;
    // Compute number of blocks to cover all data; limit to 1024 for practicality
    int blocks = std::min((int)((bytes + threads - 1) / threads), 1024);

    memcpy_kernel<<<blocks, threads, 0, stream.stream()>>>(
        static_cast<const char*>(src),
        static_cast<char*>(dst),
        bytes);

    // Check for launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::cerr << "[ERROR] Kernel launch failed: "
                  << cudaGetErrorString(launch_err) << std::endl;
    }
}



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
    int m_device_id;

public:
       IOThreadPool(int threads, size_t pinned_buffer_mb, int tp_rank, int device_id) : m_device_id(device_id) {

        // Initialize PyTorch threading globally (main thread only)
        at::init_num_threads();
        at::set_num_threads(1);
        int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);  // number of available logical CPUs
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i, threads, pinned_buffer_mb, tp_rank, device_id, num_cpus] {

                // Set CUDA device for this thread FIRST
                cudaSetDevice(device_id);

                // Compute unique CPU per thread
                int cpu_id = (tp_rank * threads + i) % num_cpus;
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(cpu_id, &cpuset);

                if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
                    std::cerr << "[WARN] Failed to set affinity for thread " << i
                              << " tp_rank=" << tp_rank << " to CPU " << cpu_id << "\n";
                }

                pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
                int actual_cpu = sched_getcpu();

                DEBUG_PRINT("IO thread " << i
                          << " set CUDA device to " << device_id
                          << " (tid=" << tid << ", tp_rank=" << tp_rank
                          << ") pinned to CPU " << cpu_id
                          << " (running on CPU " << actual_cpu << ")");

                // Attach preallocated pinned buffer
                if (i < g_pinned_buffers.size() && g_pinned_buffers[i].ptr != nullptr) {
                    t_pinned_ptr = g_pinned_buffers[i].ptr;
                    t_pinned_size = g_pinned_buffers[i].size;
                    DEBUG_PRINT("IO thread " << i
                            << " attached to preallocated pinned buffer "
                            << (t_pinned_size / (1024 * 1024)) << " MB");
                } else {
                    std::cerr << "[WARN] IO thread " << i
                            << " has no preallocated pinned buffer\n";
                }

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
void init_performance_resources(int io_threads = 0, size_t pinned_buffer_size_mb = 256, size_t max_pinned_memory_gb = 10, int tp_rank = 1) {
    if (!g_io_pool) {
        if (io_threads == 0) {
            io_threads = std::max(4u, std::thread::hardware_concurrency() / 2);
        }

        // Get current device (should be set by vLLM before calling this)
        int device_id;
        cudaGetDevice(&device_id);

        std::cout << "[INFO] Initializing IOThreadPool with "
                  << io_threads << " threads on device " << device_id
                  << ", " << pinned_buffer_size_mb << " MB pinned buffer per thread, "
                  << max_pinned_memory_gb << " GB max pinned memory\n";

        // Enable GPU access to mapped host memory (needed only for cudaHostAllocMapped before any CUDA context)
        cudaSetDeviceFlags(cudaDeviceMapHost);

        // Pre-allocate pinned buffers before launching threads
        preallocate_pinned_buffers(io_threads, pinned_buffer_size_mb);

        // Pass device_id to thread pool
        g_io_pool = std::make_unique<IOThreadPool>(io_threads, pinned_buffer_size_mb,
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

    // Ensure we're on the same device as the source tensors
    // c10::cuda::CUDAGuard device_guard(src_tensors[0].device());

    const int64_t num_blocks_tot = shape[1];
    const int64_t H = shape[2];
    const int64_t B = shape[3];
    const int64_t D = shape[4];
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

            const char* env = std::getenv("USE_KERNEL_COPY_WRITE");
            bool use_kernel_copy_write = (env && std::string(env) == "1");

            if (!use_kernel_copy_write) {
                // Default behavior: Standard cudaMemcpyAsync path
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
            } else {
                // Experimental kernel path
                void* dst_K_mapped = nullptr;
                void* dst_V_mapped = nullptr;

                cudaError_t map_err1 = cudaHostGetDevicePointer(&dst_K_mapped, dst_K, 0);
                cudaError_t map_err2 = cudaHostGetDevicePointer(&dst_V_mapped, dst_V, 0);

                if (map_err1 != cudaSuccess || map_err2 != cudaSuccess) {
                    std::cerr << "[ERROR] cudaHostGetDevicePointer failed for block=" << gblock
                              << " layer=" << layer
                              << " err1=" << cudaGetErrorString(map_err1)
                              << " err2=" << cudaGetErrorString(map_err2)
                              << std::endl;
                    TORCH_CHECK(false, "cudaHostGetDevicePointer failed");
                }

                // Use GPU kernel to copy into mapped host memory
                launch_memcpy_kernel(src_K, dst_K_mapped, bytes_per_plane, stream);
                launch_memcpy_kernel(src_V, dst_V_mapped, bytes_per_plane, stream);
            }
        }
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "Kernel launch failed");
    }

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
    // Write first to a temporary file to ensure atomic rename later
    std::string tmp_path = target_path + ".tmp";

    // Define a larger buffer (1MB) to reduce syscall overhead and speed up I/O
    const size_t WRITE_BUFFER_SIZE = 1 * 1024 * 1024; // 1MB buffer

    std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary);
    if (!ofs) {
        std::cerr << "[ERROR] Failed to open temporary file for writing: " << tmp_path << "\n";
        return false;
    }

    // Allocate custom I/O buffer for this stream (replaces small default buffer)
    std::vector<char> buffer(WRITE_BUFFER_SIZE);
    // Apply the custom buffer to the file stream
    ofs.rdbuf()->pubsetbuf(buffer.data(), WRITE_BUFFER_SIZE);

    ofs.write(reinterpret_cast<const char*>(data_ptr), nbytes);
    ofs.close();
    // Atomically rename temp file to final target name after successful write
    return (std::rename(tmp_path.c_str(), target_path.c_str()) == 0);
}

// Async GPU → Storage transfer (PUT)
bool transfer_async_put_ext(int job_id,
                        std::vector<std::string> target_files,
                        std::vector<torch::Tensor> src_tensors,
                        std::vector<std::vector<int64_t>> all_block_ids) {

    auto job_state = std::make_unique<JobState>();
    job_state->total_tasks = target_files.size();

    // FIX 1: Store shared_ptr to tensors to avoid repeated refcount changes
    auto shared_tensors = std::make_shared<std::vector<torch::Tensor>>(std::move(src_tensors));

    for (size_t i = 0; i < target_files.size(); i++) {
        std::string target = target_files[i];
        auto bids = all_block_ids[i];

        // FIX 2: Capture shared_ptr by value (cheap) instead of tensor vector
        auto future = g_io_pool->enqueue([target, bids, shared_tensors, job_state = job_state.get()]() -> bool {

            // Ensure correct device is set (thread-local)
            int device_id;
            cudaGetDevice(&device_id);

            // Each thread gets a dedicated CUDA stream for async GPU ops.
            if (!thread_stream.has_value()) {
                thread_stream = at::cuda::getStreamFromPool(/* isHighPriority = */ false);
            }

            // Save current CUDA stream so we can restore it later.
            auto current_stream = at::cuda::getCurrentCUDAStream();
            at::cuda::setCurrentCUDAStream(*thread_stream);

            try {
                // FIX 3: Use reference to avoid copy - dereference shared_ptr
                const auto& src = *shared_tensors;

                // Stage 1: Asynchronously copy tensors from GPU to pinned CPU buffer.
                auto host_buf = copy_gpu_tensors_to_buffer(src, bids, *thread_stream);

                // FIX 4: Use event-based sync instead of stream sync for lower overhead
                cudaEvent_t copy_done;
                cudaEventCreate(&copy_done);
                cudaEventRecord(copy_done, thread_stream->stream());
                cudaEventSynchronize(copy_done);  // Lower overhead than cudaStreamSynchronize
                cudaEventDestroy(copy_done);

                // Stage 2: Write the pinned buffer to disk (blocking operation).
                bool ok = TIME_EXPR("write_file_to_disk ", write_file_to_disk(target, host_buf),
                  ("file:" + target + " size:" + std::to_string(host_buf.nbytes()))
                );

                if (!ok)
                    std::cerr << "[ERROR] PUT failed during file write: " << target << "\n";

                // Restore original CUDA stream for safety.
                at::cuda::setCurrentCUDAStream(current_stream);

                // Atomically mark task completion.
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

    std::lock_guard<std::mutex> lock(jobs_mutex);
    jobs[job_id] = std::move(job_state);

    return true;
}
// ----------------------------------------------------------------------
// Storage -> GPU (GET)
// ----------------------------------------------------------------------

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
// Copy buffer to GPU tensors asynchronously using cudaMemcpyAsync (no swap_blocks)
// CPU layout (contiguous): [num_blocks_in_file, num_layers, 2, H, B, D]
// GPU layout (contiguous): [2, num_blocks_total, H, B, D]
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

    // Ensure we're on the same device as the destination tensors
    c10::cuda::CUDAGuard device_guard(dst_tensors[0].device());

    const int64_t num_blocks_tot = shape[1];
    const int64_t H = shape[2];
    const int64_t B = shape[3];
    const int64_t D = shape[4];
    const int num_layers = static_cast<int>(dst_tensors.size());

    const size_t elem_size = ref.element_size();
    const size_t bytes_per_plane = static_cast<size_t>(H * B * D) * elem_size;
    const size_t bytes_per_block = 2 * bytes_per_plane;  // [K,V] in CPU buf

    auto* cpu_base = cpu_buf.data_ptr<uint8_t>();

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

            const char* env = std::getenv("USE_KERNEL_COPY_READ");
            bool use_kernel_copy_read = (!env || std::string(env) != "0");
            if (use_kernel_copy_read) { //Default behaviour
                // Replace cudaMemcpyAsync with kernel-based copy
                launch_memcpy_kernel(src_K, dst_K, bytes_per_plane, stream);
                launch_memcpy_kernel(src_V, dst_V, bytes_per_plane, stream);
            } else {
                // Standard cudaMemcpyAsync path
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

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
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
                // thread_stream = at::cuda::getStreamFromPool(/* isHighPriority = */ false);

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
                host_buf = TIME_EXPR("read_file_to_pinned_tensor", read_file_to_pinned_tensor(src_file),
                    ("file:" + src_file)
                );
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
                    // Perform asynchronous GPU copy and tensor swap.
                    ok = copy_buffer_to_gpu_tensors(
                        host_buf, block_ids, dst_tensors, gpu_blocks_per_file, *thread_stream);

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
