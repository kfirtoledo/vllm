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
#include <cstdio>
#include <liburing.h>

namespace py = pybind11;

#define TIME_EXPR(label, expr) ([&]() { \
    auto __t0 = std::chrono::high_resolution_clock::now(); \
    auto __ret = (expr); \
    auto __t1 = std::chrono::high_resolution_clock::now(); \
    double __ms = std::chrono::duration<double, std::milli>(__t1 - __t0).count(); \
    std::cout << "[TIME] " << label << " took " << __ms << " ms\n"; \
    return __ret; \
})()
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

thread_local io_uring* t_ring = nullptr;

// Mutex protecting access to the jobs map
static std::mutex jobs_mutex;
// Global map of job_id → JobState, tracking async job progress
static std::map<int, std::unique_ptr<JobState>> jobs;

struct PinnedBufferInfo {
    void* ptr = nullptr;
    size_t size = 0;
};

static std::vector<PinnedBufferInfo> g_pinned_buffers;
void preallocate_pinned_buffers(size_t io_threads, size_t pinned_buffer_size_mb) {
    g_pinned_buffers.resize(io_threads);
    size_t alloc_bytes = pinned_buffer_size_mb * 1024 * 1024;

    for (size_t i = 0; i < io_threads; ++i) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, alloc_bytes);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to allocate pinned buffer for thread "
                      << i << ": " << cudaGetErrorString(err) << std::endl;
            g_pinned_buffers[i] = {nullptr, 0};
        } else {
            g_pinned_buffers[i] = {ptr, alloc_bytes};

        }
    }

    std::cout << "[INFO] Pre-allocated pinned buffer "
            << (alloc_bytes / (1024 * 1024))
            << " MB for " << io_threads << " threads" << std::endl;
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

public:
    IOThreadPool(int threads, size_t pinned_buffer_mb, int tp_rank) {
        // Initialize PyTorch threading globally (main thread only)
        at::init_num_threads();
        at::set_num_threads(1);

        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i, threads, pinned_buffer_mb, tp_rank] {
                // Compute unique CPU per thread
                int cpu_id = tp_rank * threads + i;
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(cpu_id, &cpuset);

                if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
                    std::cerr << "[WARN] Failed to set affinity for thread " << i
                              << " tp_rank=" << tp_rank << " to CPU " << cpu_id << "\n";
                }

                pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
                int actual_cpu = sched_getcpu();

                std::cout << "[INFO] IO thread " << i
                          << " (tid=" << tid << ", tp_rank=" << tp_rank
                          << ") pinned to CPU " << cpu_id
                          << " (running on CPU " << actual_cpu << ")\n";

                if (i < g_pinned_buffers.size() && g_pinned_buffers[i].ptr != nullptr) {
                    t_pinned_ptr = g_pinned_buffers[i].ptr;
                    t_pinned_size = g_pinned_buffers[i].size;
                    std::cout << "[INFO] IO thread " << i
                            << " attached to preallocated pinned buffer "
                            << (t_pinned_size / (1024 * 1024)) << " MB\n";
                } else {
                    std::cerr << "[WARN] IO thread " << i
                            << " has no preallocated pinned buffer\n";
                }
                // Initialize io_uring for this thread
                if (!t_ring) {
                    t_ring = new io_uring();
                    int ret = io_uring_queue_init(256, t_ring, 0);
                    if (ret < 0) {
                        std::cerr << "[ERROR] Failed to initialize io_uring for thread "
                                << i << ": " << strerror(-ret) << std::endl;
                        t_ring = nullptr;
                    } else {
                        std::cout << "[INFO] io_uring initialized for thread "
                                << i << " with depth 256\n";
                    }
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
void init_performance_resources(int io_threads = 0, size_t pinned_buffer_size_mb = 256 ,size_t max_pinned_memory_gb = 10, int tp_rank = 1) {
    if (!g_io_pool) {
        if (io_threads == 0) {
            io_threads = std::max(4u, std::thread::hardware_concurrency() / 2);
        }
        std::cout << "[INFO] Initializing IOThreadPool with "
                  << io_threads << " threads, "
                  << pinned_buffer_size_mb << " MB pinned buffer per thread, "
                  << max_pinned_memory_gb << " GB max pinned memory\n"
                  << std::thread::hardware_concurrency() << " hardware_concurrency\n";

        // Pre-allocate pinned buffers before launching threads
        preallocate_pinned_buffers(io_threads, pinned_buffer_size_mb);

        g_io_pool = std::make_unique<IOThreadPool>(io_threads, pinned_buffer_size_mb, tp_rank);
        // Create dedicated streams for each thread
        g_streams.clear();
        g_streams.reserve(io_threads);
        for (size_t i = 0; i < io_threads; i++) {
            g_streams.push_back(at::cuda::getStreamFromPool(/*isHighPriority=*/false));
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

    // Cleanup io_uring when thread exits
    if (t_ring) {
        io_uring_queue_exit(t_ring);
        delete t_ring;
        t_ring = nullptr;
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

    auto dtype = src_tensors[0].dtype();
    size_t element_size = src_tensors[0].element_size();

    // Calculate dimensions
    size_t total_elements = 0;
    for (int64_t block_id : block_ids_list) {
        for (const auto &tensor : src_tensors) {
            auto block = tensor.index({torch::indexing::Slice(), block_id});
            total_elements += block.numel();
        }
    }

    size_t required_bytes = total_elements * element_size;
    auto [pinned_ptr, pinned_size] = get_thread_local_pinned(required_bytes);

    // Create output tensor view
    torch::Tensor result_cpu = torch::from_blob(
        pinned_ptr,
        {static_cast<long>(total_elements)},
        torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true)
    );

    // Direct async copy without intermediate cat
    size_t offset = 0;
    for (int64_t block_id : block_ids_list) {
        for (const auto &tensor : src_tensors) {
            auto block = tensor.index({torch::indexing::Slice(), block_id}).contiguous();
            size_t block_bytes = block.nbytes();

            cudaMemcpyAsync(
                static_cast<char*>(pinned_ptr) + offset,
                block.data_ptr(),
                block_bytes,
                cudaMemcpyDeviceToHost,
                stream.stream()
            );

            offset += block_bytes;
        }
    }

    if (result_cpu.dtype() == torch::kBFloat16) {
        result_cpu = result_cpu.view(torch::kUInt16);
    }

    return result_cpu;
}

// File write helpers - optimized for large writes
bool write_file_to_disk(const std::string &target_path,
                                 const torch::Tensor &host_buf,
                                 io_uring* ring) {
    const void* data_ptr = host_buf.data_ptr();
    size_t nbytes = host_buf.nbytes();
    std::string tmp_path = target_path + ".tmp";

    int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        std::cerr << "[ERROR] open failed: " << tmp_path << "\n";
        return false;
    }

    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    if (!sqe) {
        std::cerr << "[ERROR] io_uring_get_sqe failed\n";
        ::close(fd);
        return false;
    }

    io_uring_prep_write(sqe, fd, data_ptr, nbytes, 0);
    sqe->flags |= IOSQE_IO_LINK; // optional: chain writes if batching

    int ret = io_uring_submit(ring);
    if (ret < 0) {
        std::cerr << "[ERROR] io_uring_submit failed: " << strerror(-ret) << "\n";
        ::close(fd);
        return false;
    }

    struct io_uring_cqe* cqe;
    ret = io_uring_wait_cqe(ring, &cqe);
    if (ret < 0 || cqe->res < 0) {
        std::cerr << "[ERROR] io_uring write failed: " << strerror(-cqe->res) << "\n";
        io_uring_cqe_seen(ring, cqe);
        ::close(fd);
        return false;
    }

    io_uring_cqe_seen(ring, cqe);
    ::close(fd);

    std::rename(tmp_path.c_str(), target_path.c_str());
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

    // For each target file, enqueue one async task in the I/O thread pool.
    for (size_t i = 0; i < target_files.size(); i++) {
        std::string target = target_files[i];
        auto src = src_tensors;
        auto bids = all_block_ids[i];

        auto future = g_io_pool->enqueue([=, job_state = job_state.get()]() -> bool {

          // Each thread gets a dedicated CUDA stream for async GPU ops.
            if (!thread_stream.has_value()) {
                thread_stream = g_streams[thread_stream_idx % g_streams.size()];
            }

            // Save current CUDA stream so we can restore it later.
            auto current_stream = at::cuda::getCurrentCUDAStream();
            at::cuda::setCurrentCUDAStream(*thread_stream);

            try {
                // Stage 1: Asynchronously copy tensors from GPU to pinned CPU buffer.
                auto host_buf = copy_gpu_tensors_to_buffer(src, bids, *thread_stream);

                // Stage 2: Synchronize only this thread's CUDA stream (not all).
                cudaStreamSynchronize(thread_stream->stream());

                // Stage 3: Write the pinned buffer to disk (blocking operation).
                //bool ok = TIME_EXPR("write_file_to_disk", write_file_to_disk(target, host_buf));
                bool ok = write_file_to_disk(target, host_buf, t_ring);

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
torch::Tensor read_file_to_pinned_tensor(const std::string& path, io_uring* ring) {
    if (ring == nullptr) {
        std::cerr << "[ERROR] io_uring ring is null for thread when reading " << path << "\n";
        throw std::runtime_error("io_uring ring not initialized");
    }

    // Get file size via stat
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        std::cerr << "[ERROR] stat failed for " << path << ": " << strerror(errno) << "\n";
        throw std::runtime_error("stat failed for " + path);
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    // Allocate pinned buffer
    auto [pinned_ptr, pinned_size] = get_thread_local_pinned(file_size);
    if (!pinned_ptr || pinned_size < file_size) {
        std::cerr << "[ERROR] pinned buffer too small for file " << path
                  << " need " << file_size << " have " << pinned_size << "\n";
        throw std::runtime_error("pinned buffer too small");
    }

    // Open the file
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[ERROR] open failed for " << path << ": " << strerror(errno) << "\n";
        throw std::runtime_error("open failed for " + path);
    }

    // Prepare and submit read
    io_uring_sqe* sqe = io_uring_get_sqe(ring);
    if (!sqe) {
        ::close(fd);
        std::cerr << "[ERROR] io_uring_get_sqe failed for " << path << "\n";
        throw std::runtime_error("io_uring_get_sqe failed");
    }

    io_uring_prep_read(sqe, fd, pinned_ptr, file_size, 0);

    int ret = io_uring_submit(ring);
    if (ret < 0) {
        ::close(fd);
        std::cerr << "[ERROR] io_uring_submit failed: " << strerror(-ret) << "\n";
        throw std::runtime_error("io_uring_submit failed");
    }

    // Wait for completion
    io_uring_cqe* cqe = nullptr;
    ret = io_uring_wait_cqe(ring, &cqe);
    if (ret < 0 || cqe->res < 0) {
        const int err = (ret < 0) ? -ret : -cqe->res;
        if (cqe) io_uring_cqe_seen(ring, cqe);
        ::close(fd);
        std::cerr << "[ERROR] io_uring read failed for " << path << ": " << strerror(err) << "\n";
        throw std::runtime_error("io_uring read failed");
    }

    io_uring_cqe_seen(ring, cqe);
    ::close(fd);

    // Wrap pinned buffer into a Torch tensor (no copy)
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCPU)
        .pinned_memory(true);

    return torch::from_blob(
        pinned_ptr,
        {static_cast<long>(file_size)},
        [pinned_ptr](void* /*unused*/) {},  // no-op deleter
        options
    );
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

    const auto shape = ref.sizes();  // [2, num_blocks_total, H, B, D]
    TORCH_CHECK(shape.size() == 5, "Expected shape [2, num_blocks, H, B, D]");

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
                //host_buf = read_file_to_pinned_tensor(src_file);
                host_buf = TIME_EXPR("read_file_to_pinned_tensor", read_file_to_pinned_tensor(src_file, t_ring));
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
