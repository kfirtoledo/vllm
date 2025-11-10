// storage_thread_pool.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <sys/syscall.h>
#include <unistd.h>
#include <numa.h>

#include "buffer.cpp"
#include "debug_utils.cpp"
// ----------------------------------
// Thread-local globals
// ----------------------------------
extern thread_local void* t_pinned_ptr;
extern thread_local size_t t_pinned_size;
extern thread_local size_t thread_stream_idx;

// -------------------------------
// I/O thread pool for scheduling
// -------------------------------
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    int m_device_id;

public:
    ThreadPool(int threads, size_t pinned_buffer_mb, int tp_rank, int device_id): m_device_id(device_id) {

        // Initialize PyTorch threading globally (main thread only)
        // at::init_num_threads();
        // at::set_num_threads(1);

        // Get GPU NUMA node ONCE outside the thread loop
        int gpu_numa = get_gpu_numa_node(device_id);
        std::cout << "[INFO] GPU " << device_id << " mapped to NUMA node " << gpu_numa << "\n";

        // Get all CPUs in that NUMA node
        auto local_cpus = get_cpus_in_numa_node(gpu_numa);

        if (local_cpus.empty()) {
            std::cerr << "[WARN] No CPUs found for NUMA node " << gpu_numa
                    << ". System may not be NUMA-aware. Using all CPUs.\n";
            // Populate with all available CPUs as fallback
            int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
            for (int i = 0; i < num_cpus; ++i) {
                local_cpus.push_back(i);
            }
        }

        std::cout << "CPUs available for GPU " << device_id << " (NUMA " << gpu_numa << "): ";
        for (int cpu : local_cpus) std::cout << cpu << " ";
        std::cout << "\n";

        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i, threads, pinned_buffer_mb, tp_rank, device_id,
                                gpu_numa, local_cpus] {

                cudaSetDevice(device_id);

                // Round-robin CPUs within the NUMA node
                int cpu_id = local_cpus[i % local_cpus.size()];

                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(cpu_id, &cpuset);

                if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
                    std::cerr << "[ERROR] Failed to set affinity for thread " << i
                            << " to CPU " << cpu_id << "\n";
                }

                int actual_cpu = sched_getcpu();
                pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
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

        std::cout << "[INFO] All " << threads << " I/O threads initialized with pinned buffers\n";
    }


    ~ThreadPool() {
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
