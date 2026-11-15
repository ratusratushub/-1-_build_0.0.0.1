#pragma once
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <exception>
#include <utility>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
namespace space {
class ThreadPool {
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> jobs_;
    bool closing_ = false;
    std::vector<std::jthread> workers_;
public:
    explicit ThreadPool(unsigned count = std::thread::hardware_concurrency()) {
        try {
            for (unsigned i = 0; i < std::max(1u, count); ++i)
                workers_.emplace_back([this] {
                    for (;;) {
                        std::function<void()> job;
                        {
                            std::unique_lock lock(mutex_);
                            ready_.wait(lock, [this] { return closing_ || !jobs_.empty(); });
                            if (closing_ && jobs_.empty()) return;
                            job = std::move(jobs_.front()); jobs_.pop_front();
                        }
                        job();
                    }
                });
        } catch (...) { close(); throw; }
    }
    ~ThreadPool() { close(); }
    void close() noexcept {
        { std::lock_guard lock(mutex_); closing_ = true; }
        ready_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    template<class F> std::future<void> submit(F&& f) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
        auto result = task->get_future();
        { std::lock_guard lock(mutex_);
          if (closing_) throw std::runtime_error("ThreadPool is closed");
          jobs_.emplace_back([task] { (*task)(); }); }
        ready_.notify_one(); return result;
    }
    // Coarse ranges avoid one task allocation per particle or cell.
    template<class F> void parallelFor(std::size_t count, std::size_t grain, F fn) {
        if (!grain) throw std::invalid_argument("zero task grain");
        std::vector<std::future<void>> tasks;
        tasks.reserve(count/grain+(count%grain!=0));
        std::exception_ptr error;
        try {
            for (std::size_t first = 0; first < count; first += grain)
                tasks.push_back(submit([=] { fn(first, std::min(count, first + grain)); }));
        } catch (...) { error=std::current_exception(); }
        for (auto& task : tasks) {
            try { task.get(); } catch (...) { if (!error) error = std::current_exception(); }
        }
        if (error) std::rethrow_exception(error);
    }
};
}
