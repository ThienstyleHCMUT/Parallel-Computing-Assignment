#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    bool is_running_{false};

public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = true;
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time_;
        is_running_ = false;
        return elapsed.count();
    }

    double elapsed() const {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time_;
        return elapsed.count();
    }
};

inline double calculate_gflops(size_t n, double time_sec) {
    if (time_sec <= 0.0) return 0.0;
    double total_ops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    return (total_ops / time_sec) / 1e9;
}

#endif
