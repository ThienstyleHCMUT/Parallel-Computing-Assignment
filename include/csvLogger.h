#ifndef CSV_LOGGER_H
#define CSV_LOGGER_H

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

inline void write_benchmark_csv(
    const std::string& filepath,
    const std::string& machine_name,
    const std::string& algo_name,
    const std::string& backend,
    size_t matrix_size,
    int num_threads,
    int num_procs,
    int leaf_size,
    int trial_id,
    double execution_time,
    double gflops,
    bool correctness_passed,
    bool was_verified = true
) {
    std::ifstream check_file(filepath);
    bool file_exists = check_file.good() && check_file.peek() != std::ifstream::traits_type::eof();
    check_file.close();

    std::ofstream file(filepath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Khong the mo file CSV: " << filepath << "\n";
        return;
    }

    if (!file_exists) {
        file << "Machine,Algorithm,Backend,MatrixSize,Threads,Processes,LeafSize,Trial,ExecutionTime_s,GFLOPS,Status\n";
    }

    file << std::setprecision(12);
    file << machine_name << ","
         << algo_name << ","
         << backend << ","
         << matrix_size << ","
         << num_threads << ","
         << num_procs << ","
         << leaf_size << ","
         << trial_id << ","
         << execution_time << ","
         << gflops << ","
         << (!was_verified ? "SKIPPED" : (correctness_passed ? "PASSED" : "FAILED")) << "\n";

    file.close();
}

#endif
