#ifndef CSV_LOGGER_H
#define CSV_LOGGER_H

#include <string>
#include <fstream>
#include <iostream>

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
    bool correctness_passed
) {
    std::ifstream check_file(filepath);
    bool file_exists = check_file.good();
    check_file.close();

    std::ofstream file(filepath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Khong the mo file CSV: " << filepath << "\n";
        return;
    }

    if (!file_exists) {
        file << "Machine,Algorithm,Backend,MatrixSize,Threads,Processes,LeafSize,Trial,ExecutionTime_s,GFLOPS,Status\n";
    }

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
         << (correctness_passed ? "PASSED" : "FAILED") << "\n";

    file.close();
}

#endif
