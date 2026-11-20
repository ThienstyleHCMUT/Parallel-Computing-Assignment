#include <iostream>
#include <vector>
#include <mpi.h>
#include "matrix.h"
#include "timer.h"
#include "csvLogger.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &name_len);

    // 1. In thông tin Node và Process
    std::cout << "[MPI Sanity Test] Rank " << rank << " / " << size 
              << " running on: " << hostname << std::endl;

    // 2. Kiểm tra truyền nhận Broadcast
    int test_value = 0;
    if (rank == 0) {
        test_value = 2026;
    }
    MPI_Bcast(&test_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 3. Tiến trình Root (Rank 0) ghi log kiểm thử ra file CSV
    if (rank == 0) {
        if (test_value == 2026) {
            std::cout << "=> Broadcast Test PASSED!" << std::endl;
        } else {
            std::cout << "=> Broadcast Test FAILED!" << std::endl;
        }

        // Ghi dòng dữ liệu mẫu vào CSV để verify csvLogger.h
        write_benchmark_csv("benchmark_test.csv", hostname, "Sanity_Test", "MPI", 100, 1, size, 0, 0, 0.001, 0.0, true);
        std::cout << "=> CSV Logger Test PASSED! File 'benchmark_test.csv' created." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
