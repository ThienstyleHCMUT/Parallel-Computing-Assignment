// Correctness test for Strassen (all backends).
//   seq / omp : strassen_multiply          vs naive_multiply
//   mpi/hybrid: strassen_multiply_mpi      vs naive_multiply, for depth = auto,1,2,3
// Build: make tests      Run: ./bin/test_strassen_seq | ./bin/test_strassen_omp
//                              mpirun -np 4 ./bin/test_strassen_mpi   (or _hybrid)
// Exit code != 0 when any case fails. Set STRASSEN_TEST_VERBOSE=1 for one line per case.
#include <cstdio>
#include <cstdlib>
#include "matrix.h"
#include "algorithms.h"
#ifdef USE_MPI
#include <mpi.h>
#endif

// Returns the number of failed cases (only meaningful on rank 0).
int run_strassen_tests(int rank, int size) {
    const size_t sizes[]  = {1, 2, 3, 7, 33, 64, 100, 129, 257};   // odd / non power of 2 => padding
    const size_t leaves[] = {8, 32};
#ifdef USE_MPI
    const int depths[] = {0, 1, 2, 3};                              // 0 = automatic
#else
    const int depths[] = {0};
#endif
    const bool verbose = std::getenv("STRASSEN_TEST_VERBOSE") != nullptr;
    int failures = 0, cases = 0;
    for (size_t n : sizes) for (size_t leaf : leaves) for (int dep : depths) {
        Matrix A(rank == 0 ? n : 0), B(rank == 0 ? n : 0), C(rank == 0 ? n : 0), R(rank == 0 ? n : 0);
        if (rank == 0) { A.randomize(11 + static_cast<unsigned>(n)); B.randomize(22 + static_cast<unsigned>(n)); naive_multiply(A, B, R); }
#ifdef USE_MPI
        strassen_multiply_mpi(A.data(), B.data(), C.data(), n, leaf, rank, size, dep);
        const int eff = strassen_mpi_depth(n, leaf, size, dep);
#else
        (void)size; (void)dep;
        if (rank == 0) strassen_multiply(A, B, C, leaf);
        const int eff = 0;
#endif
        if (rank == 0) {
            ++cases;
            const bool ok = are_matrices_equal(C, R, 1e-8) && freivalds_verify(A, B, C, 5, 1e-8);
            if (!ok) ++failures;
            if (!ok || verbose)
                std::printf("[%s] n=%zu leaf=%zu depth_req=%d depth_used=%d procs=%d\n",
                            ok ? "PASS" : "FAIL", n, leaf, dep, eff, size);
        }
    }
    if (rank == 0) std::printf("strassen test: %d/%d cases passed (procs=%d)\n", cases - failures, cases, size);
    return failures;
}

#ifndef SIM_NO_MAIN
int main(int argc, char** argv) {
    int rank = 0, size = 1;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    (void)argc; (void)argv;
#endif
    int f = run_strassen_tests(rank, size);
#ifdef USE_MPI
    MPI_Bcast(&f, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    return f ? 1 : 0;
}
#endif
