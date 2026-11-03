// One driver for all backends. Backend is fixed at compile time:
//   bench_seq (none) | bench_omp (-fopenmp) | bench_mpi (-DUSE_MPI) | bench_hybrid (both)
// Usage: bench_xxx --algo naive|strassen --n N [--trials T] [--leaf L]
//                  [--machine NAME] [--csv FILE] [--no-verify]
#include <cstdlib>
#include <iostream>
#include <string>
#include "matrix.h"
#include "timer.h"
#include "csvLogger.h"
#include "algorithms.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
    int rank = 0, size = 1;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    std::string algo = "naive", machine = "unknown", csv = "results/results.csv";
    size_t n = 1000, leaf = 128;
    int trials = 3;
    bool verify = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--algo") algo = next();
        else if (a == "--n") n = std::stoul(next());
        else if (a == "--trials") trials = std::stoi(next());
        else if (a == "--leaf") leaf = std::stoul(next());
        else if (a == "--machine") machine = next();
        else if (a == "--csv") csv = next();
        else if (a == "--no-verify") verify = false;
    }
    if (algo != "naive" && algo != "strassen") {
        if (rank == 0) std::cerr << "unknown --algo " << algo << "\n";
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

#if defined(USE_MPI) && defined(_OPENMP)
    const std::string backend = "hybrid";
#elif defined(USE_MPI)
    const std::string backend = "mpi";
#elif defined(_OPENMP)
    const std::string backend = "omp";
#else
    const std::string backend = "seq";
#endif
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif

    // Only rank 0 owns the full matrices.
    Matrix A(rank == 0 ? n : 0), B(rank == 0 ? n : 0), C(rank == 0 ? n : 0);
    if (rank == 0) { A.randomize(1234); B.randomize(5678); }

    for (int t = 1; t <= trials; ++t) {
        if (rank == 0) C.set_zero();
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        Timer timer;
        timer.start();
#ifdef USE_MPI
        if (algo == "naive") naive_multiply_mpi(A.data(), B.data(), C.data(), n, rank, size);
        else                 strassen_multiply_mpi(A.data(), B.data(), C.data(), n, leaf, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
#else
        if (algo == "naive") naive_multiply(A, B, C);
        else                 strassen_multiply(A, B, C, leaf);
#endif
        const double sec = timer.stop();

        if (rank == 0) {                                   // verification is NOT timed
            const bool ok = verify ? freivalds_verify(A, B, C) : true;
            const double gf = calculate_gflops(n, sec);
            write_benchmark_csv(csv, machine, algo, backend, n, threads, size,
                                algo == "strassen" ? static_cast<int>(leaf) : 0,
                                t, sec, gf, ok);
            std::cout << algo << " " << backend << " n=" << n << " procs=" << size
                      << " threads=" << threads << " trial=" << t << "  "
                      << sec << " s  " << gf << " GFLOPS  " << (ok ? "PASSED" : "FAILED") << std::endl;
        }
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
