// Correctness tests (run with: make test). Compiled with -fopenmp so the OpenMP paths are tested too.
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <string>
#include "matrix.h"
#include "timer.h"
#include "csvLogger.h"
#include "algorithms.h"

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { ++failures; std::cout << "  [FAIL] " << (msg) << "\n"; } \
                              else { std::cout << "  [ ok ] " << (msg) << "\n"; } } while (0)

static void reference(const Matrix& A, const Matrix& B, Matrix& C) {   // plain i-j-k
    const size_t n = A.rows();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0;
            for (size_t k = 0; k < n; ++k) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
}

int main() {
    std::cout << "== identity ==\n";
    {
        Matrix A(50), I(50), C(50);
        A.randomize(1); I.set_identity();
        naive_multiply(A, I, C);
        CHECK(are_matrices_equal(A, C), "A * I == A (naive)");
        strassen_multiply(A, I, C, 8);
        CHECK(are_matrices_equal(A, C), "A * I == A (strassen, n=50, leaf=8)");
    }

    std::cout << "== random sizes vs reference ==\n";
    const size_t sizes[] = {1, 2, 3, 17, 64, 100, 129, 256};
    for (size_t n : sizes) {
        Matrix A(n), B(n), R(n), C(n);
        A.randomize(11); B.randomize(22);
        reference(A, B, R);

        naive_multiply(A, B, C);
        CHECK(are_matrices_equal(R, C, 1e-6), "naive    n=" + std::to_string(n));
        CHECK(freivalds_verify(A, B, C),      "freivalds(naive) n=" + std::to_string(n));

        for (size_t leaf : {size_t(4), size_t(16), size_t(64)}) {
            C.set_zero();
            strassen_multiply(A, B, C, leaf);
            CHECK(are_matrices_equal(R, C, 1e-6),
                  "strassen n=" + std::to_string(n) + " leaf=" + std::to_string(leaf));
        }
    }

    std::cout << "== freivalds must reject a wrong result ==\n";
    {
        Matrix A(64), B(64), C(64);
        A.randomize(3); B.randomize(4);
        naive_multiply(A, B, C);
        C(10, 20) += 1.0;
        CHECK(!freivalds_verify(A, B, C), "corrupted C detected");
    }

    std::cout << "== padded size ==\n";
    CHECK(strassen_padded_size(10000, 128) == 10112, "padded(10000,128) == 10112");
    CHECK(strassen_padded_size(1024, 64) == 1024,    "padded(1024,64) == 1024");

    std::cout << "== csv logger ==\n";
    std::remove("benchmark_test.csv");
    write_benchmark_csv("benchmark_test.csv", "test", "naive", "seq", 10, 1, 1, 0, 1, 0.5, 1.0, true);
    std::ifstream f("benchmark_test.csv"); std::string l1, l2;
    std::getline(f, l1); std::getline(f, l2);
    CHECK(!l1.empty() && l2.find("PASSED") != std::string::npos, "header + row written");

    std::cout << (failures ? "\nSOME TESTS FAILED\n" : "\nALL TESTS PASSED\n");
    return failures ? 1 : 0;
}
