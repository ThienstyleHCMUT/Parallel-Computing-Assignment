#include <iostream>
#include <cassert>
#include <cmath>
#include "matrix.h"
#include "timer.h"
#include "csvLogger.h"

int main() {
    std::cout << "=== BAT DAU KIEM TRA KHUNG CODE MATRIX ===\n";

    size_t N = 4;
    Matrix A(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A(i, j) = static_cast<double>(i * N + j);
        }
    }
    assert(A(0, 0) == 0.0);
    assert(A(1, 0) == 4.0);
    assert(A(3, 3) == 15.0);

    double* raw_ptr = A.data();
    assert(raw_ptr[5] == A(1, 1));
    std::cout << "[PASS] Test 1: Khoi tao va cap phat phang thanh cong!\n";

    Matrix I(N);
    I.set_identity();
    assert(I(0, 0) == 1.0 && I(1, 1) == 1.0 && I(0, 1) == 0.0);

    Matrix Z(N);
    Z.set_zero();
    assert(Z(2, 2) == 0.0 && Z(0, 3) == 0.0);
    std::cout << "[PASS] Test 2: Set Zero va Identity thanh cong!\n";

    Matrix R(N);
    R.randomize(0, -5.0, 5.0);
    bool has_nonzero = false;
    for (size_t i = 0; i < R.size(); ++i) {
        assert(R.data()[i] >= -5.0 && R.data()[i] <= 5.0);
        if (std::abs(R.data()[i]) > 1e-4) has_nonzero = true;
    }
    assert(has_nonzero);
    std::cout << "[PASS] Test 3: Randomize voi seed = 0 an toan thanh cong!\n";

    Matrix A_copy = A.clone();
    assert(are_matrices_equal(A, A_copy));

    Matrix A_moved = std::move(A_copy);
    assert(A_moved.rows() == N && A_moved.cols() == N);
    assert(A_copy.data() == nullptr);
    std::cout << "[PASS] Test 4: Clone va Move semantics thanh cong!\n";

    Matrix C(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A(i, k) * I(k, j);
            }
            C(i, j) = sum;
        }
    }
    assert(freivalds_verify(A, I, C) == true);

    C(0, 0) += 10.0;
    assert(freivalds_verify(A, I, C) == false);
    std::cout << "[PASS] Test 5: Kiem tra tinh dung dan Freivalds Algorithm O(N^2) thanh cong!\n";

    Timer timer;
    timer.start();
    volatile double dummy = 0.0;
    for (int i = 0; i < 1000000; ++i) dummy += i;
    double elapsed = timer.stop();
    assert(elapsed >= 0.0);

    double gflops = calculate_gflops(N, elapsed);
    write_benchmark_csv("benchmark_test.csv", "LocalMachine", "NaiveGEMM", "Sequential", N, 1, 1, 0, 1, elapsed, gflops, true);
    std::cout << "[PASS] Test 6: Timer va CSV Logger hoat dong tot!\n";

    std::cout << "\n>>> TAT CA KIEM TRA DEU THANH CONG! KHUNG CODE CHUAN SAN SANG TRIEN KHAI! <<<\n";
    return 0;
}
