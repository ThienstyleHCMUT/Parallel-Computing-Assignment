#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <iostream>
#include <stdexcept>

class Matrix {
private:
    size_t rows_{0};
    size_t cols_{0};
    double* data_{nullptr};

public:
    Matrix(size_t rows, size_t cols);
    explicit Matrix(size_t n) : Matrix(n, n) {}
    ~Matrix();

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    void randomize(unsigned int seed, double min_val = -10.0, double max_val = 10.0);
    void set_identity();
    void set_zero();

    inline double& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    inline const double& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    double* data() { return data_; }
    const double* data() const { return data_; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    Matrix clone() const;
    void print(size_t max_display = 6) const;
};

bool are_matrices_equal(const Matrix& A, const Matrix& B, double tol = 1e-5);
bool freivalds_verify(const Matrix& A, const Matrix& B, const Matrix& C, int iterations = 5, double tol = 1e-5);

Matrix multiplyNaive(const Matrix& matrixA, const Matrix& matrixB);
Matrix multiplyStrassen(const Matrix& matrixA, const Matrix& matrixB, size_t threshold = 64);
Matrix multiplyNaiveOpenMP(const Matrix& matrixA, const Matrix& matrixB, int numThreads = 0);
Matrix multiplyStrassenOpenMP(const Matrix& matrixA, const Matrix& matrixB, size_t threshold = 64, int numThreads = 0);

#endif
