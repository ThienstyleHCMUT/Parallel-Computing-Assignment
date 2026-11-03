#include "matrix.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>

#if defined(_WIN32)
    #include <malloc.h>
#endif

static void* allocate_aligned_memory(size_t bytes, size_t alignment = 64) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, alignment);
    if (!ptr) throw std::bad_alloc();
#else
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        throw std::bad_alloc();
    }
#endif
    return ptr;
}

static void free_aligned_memory(void* ptr) {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(nullptr) {
    if (rows == 0 || cols == 0) return;

    size_t total_bytes = rows_ * cols_ * sizeof(double);
    data_ = static_cast<double*>(allocate_aligned_memory(total_bytes, 64));
    std::memset(data_, 0, total_bytes);
}

Matrix::~Matrix() {
    if (data_) {
        free_aligned_memory(data_);
        data_ = nullptr;
    }
}

Matrix::Matrix(Matrix&& other) noexcept 
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_ = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        if (data_) {
            free_aligned_memory(data_);
        }
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;

        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

Matrix Matrix::clone() const {
    Matrix copy(rows_, cols_);
    if (data_ && copy.data_) {
        std::memcpy(copy.data_, data_, rows_ * cols_ * sizeof(double));
    }
    return copy;
}

void Matrix::set_zero() {
    if (data_) {
        std::memset(data_, 0, rows_ * cols_ * sizeof(double));
    }
}

void Matrix::set_identity() {
    set_zero();
    size_t min_dim = std::min(rows_, cols_);
    for (size_t i = 0; i < min_dim; ++i) {
        (*this)(i, i) = 1.0;
    }
}

void Matrix::randomize(unsigned int seed, double min_val, double max_val) {
    uint64_t state = (seed == 0) ? 0x853c49e6748fea9bULL : seed;

    auto simple_rand = [&state]() -> double {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        uint64_t val = state * 0x2545F4914F6CDD1DULL;
        return static_cast<double>(val >> 11) / (1ULL << 53);
    };

    size_t total = rows_ * cols_;
    double range = max_val - min_val;
    for (size_t i = 0; i < total; ++i) {
        data_[i] = min_val + simple_rand() * range;
    }
}

void Matrix::print(size_t max_display) const {
    size_t r_limit = std::min(rows_, max_display);
    size_t c_limit = std::min(cols_, max_display);
    std::cout << "Matrix (" << rows_ << "x" << cols_ << "):\n";
    for (size_t i = 0; i < r_limit; ++i) {
        std::cout << "  [ ";
        for (size_t j = 0; j < c_limit; ++j) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(3) << (*this)(i, j) << " ";
        }
        if (cols_ > max_display) std::cout << "... ";
        std::cout << "]\n";
    }
    if (rows_ > max_display) {
        std::cout << "  ... (" << rows_ - max_display << " rows omitted)\n";
    }
}

bool are_matrices_equal(const Matrix& A, const Matrix& B, double tol) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }

    size_t n = A.size();
    const double* a = A.data();
    const double* b = B.data();

    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(a[i] - b[i]);
        double max_val = std::max(std::abs(a[i]), std::abs(b[i]));
        if (diff > tol && (diff / (max_val + 1e-9)) > tol) {
            return false;
        }
    }
    return true;
}

bool freivalds_verify(const Matrix& A, const Matrix& B, const Matrix& C, int iterations, double tol) {
    if (A.rows() != C.rows() || A.cols() != B.rows() || B.cols() != C.cols()) {
        return false;
    }

    size_t m = A.rows();
    size_t k = A.cols();
    size_t n = B.cols();

    std::vector<double> r(n);
    std::vector<double> v(k);
    std::vector<double> u(m);
    std::vector<double> w(m);

    uint64_t rng_state = 123456789ULL;
    auto quick_rand_sign = [&rng_state]() -> double {
        rng_state ^= rng_state >> 12;
        rng_state ^= rng_state << 25;
        rng_state ^= rng_state >> 27;
        return (rng_state & 1) ? 1.0 : -1.0;
    };

    for (int it = 0; it < iterations; ++it) {
        for (size_t j = 0; j < n; ++j) {
            r[j] = quick_rand_sign();
        }

        for (size_t i = 0; i < k; ++i) {
            double sum = 0.0;
            const double* b_row = B.data() + i * n;
            for (size_t j = 0; j < n; ++j) {
                sum += b_row[j] * r[j];
            }
            v[i] = sum;
        }

        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            const double* a_row = A.data() + i * k;
            for (size_t j = 0; j < k; ++j) {
                sum += a_row[j] * v[j];
            }
            u[i] = sum;
        }

        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            const double* c_row = C.data() + i * n;
            for (size_t j = 0; j < n; ++j) {
                sum += c_row[j] * r[j];
            }
            w[i] = sum;
        }

        for (size_t i = 0; i < m; ++i) {
            double diff = std::abs(u[i] - w[i]);
            double scale = std::max(std::abs(u[i]), std::abs(w[i]));
            if (diff > tol && (diff / (scale + 1e-9)) > tol) {
                return false;
            }
        }
    }
    return true;
}
