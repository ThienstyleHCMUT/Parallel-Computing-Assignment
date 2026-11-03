#include "algorithms.h"
#include <algorithm>
#include <cstring>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace {

constexpr int TASK_DEPTH = 3;   // recursion levels that spawn OpenMP tasks

// ---------------------------------------------------------------- base kernel
// C(n x n, ld ldc) = A * B, i-k-j order (cache friendly, no blocking: "naive")
void kernel(const double* A, size_t lda, const double* B, size_t ldb,
            double* C, size_t ldc, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        double* c = C + i * ldc;
        std::fill(c, c + n, 0.0);
        for (size_t k = 0; k < n; ++k) {
            const double a = A[i * lda + k];
            const double* b = B + k * ldb;
            for (size_t j = 0; j < n; ++j) c[j] += a * b[j];
        }
    }
}

// C(rows x n) = A(rows x n) * B(n x n), parallel over rows of A.
void naive_rows(const double* A, const double* B, double* C, size_t rows, size_t n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(rows); ++i) {
        double* c = C + static_cast<size_t>(i) * n;
        std::fill(c, c + n, 0.0);
        for (size_t k = 0; k < n; ++k) {
            const double a = A[static_cast<size_t>(i) * n + k];
            const double* b = B + k * n;
            for (size_t j = 0; j < n; ++j) c[j] += a * b[j];
        }
    }
}

// ------------------------------------------------------------------- Strassen
void strassen_rec(const double* A, size_t lda, const double* B, size_t ldb,
                  double* C, size_t ldc, size_t n, size_t leaf, int depth);

// D = X + sgn*Y  (h x h, D contiguous)
void combine_ops(std::vector<double>& D, size_t h, const double* X, size_t lx,
                 const double* Y, size_t ly, double sgn) {
    D.resize(h * h);
    for (size_t i = 0; i < h; ++i)
        for (size_t j = 0; j < h; ++j)
            D[i * h + j] = X[i * lx + j] + sgn * Y[i * ly + j];
}

// Computes product M_k (k = 0..6) into M (h x h contiguous)
//   M1=(A11+A22)(B11+B22)  M2=(A21+A22)B11   M3=A11(B12-B22)  M4=A22(B21-B11)
//   M5=(A11+A12)B22        M6=(A21-A11)(B11+B12)  M7=(A12-A22)(B21+B22)
void strassen_product(int k, const double* A, size_t lda, const double* B, size_t ldb,
                      size_t h, double* M, size_t leaf, int depth) {
    const double *A11 = A, *A12 = A + h, *A21 = A + h * lda, *A22 = A21 + h;
    const double *B11 = B, *B12 = B + h, *B21 = B + h * ldb, *B22 = B21 + h;
    std::vector<double> S, T;
    const double *pa = nullptr, *pb = nullptr;
    size_t la = h, lb = h;
    switch (k) {
        case 0: combine_ops(S, h, A11, lda, A22, lda, 1);  combine_ops(T, h, B11, ldb, B22, ldb, 1);
                pa = S.data(); pb = T.data(); break;
        case 1: combine_ops(S, h, A21, lda, A22, lda, 1);
                pa = S.data(); pb = B11; lb = ldb; break;
        case 2: combine_ops(T, h, B12, ldb, B22, ldb, -1);
                pa = A11; la = lda; pb = T.data(); break;
        case 3: combine_ops(T, h, B21, ldb, B11, ldb, -1);
                pa = A22; la = lda; pb = T.data(); break;
        case 4: combine_ops(S, h, A11, lda, A12, lda, 1);
                pa = S.data(); pb = B22; lb = ldb; break;
        case 5: combine_ops(S, h, A21, lda, A11, lda, -1); combine_ops(T, h, B11, ldb, B12, ldb, 1);
                pa = S.data(); pb = T.data(); break;
        default:combine_ops(S, h, A12, lda, A22, lda, -1); combine_ops(T, h, B21, ldb, B22, ldb, 1);
                pa = S.data(); pb = T.data(); break;
    }
    strassen_rec(pa, la, pb, lb, M, h, h, leaf, depth + 1);
}

// M = 7 contiguous h*h blocks -> C (n=2h, leading dim ldc)
void strassen_assemble(const double* M, double* C, size_t ldc, size_t h) {
    const size_t hh = h * h;
    const double *M1 = M, *M2 = M + hh, *M3 = M + 2 * hh, *M4 = M + 3 * hh,
                 *M5 = M + 4 * hh, *M6 = M + 5 * hh, *M7 = M + 6 * hh;
    for (size_t i = 0; i < h; ++i)
        for (size_t j = 0; j < h; ++j) {
            const size_t x = i * h + j;
            C[i * ldc + j]           = M1[x] + M4[x] - M5[x] + M7[x];
            C[i * ldc + h + j]       = M3[x] + M5[x];
            C[(h + i) * ldc + j]     = M2[x] + M4[x];
            C[(h + i) * ldc + h + j] = M1[x] - M2[x] + M3[x] + M6[x];
        }
}

void strassen_rec(const double* A, size_t lda, const double* B, size_t ldb,
                  double* C, size_t ldc, size_t n, size_t leaf, int depth) {
    if (n <= leaf || (n % 2) != 0) { kernel(A, lda, B, ldb, C, ldc, n); return; }
    const size_t h = n / 2;
    std::vector<double> M(7 * h * h);
    double* Mp = M.data();
    for (int k = 0; k < 7; ++k) {
        #pragma omp task if (depth < TASK_DEPTH) firstprivate(k) shared(Mp)
        strassen_product(k, A, lda, B, ldb, h, Mp + static_cast<size_t>(k) * h * h, leaf, depth);
    }
    #pragma omp taskwait
    strassen_assemble(Mp, C, ldc, h);
}

} // namespace

// ------------------------------------------------------------------ public API
void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    naive_rows(A.data(), B.data(), C.data(), A.rows(), A.cols());
}

size_t strassen_padded_size(size_t n, size_t leaf) {
    if (leaf < 1) leaf = 1;
    size_t s = n, L = 0;
    while (s > leaf) { s = (s + 1) / 2; ++L; }
    const size_t unit = size_t(1) << L;
    return ((n + unit - 1) / unit) * unit;
}

void strassen_multiply(const Matrix& A, const Matrix& B, Matrix& C, size_t leaf) {
    const size_t n = A.rows();
    const size_t m = strassen_padded_size(n, leaf);
    if (m == n) {
        #pragma omp parallel
        #pragma omp single
        strassen_rec(A.data(), n, B.data(), n, C.data(), n, n, leaf, 0);
        return;
    }
    Matrix Ap(m), Bp(m), Cp(m);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(Ap.data() + i * m, A.data() + i * n, n * sizeof(double));
        std::memcpy(Bp.data() + i * m, B.data() + i * n, n * sizeof(double));
    }
    #pragma omp parallel
    #pragma omp single
    strassen_rec(Ap.data(), m, Bp.data(), m, Cp.data(), m, m, leaf, 0);
    for (size_t i = 0; i < n; ++i)
        std::memcpy(C.data() + i * n, Cp.data() + i * m, n * sizeof(double));
}

// ------------------------------------------------------------------------ MPI
#ifdef USE_MPI
void naive_multiply_mpi(const double* A, const double* B, double* C,
                        size_t n, int rank, int size) {
    std::vector<int> rows(size), cnt(size), disp(size);
    const size_t base = n / size, extra = n % size;
    size_t off = 0;
    for (int r = 0; r < size; ++r) {
        rows[r] = static_cast<int>(base + (static_cast<size_t>(r) < extra ? 1 : 0));
        cnt[r]  = static_cast<int>(rows[r] * n);
        disp[r] = static_cast<int>(off * n);
        off += rows[r];
    }
    std::vector<double> la(static_cast<size_t>(cnt[rank])), lc(la.size()), bb;
    const double* Bp = B;
    if (rank != 0) { bb.resize(n * n); Bp = bb.data(); }

    MPI_Scatterv(A, cnt.data(), disp.data(), MPI_DOUBLE, la.data(), cnt[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(const_cast<double*>(Bp), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    naive_rows(la.data(), Bp, lc.data(), static_cast<size_t>(rows[rank]), n);   // OpenMP inside = hybrid
    MPI_Gatherv(lc.data(), cnt[rank], MPI_DOUBLE, C, cnt.data(), disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void strassen_multiply_mpi(const double* A, const double* B, double* C,
                           size_t n, size_t leaf, int rank, int size) {
    const size_t m = strassen_padded_size(n, leaf);
    if (m <= leaf) {                       // too small to split: rank 0 does it all
        if (rank == 0) kernel(A, n, B, n, C, n, n);
        return;
    }
    const size_t h = m / 2, hh = h * h;
    std::vector<double> Ap(m * m, 0.0), Bp(m * m, 0.0);
    if (rank == 0)
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(Ap.data() + i * m, A + i * n, n * sizeof(double));
            std::memcpy(Bp.data() + i * m, B + i * n, n * sizeof(double));
        }
    MPI_Bcast(Ap.data(), static_cast<int>(m * m), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Bp.data(), static_cast<int>(m * m), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> mine;                 // products owned by this rank
    for (int k = 0; k < 7; ++k) if (k % size == rank) mine.push_back(k);

    std::vector<double> store(rank == 0 ? 7 * hh : mine.size() * hh);
    auto slot = [&](size_t j, int k) { return store.data() + (rank == 0 ? static_cast<size_t>(k) : j) * hh; };

    #pragma omp parallel
    #pragma omp single
    for (size_t j = 0; j < mine.size(); ++j) {
        const int k = mine[j];
        double* out = slot(j, k);
        #pragma omp task firstprivate(k, out)
        strassen_product(k, Ap.data(), m, Bp.data(), m, h, out, leaf, 1);
    }

    if (rank != 0) {
        for (size_t j = 0; j < mine.size(); ++j)
            MPI_Send(slot(j, mine[j]), static_cast<int>(hh), MPI_DOUBLE, 0, mine[j], MPI_COMM_WORLD);
        return;
    }
    for (int k = 0; k < 7; ++k)
        if (k % size != 0)
            MPI_Recv(store.data() + static_cast<size_t>(k) * hh, static_cast<int>(hh), MPI_DOUBLE,
                     k % size, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<double> Cp(m * m);
    strassen_assemble(store.data(), Cp.data(), m, h);
    for (size_t i = 0; i < n; ++i)
        std::memcpy(C + i * n, Cp.data() + i * m, n * sizeof(double));
}
#endif
