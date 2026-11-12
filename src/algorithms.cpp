#include "algorithms.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
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

// [STRASSEN-MPI] a read-only view of a sub-matrix (pointer + leading dimension)
struct View { const double* p; size_t ld; };

// [STRASSEN-MPI] Operands of product M_k (k = 0..6) for the 2h x 2h pair (a, b).
//   M1=(A11+A22)(B11+B22)  M2=(A21+A22)B11   M3=A11(B12-B22)  M4=A22(B21-B11)
//   M5=(A11+A12)B22        M6=(A21-A11)(B11+B12)  M7=(A12-A22)(B21+B22)
// Sums are written to S / T (contiguous h x h); an operand that is a plain
// quadrant is returned as a view into a / b (no copy).
void strassen_operands(int k, View a, View b, size_t h,
                       std::vector<double>& S, std::vector<double>& T,
                       View& pa, View& pb) {
    const double *A11 = a.p, *A12 = a.p + h, *A21 = a.p + h * a.ld, *A22 = A21 + h;
    const double *B11 = b.p, *B12 = b.p + h, *B21 = b.p + h * b.ld, *B22 = B21 + h;
    const size_t lda = a.ld, ldb = b.ld;
    switch (k) {
        case 0: combine_ops(S, h, A11, lda, A22, lda, 1);  combine_ops(T, h, B11, ldb, B22, ldb, 1);
                pa = {S.data(), h}; pb = {T.data(), h}; break;
        case 1: combine_ops(S, h, A21, lda, A22, lda, 1);
                pa = {S.data(), h}; pb = {B11, ldb}; break;
        case 2: combine_ops(T, h, B12, ldb, B22, ldb, -1);
                pa = {A11, lda}; pb = {T.data(), h}; break;
        case 3: combine_ops(T, h, B21, ldb, B11, ldb, -1);
                pa = {A22, lda}; pb = {T.data(), h}; break;
        case 4: combine_ops(S, h, A11, lda, A12, lda, 1);
                pa = {S.data(), h}; pb = {B22, ldb}; break;
        case 5: combine_ops(S, h, A21, lda, A11, lda, -1); combine_ops(T, h, B11, ldb, B12, ldb, 1);
                pa = {S.data(), h}; pb = {T.data(), h}; break;
        default:combine_ops(S, h, A12, lda, A22, lda, -1); combine_ops(T, h, B21, ldb, B22, ldb, 1);
                pa = {S.data(), h}; pb = {T.data(), h}; break;
    }
}

// Computes product M_k (k = 0..6) into M (h x h contiguous)
void strassen_product(int k, const double* A, size_t lda, const double* B, size_t ldb,
                      size_t h, double* M, size_t leaf, int depth) {
    std::vector<double> S, T;
    View pa{nullptr, 0}, pb{nullptr, 0};
    strassen_operands(k, View{A, lda}, View{B, ldb}, h, S, T, pa, pb);
    strassen_rec(pa.p, pa.ld, pb.p, pb.ld, M, h, h, leaf, depth + 1);
}

// [STRASSEN-MPI] one row i of the assembly: M = 7 contiguous h*h blocks -> C
inline void assemble_row(const double* M, double* C, size_t ldc, size_t h, size_t i) {
    const size_t hh = h * h;
    const double *M1 = M, *M2 = M + hh, *M3 = M + 2 * hh, *M4 = M + 3 * hh,
                 *M5 = M + 4 * hh, *M6 = M + 5 * hh, *M7 = M + 6 * hh;
    for (size_t j = 0; j < h; ++j) {
        const size_t x = i * h + j;
        C[i * ldc + j]           = M1[x] + M4[x] - M5[x] + M7[x];
        C[i * ldc + h + j]       = M3[x] + M5[x];
        C[(h + i) * ldc + j]     = M2[x] + M4[x];
        C[(h + i) * ldc + h + j] = M1[x] - M2[x] + M3[x] + M6[x];
    }
}

// M = 7 contiguous h*h blocks -> C (n=2h, leading dim ldc)
void strassen_assemble(const double* M, double* C, size_t ldc, size_t h) {
    for (size_t i = 0; i < h; ++i) assemble_row(M, C, ldc, h, i);
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

// =============================================================================
// [STRASSEN-MPI]  Strassen over MPI (+ OpenMP tasks inside each rank = hybrid)
//
//  Idea: unroll the first d recursion levels of Strassen. That gives 7^d
//  independent products ("tasks") of size q = m / 2^d. Task t (base-7 digits
//  k_1..k_d, k_1 most significant) needs the operands obtained by applying the
//  Strassen combination k_1, then k_2, ... to (A, B). Rank r computes the tasks
//  t = r, r+P, r+2P, ... (round robin) with strassen_rec, which uses OpenMP
//  tasks (so every core of the rank is busy). Results go to rank 0, which
//  re-assembles level by level (7 children -> 1 parent, children of parent g
//  are tasks 7g..7g+6, so they are already contiguous).
//
//  depth=1 : 7 tasks   (old design, useful up to 7 ranks)
//  depth=2 : 49 tasks  (useful up to 49 ranks, balances 2..6 ranks well)
//  depth=3 : 343 tasks (root needs (7/4)^3 m^2 doubles for the products!)
// =============================================================================
namespace {

constexpr int MAX_MPI_DEPTH = 3;

size_t ipow7(int d) { size_t r = 1; while (d-- > 0) r *= 7; return r; }

// number of halvings until the block size is <= leaf (same loop as padded_size)
int strassen_levels(size_t n, size_t leaf) {
    if (leaf < 1) leaf = 1;
    size_t s = n; int L = 0;
    while (s > leaf) { s = (s + 1) / 2; ++L; }
    return L;
}

// MPI counts are int: broadcast big buffers in chunks of 2^28 doubles
void bcast_large(double* buf, size_t count) {
    const size_t CH = size_t(1) << 28;
    for (size_t off = 0; off < count; off += CH)
        MPI_Bcast(buf + off, static_cast<int>(std::min(CH, count - off)), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Product number t (0 <= t < 7^d) of the unrolled recursion -> out (q x q contiguous)
void strassen_task(size_t t, int d, const double* A, const double* B, size_t m,
                   size_t leaf, double* out) {
    std::vector<std::vector<double>> S(d), T(d);    // one buffer pair per level (no aliasing)
    View a{A, m}, b{B, m};
    size_t s = m, div = ipow7(d - 1);
    for (int lvl = 0; lvl < d; ++lvl) {
        const int k = static_cast<int>((t / div) % 7);
        div /= 7;
        s /= 2;
        View na{nullptr, 0}, nb{nullptr, 0};
        strassen_operands(k, a, b, s, S[lvl], T[lvl], na, nb);
        a = na; b = nb;
    }
    strassen_rec(a.p, a.ld, b.p, b.ld, out, s, s, leaf, d);
}

// G parents, each built from 7 consecutive h x h blocks of M; parent g goes to
// out + g*4*h*h as a contiguous (2h x 2h) matrix.
void assemble_level(const double* M, double* out, size_t G, size_t h) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (long g = 0; g < static_cast<long>(G); ++g)
        for (long i = 0; i < static_cast<long>(h); ++i)
            assemble_row(M + static_cast<size_t>(g) * 7 * h * h,
                         out + static_cast<size_t>(g) * 4 * h * h, 2 * h, h, static_cast<size_t>(i));
}

} // namespace

int strassen_mpi_depth(size_t n, size_t leaf, int size, int requested) {
    const int dmax = std::min(strassen_levels(n, leaf), MAX_MPI_DEPTH);
    if (dmax <= 0) return 0;
    if (requested > 0) return std::min(requested, dmax);
    if (size < 1) size = 1;
    // makespan of d levels ~ ceil(7^d / P) / 7^d (relative units: a task costs 7^-d).
    // pick the smallest d that is within 10% of the best one: less traffic, less memory.
    double mk[MAX_MPI_DEPTH + 1] = {0, 0, 0, 0}, best = 1e300;
    for (int d = 1; d <= dmax; ++d) {
        const double tasks = static_cast<double>(ipow7(d));
        mk[d] = std::ceil(tasks / size) / tasks;
        best = std::min(best, mk[d]);
    }
    for (int d = 1; d <= dmax; ++d) if (mk[d] <= 1.10 * best) return d;
    return dmax;
}

void strassen_multiply_mpi(const double* A, const double* B, double* C,
                           size_t n, size_t leaf, int rank, int size, int depth) {
    const int d = strassen_mpi_depth(n, leaf, size, depth);
    if (d == 0) {                                   // n <= leaf: too small to split
        if (rank == 0) naive_rows(A, B, C, n, n);
        return;
    }
    const size_t m = strassen_padded_size(n, leaf); // divisible by 2^levels >= 2^d
    const size_t q = m >> d, qq = q * q, nt = ipow7(d);
    const size_t P = static_cast<size_t>(size), me = static_cast<size_t>(rank);

    // ---- 1. padded operands on every rank (rank 0 can use A,B directly if m == n)
    std::vector<double> Apad, Bpad;
    const double *Am, *Bm;
    if (rank == 0 && m == n) { Am = A; Bm = B; }
    else {
        Apad.assign(m * m, 0.0); Bpad.assign(m * m, 0.0);
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < static_cast<long>(n); ++i) {
                std::memcpy(Apad.data() + static_cast<size_t>(i) * m, A + static_cast<size_t>(i) * n, n * sizeof(double));
                std::memcpy(Bpad.data() + static_cast<size_t>(i) * m, B + static_cast<size_t>(i) * n, n * sizeof(double));
            }
        }
        Am = Apad.data(); Bm = Bpad.data();
    }
    bcast_large(const_cast<double*>(Am), m * m);
    bcast_large(const_cast<double*>(Bm), m * m);

    // ---- 2. result storage: rank 0 keeps all 7^d products (indexed by t),
    //         other ranks only their own (indexed by j = t / P). Not zero-filled.
    const size_t nmine = (me < nt) ? (nt - me + P - 1) / P : 0;
    std::unique_ptr<double[]> R(new double[(rank == 0 ? nt : nmine) * qq + 1]);
    std::vector<MPI_Request> reqs;
    if (rank == 0) {                                // post all receives first: overlap with compute
        reqs.reserve(nt);
        for (size_t t = 0; t < nt; ++t) {
            if (t % P == 0) continue;
            reqs.emplace_back();
            MPI_Irecv(R.get() + t * qq, static_cast<int>(qq), MPI_DOUBLE,
                      static_cast<int>(t % P), static_cast<int>(t), MPI_COMM_WORLD, &reqs.back());
        }
    }

    // ---- 3. compute my products: one OpenMP task each (+ nested tasks inside)
    #pragma omp parallel
    #pragma omp single
    for (size_t j = 0; j < nmine; ++j) {
        const size_t t = me + j * P;
        double* out = R.get() + (rank == 0 ? t : j) * qq;
        #pragma omp task firstprivate(t, out)
        strassen_task(t, d, Am, Bm, m, leaf, out);
    }

    // ---- 4a. workers: ship products to rank 0 and leave
    if (rank != 0) {
        std::vector<MPI_Request> sreq(nmine);
        for (size_t j = 0; j < nmine; ++j)
            MPI_Isend(R.get() + j * qq, static_cast<int>(qq), MPI_DOUBLE, 0,
                      static_cast<int>(me + j * P), MPI_COMM_WORLD, &sreq[j]);
        if (nmine) MPI_Waitall(static_cast<int>(nmine), sreq.data(), MPI_STATUSES_IGNORE);
        return;
    }

    // ---- 4b. rank 0: wait for the rest, then assemble level by level
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    std::vector<double>().swap(Apad);               // give the memory back before assembling
    std::vector<double>().swap(Bpad);

    std::vector<double> Cp;
    std::unique_ptr<double[]> cur = std::move(R);
    for (int lev = d; lev >= 1; --lev) {
        const size_t h = m >> lev, G = ipow7(lev - 1);
        std::unique_ptr<double[]> next;
        double* out;
        if (lev == 1) {                             // final level: 1 parent of size m
            if (m == n) out = C;
            else { Cp.resize(m * m); out = Cp.data(); }
        } else {
            next.reset(new double[G * 4 * h * h]);
            out = next.get();
        }
        assemble_level(cur.get(), out, G, h);
        cur = std::move(next);                      // frees the finer level
    }
    if (m != n) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < static_cast<long>(n); ++i)
            std::memcpy(C + static_cast<size_t>(i) * n, Cp.data() + static_cast<size_t>(i) * m, n * sizeof(double));
    }
}
#endif
