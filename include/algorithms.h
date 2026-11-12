#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <cstddef>
#include "matrix.h"

// -----------------------------------------------------------------------------
// The SAME source gives different backends depending on compiler flags:
//   no flags                 -> sequential
//   -fopenmp                 -> OpenMP (shared memory)
//   -DUSE_MPI (mpicxx)       -> MPI
//   -DUSE_MPI -fopenmp       -> hybrid MPI + OpenMP
// -----------------------------------------------------------------------------

// C = A * B (all n x n). Parallelised with OpenMP when compiled with -fopenmp.
void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// Strassen. 'leaf' = block size at/below which the plain kernel is used.
// Any n is accepted (matrices are zero-padded internally, cost is negligible).
void strassen_multiply(const Matrix& A, const Matrix& B, Matrix& C, size_t leaf);

// Size the matrices are padded to for a given n and leaf (exposed for tests).
size_t strassen_padded_size(size_t n, size_t leaf);

#ifdef USE_MPI
// A, B, C are only dereferenced on rank 0 (pass nullptr elsewhere).
// Naive: row-block distribution (Scatterv A, Bcast B, Gatherv C).
void naive_multiply_mpi(const double* A, const double* B, double* C,
                        size_t n, int rank, int size);

// [STRASSEN-MPI] Strassen, MPI (+ OpenMP tasks inside every rank = hybrid).
// The top 'depth' recursion levels are unrolled into 7^depth independent
// products that are dealt round-robin to the ranks (task t -> rank t % size).
// depth <= 0 means "choose automatically" (see strassen_mpi_depth).
void strassen_multiply_mpi(const double* A, const double* B, double* C,
                           size_t n, size_t leaf, int rank, int size, int depth = 0);

// [STRASSEN-MPI] Depth actually used for (n, leaf, size, requested depth).
// Returns 0 when n <= leaf (rank 0 multiplies alone). Pure function: every
// rank gets the same answer, safe to call for logging.
int strassen_mpi_depth(size_t n, size_t leaf, int size, int requested = 0);
#endif

#endif
