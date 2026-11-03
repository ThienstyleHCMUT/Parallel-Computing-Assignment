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
// Strassen: the 7 sub-products M1..M7 are distributed round-robin over ranks
// (so at most 7 ranks do useful work); each rank recurses with OpenMP tasks.
void strassen_multiply_mpi(const double* A, const double* B, double* C,
                           size_t n, size_t leaf, int rank, int size);
#endif

#endif
