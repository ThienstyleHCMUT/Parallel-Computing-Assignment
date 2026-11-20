#include "matrix.h"
#include <omp.h>

using namespace std;

Matrix multiplyNaiveOpenMP(const Matrix& matrixA, const Matrix& matrixB, int numThreads) {
    size_t matrixSize = matrixA.rows();
    Matrix resultMatrix(matrixSize);

    const double* dataA = matrixA.data();
    const double* dataB = matrixB.data();
    double* dataC = resultMatrix.data();

    if (numThreads <= 0) {
        numThreads = omp_get_max_threads();
    }

    #pragma omp parallel for num_threads(numThreads) schedule(static)
    for (size_t i = 0; i < matrixSize; ++i) {
        for (size_t j = 0; j < matrixSize; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < matrixSize; ++k) {
                sum += dataA[i * matrixSize + k] * dataB[k * matrixSize + j];
            }
            dataC[i * matrixSize + j] = sum;
        }
    }

    return resultMatrix;
}
