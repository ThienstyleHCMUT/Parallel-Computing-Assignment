#include "matrix.h"
#include <omp.h>

using namespace std;

namespace {
Matrix addMatrices(const Matrix& matrixA, const Matrix& matrixB) {
    size_t matrixSize = matrixA.rows();
    Matrix resultMatrix(matrixSize);
    const double* dataA = matrixA.data();
    const double* dataB = matrixB.data();
    double* dataResult = resultMatrix.data();
    size_t totalElements = matrixSize * matrixSize;
    for (size_t i = 0; i < totalElements; ++i) {
        dataResult[i] = dataA[i] + dataB[i];
    }
    return resultMatrix;
}

Matrix subtractMatrices(const Matrix& matrixA, const Matrix& matrixB) {
    size_t matrixSize = matrixA.rows();
    Matrix resultMatrix(matrixSize);
    const double* dataA = matrixA.data();
    const double* dataB = matrixB.data();
    double* dataResult = resultMatrix.data();
    size_t totalElements = matrixSize * matrixSize;
    for (size_t i = 0; i < totalElements; ++i) {
        dataResult[i] = dataA[i] - dataB[i];
    }
    return resultMatrix;
}

void splitMatrix(const Matrix& parentMatrix, Matrix& sub11, Matrix& sub12, Matrix& sub21, Matrix& sub22) {
    size_t halfSize = parentMatrix.rows() / 2;
    for (size_t i = 0; i < halfSize; ++i) {
        for (size_t j = 0; j < halfSize; ++j) {
            sub11(i, j) = parentMatrix(i, j);
            sub12(i, j) = parentMatrix(i, j + halfSize);
            sub21(i, j) = parentMatrix(i + halfSize, j);
            sub22(i, j) = parentMatrix(i + halfSize, j + halfSize);
        }
    }
}

Matrix multiplyStrassenOpenMPRecursive(const Matrix& matrixA, const Matrix& matrixB, size_t threshold, int currentDepth, int maxDepth) {
    size_t matrixSize = matrixA.rows();

    if (matrixSize <= threshold) {
        return multiplyNaive(matrixA, matrixB);
    }

    if (matrixSize % 2 != 0) {
        size_t paddedSize = matrixSize + 1;
        Matrix paddedA(paddedSize);
        Matrix paddedB(paddedSize);
        paddedA.set_zero();
        paddedB.set_zero();

        for (size_t i = 0; i < matrixSize; ++i) {
            for (size_t j = 0; j < matrixSize; ++j) {
                paddedA(i, j) = matrixA(i, j);
                paddedB(i, j) = matrixB(i, j);
            }
        }

        Matrix paddedResult = multiplyStrassenOpenMPRecursive(paddedA, paddedB, threshold, currentDepth, maxDepth);

        Matrix resultMatrix(matrixSize);
        for (size_t i = 0; i < matrixSize; ++i) {
            for (size_t j = 0; j < matrixSize; ++j) {
                resultMatrix(i, j) = paddedResult(i, j);
            }
        }
        return resultMatrix;
    }

    size_t halfSize = matrixSize / 2;

    Matrix a11(halfSize);
    Matrix a12(halfSize);
    Matrix a21(halfSize);
    Matrix a22(halfSize);

    Matrix b11(halfSize);
    Matrix b12(halfSize);
    Matrix b21(halfSize);
    Matrix b22(halfSize);

    splitMatrix(matrixA, a11, a12, a21, a22);
    splitMatrix(matrixB, b11, b12, b21, b22);

    Matrix m1(halfSize);
    Matrix m2(halfSize);
    Matrix m3(halfSize);
    Matrix m4(halfSize);
    Matrix m5(halfSize);
    Matrix m6(halfSize);
    Matrix m7(halfSize);

    if (currentDepth < maxDepth) {
        #pragma omp task shared(m1, a11, a22, b11, b22)
        {
            m1 = multiplyStrassenOpenMPRecursive(addMatrices(a11, a22), addMatrices(b11, b22), threshold, currentDepth + 1, maxDepth);
        }

        #pragma omp task shared(m2, a21, a22, b11)
        {
            m2 = multiplyStrassenOpenMPRecursive(addMatrices(a21, a22), b11, threshold, currentDepth + 1, maxDepth);
        }

        #pragma omp task shared(m3, a11, b12, b22)
        {
            m3 = multiplyStrassenOpenMPRecursive(a11, subtractMatrices(b12, b22), threshold, currentDepth + 1, maxDepth);
        }

        #pragma omp task shared(m4, a22, b21, b11)
        {
            m4 = multiplyStrassenOpenMPRecursive(a22, subtractMatrices(b21, b11), threshold, currentDepth + 1, maxDepth);
        }

        #pragma omp task shared(m5, a11, a12, b22)
        {
            m5 = multiplyStrassenOpenMPRecursive(addMatrices(a11, a12), b22, threshold, currentDepth + 1, maxDepth);
        }

        #pragma omp task shared(m6, a21, a11, b11, b12)
        {
            m6 = multiplyStrassenOpenMPRecursive(subtractMatrices(a21, a11), addMatrices(b11, b12), threshold, currentDepth + 1, maxDepth);
        }

        #pragma omp task shared(m7, a12, a22, b21, b22)
        {
            m7 = multiplyStrassenOpenMPRecursive(subtractMatrices(a12, a22), addMatrices(b21, b22), threshold, currentDepth + 1, maxDepth);
        }

        #pragma omp taskwait
    } else {
        m1 = multiplyStrassenOpenMPRecursive(addMatrices(a11, a22), addMatrices(b11, b22), threshold, currentDepth + 1, maxDepth);
        m2 = multiplyStrassenOpenMPRecursive(addMatrices(a21, a22), b11, threshold, currentDepth + 1, maxDepth);
        m3 = multiplyStrassenOpenMPRecursive(a11, subtractMatrices(b12, b22), threshold, currentDepth + 1, maxDepth);
        m4 = multiplyStrassenOpenMPRecursive(a22, subtractMatrices(b21, b11), threshold, currentDepth + 1, maxDepth);
        m5 = multiplyStrassenOpenMPRecursive(addMatrices(a11, a12), b22, threshold, currentDepth + 1, maxDepth);
        m6 = multiplyStrassenOpenMPRecursive(subtractMatrices(a21, a11), addMatrices(b11, b12), threshold, currentDepth + 1, maxDepth);
        m7 = multiplyStrassenOpenMPRecursive(subtractMatrices(a12, a22), addMatrices(b21, b22), threshold, currentDepth + 1, maxDepth);
    }

    Matrix resultMatrix(matrixSize);

    for (size_t i = 0; i < halfSize; ++i) {
        for (size_t j = 0; j < halfSize; ++j) {
            resultMatrix(i, j) = m1(i, j) + m4(i, j) - m5(i, j) + m7(i, j);
            resultMatrix(i, j + halfSize) = m3(i, j) + m5(i, j);
            resultMatrix(i + halfSize, j) = m2(i, j) + m4(i, j);
            resultMatrix(i + halfSize, j + halfSize) = m1(i, j) - m2(i, j) + m3(i, j) + m6(i, j);
        }
    }

    return resultMatrix;
}
}

Matrix multiplyStrassenOpenMP(const Matrix& matrixA, const Matrix& matrixB, size_t threshold, int numThreads) {
    if (numThreads <= 0) {
        numThreads = omp_get_max_threads();
    }

    int maxDepth = 1;
    if (numThreads >= 8) {
        maxDepth = 2;
    }

    Matrix resultMatrix(matrixA.rows());

    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp single
        {
            resultMatrix = multiplyStrassenOpenMPRecursive(matrixA, matrixB, threshold, 0, maxDepth);
        }
    }

    return resultMatrix;
}
