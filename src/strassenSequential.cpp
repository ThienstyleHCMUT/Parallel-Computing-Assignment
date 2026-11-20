#include "matrix.h"

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

Matrix multiplyStrassenRecursive(const Matrix& matrixA, const Matrix& matrixB, size_t threshold) {
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

        Matrix paddedResult = multiplyStrassenRecursive(paddedA, paddedB, threshold);

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

    Matrix m1 = multiplyStrassenRecursive(addMatrices(a11, a22), addMatrices(b11, b22), threshold);
    Matrix m2 = multiplyStrassenRecursive(addMatrices(a21, a22), b11, threshold);
    Matrix m3 = multiplyStrassenRecursive(a11, subtractMatrices(b12, b22), threshold);
    Matrix m4 = multiplyStrassenRecursive(a22, subtractMatrices(b21, b11), threshold);
    Matrix m5 = multiplyStrassenRecursive(addMatrices(a11, a12), b22, threshold);
    Matrix m6 = multiplyStrassenRecursive(subtractMatrices(a21, a11), addMatrices(b11, b12), threshold);
    Matrix m7 = multiplyStrassenRecursive(subtractMatrices(a12, a22), addMatrices(b21, b22), threshold);

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

Matrix multiplyStrassen(const Matrix& matrixA, const Matrix& matrixB, size_t threshold) {
    return multiplyStrassenRecursive(matrixA, matrixB, threshold);
}
