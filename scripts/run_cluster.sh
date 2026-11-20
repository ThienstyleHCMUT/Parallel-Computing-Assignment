#!/bin/bash

echo "=== 1. Building MPI Test Target ==="
make test_mpi

if [ $? -eq 0 ]; then
    echo "=== 2. Running MPI Sanity Test ==="
    # Chạy thử 4 tiến trình trên Cluster
    mpirun -np 4 ./test_mpi
else
    echo "[ERROR] Compilation failed!"
fi
