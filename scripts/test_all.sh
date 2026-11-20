#!/usr/bin/env bash
# Usage: bash scripts/test_all.sh [N] [THREADS] [LEAF]
set -e
cd "$(dirname "$0")/.."

N=${1:-500}
T=${2:-$(nproc)}
LEAF=${3:-128}
MACHINE=${MACHINE:-$(hostname)}
CSV=results/trial_test.csv
FLAGS="-O3 -Wall -Wextra -std=c++17 -Iinclude"
SRC="src/matrix.cpp src/algorithms.cpp"

echo "=== [1/3] Build ==="
make clean >/dev/null
make all
g++ $FLAGS -Wno-unknown-pragmas $SRC src/test_strassen.cpp -o bin/test_strassen_seq
g++ $FLAGS -fopenmp $SRC src/test_strassen.cpp -o bin/test_strassen_omp

echo "=== [2/3] Correctness test ==="
./bin/test_strassen_seq
./bin/test_strassen_omp

echo "=== [3/3] Trial benchmark (N=$N threads=$T leaf=$LEAF) ==="
rm -f "$CSV"
for algo in naive strassen; do
  ./bin/bench_seq --algo $algo --n "$N" --leaf "$LEAF" --trials 1 --machine "$MACHINE" --csv "$CSV"
  OMP_NUM_THREADS=$T ./bin/bench_omp --algo $algo --n "$N" --leaf "$LEAF" --trials 1 --machine "$MACHINE" --csv "$CSV"
done

echo "=== Results ($CSV) ==="
column -t -s, "$CSV"
grep -q FAILED "$CSV" && { echo "FAILED rows found"; exit 1; }
echo "ALL OK"
