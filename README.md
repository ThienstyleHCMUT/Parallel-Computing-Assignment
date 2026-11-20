# Parallel Computing - Assignment 1

Matrix multiplication (naive + Strassen) with Sequential, OpenMP, MPI and hybrid MPI+OpenMP backends,
plus a benchmark runner and chart generator.

## Directory structure
- `include/`: `matrix.h`, `timer.h`, `csvLogger.h`, `algorithms.h`
- `src/`: `matrix.cpp`, `algorithms.cpp` (naive + Strassen, all backends), `bench_main.cpp` (benchmark driver)
- `test/`: `test_matrix.cpp` (correctness tests)
- `scripts/`: `run_benchmark.sh`, `bench_lib.py` (NumPy/BLAS baseline), `plot_results.py` (charts)
- `results/`: CSV output, `charts/`: PNG output

## Build
```bash
make test        # build + run correctness tests (needs g++ with OpenMP)
make all         # bin/bench_seq, bin/bench_omp
make mpi         # bin/bench_mpi, bin/bench_hybrid (needs mpicxx; on the cluster)
```
One source file, four executables: the backend is chosen by compiler flags
(`-fopenmp` => OpenMP, `-DUSE_MPI` via `mpicxx` => MPI, both => hybrid).

## Benchmark
```bash
# shared-memory machine (run on EACH laptop, use a different MACHINE name)
MACHINE=laptop-A BACKENDS="seq omp" SIZES="100 500 1000 2000 4000 10000" THREADS="1 2 4 8" \
  LEAVES="64 128 256" bash scripts/run_benchmark.sh
MACHINE=laptop-A THREADS="1 2 4 8" python3 scripts/bench_lib.py        # library baseline

# cluster
MACHINE=cluster BACKENDS="mpi hybrid" SIZES="1000 4000 8000 10000" PROCS="1 2 4 8" \
  HYBRID_PROCS="2 4" HYBRID_THREADS="2 4 8" MPIRUN="mpirun --hostfile hosts" \
  bash scripts/run_benchmark.sh
```
Rows are appended to `results/results.csv`:
`Machine,Algorithm,Backend,MatrixSize,Threads,Processes,LeafSize,Trial,ExecutionTime_s,GFLOPS,Status`.
Timing covers the whole multiply including MPI communication; verification (Freivalds) is not timed.

## Charts
```bash
pip install pandas matplotlib numpy
python3 scripts/plot_results.py results/*.csv --out charts
```

## Notes / limitations
- Strassen+MPI distributes the 7 sub-products, so at most 7 ranks do useful work (use hybrid threads beyond that).
- MPI counts are `int`: N must stay below ~46000 (fine for N=10000).
- N=10000 needs several GB of RAM per process (Strassen: ~7 extra h*h buffers + padding to 10112).
- GFLOPS always uses 2N^3, so for Strassen it is an "effective" rate.
