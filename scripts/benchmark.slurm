#!/bin/bash
#SBATCH --job-name=matrix_bench
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "========================================"
echo " MATRIX MULTIPLICATION BENCHMARK"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $(hostname)"
echo " Start: $(date)"
echo "========================================"

module load openmpi/5.0

mkdir -p results

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=FALSE

make bench_seq bench_omp bench_mpi bench_hybrid

CSV="results/benchmark_${SLURM_JOB_ID}.csv"

run_test() {
    local backend="$1"
    local algo="$2"
    local n="$3"
    local procs="$4"
    local threads="$5"

    echo ""
    echo ">>> Backend=$backend Algo=$algo N=$n Procs=$procs Threads=$threads"

    export OMP_NUM_THREADS="$threads"

    case "$backend" in
        seq)
            ./bench_seq \
                --algo "$algo" --n "$n" \
                --trials 3 --machine HPC \
                --csv "$CSV"
            ;;

        omp)
            ./bench_omp \
                --algo "$algo" --n "$n" \
                --trials 3 --machine HPC \
                --csv "$CSV"
            ;;

        mpi)
            mpirun --bind-to none -np "$procs" ./bench_mpi \
                --algo "$algo" --n "$n" \
                --trials 3 --machine HPC \
                --csv "$CSV"
            ;;

        hybrid)
            mpirun --bind-to none -np "$procs" ./bench_hybrid \
                --algo "$algo" --n "$n" \
                --trials 3 --machine HPC \
                --csv "$CSV"
            ;;
    esac
}

N=512

for algo in naive strassen; do
    run_test seq "$algo" "$N" 1 1
    run_test omp "$algo" "$N" 1 4
    run_test mpi "$algo" "$N" 4 1
    run_test hybrid "$algo" "$N" 2 2
done

echo ""
echo "========================================"
echo " Benchmark completed!"
echo " CSV: $CSV"
echo " End: $(date)"
echo "========================================"
