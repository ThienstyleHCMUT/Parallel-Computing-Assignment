#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh - sweeps algo x backend x size x procs x threads x leaf and
# lets bin/bench_<backend> append rows to the CSV (format of include/csvLogger.h).
# Run from the project root:   bash scripts/run_benchmark.sh
#
# Examples
#   Laptop:  MACHINE=laptop-A BACKENDS="seq omp" SIZES="100 500 1000 2000" \
#            THREADS="1 2 4 8" bash scripts/run_benchmark.sh
#   Cluster: MACHINE=cluster BACKENDS="mpi hybrid" SIZES="1000 4000 8000" \
#            PROCS="1 2 4 8" HYBRID_PROCS="2 4" HYBRID_THREADS="2 4 8" \
#            MPIRUN="mpirun --hostfile hosts" bash scripts/run_benchmark.sh
# =============================================================================
set -u
BIN_DIR=${BIN_DIR:-bin}
CSV=${CSV:-results/results.csv}
MACHINE=${MACHINE:-$(hostname)}
ALGOS=${ALGOS:-"naive strassen"}
BACKENDS=${BACKENDS:-"seq omp mpi hybrid"}
SIZES=${SIZES:-"100 500 1000 2000 4000"}
TRIALS=${TRIALS:-3}
LEAVES=${LEAVES:-"128"}                  # Strassen leaf sizes to sweep, e.g. "32 64 128 256"
THREADS=${THREADS:-"1 2 4 8"}            # omp:    threads
PROCS=${PROCS:-"1 2 4 8"}                # mpi:    processes (1 thread each)
HYBRID_PROCS=${HYBRID_PROCS:-"2 4"}      # hybrid: processes
HYBRID_THREADS=${HYBRID_THREADS:-"2 4"}  # hybrid: threads / process
MPIRUN=${MPIRUN:-mpirun}
MPI_EXPORT_FLAG=${MPI_EXPORT_FLAG:--x OMP_NUM_THREADS}   # OpenMPI. MPICH/Intel MPI: MPI_EXPORT_FLAG=""
MPI_EXTRA=${MPI_EXTRA:-}                 # e.g. "--oversubscribe"
TIMEOUT=${TIMEOUT:-1800}                 # seconds per configuration (all trials)
SKIP_SEQ_ABOVE=${SKIP_SEQ_ABOVE:-4000}   # sequential at N=10000 takes hours

mkdir -p "$(dirname "$CSV")"
EXTRA=""; [[ "${NO_VERIFY:-0}" == "1" ]] && EXTRA="--no-verify"

run_cfg () {   # algo backend n procs threads leaf
  local algo=$1 be=$2 n=$3 p=$4 t=$5 leaf=$6 exe="$BIN_DIR/bench_$2" cmd
  [[ -x "$exe" ]] || { echo "  [skip] $exe not built" >&2; return; }
  export OMP_NUM_THREADS=$t
  local args="--algo $algo --n $n --trials $TRIALS --leaf $leaf --machine $MACHINE --csv $CSV $EXTRA"
  case $be in
    seq|omp)    cmd="$exe $args" ;;
    mpi|hybrid) cmd="$MPIRUN -np $p $MPI_EXPORT_FLAG $MPI_EXTRA $exe $args" ;;
  esac
  timeout "$TIMEOUT" $cmd || echo "  [fail/timeout] $algo $be n=$n p=$p t=$t leaf=$leaf" >&2
}

for algo in $ALGOS; do
  leaves="0"; [[ $algo == strassen ]] && leaves="$LEAVES"
  for n in $SIZES; do
    for leaf in $leaves; do
      for be in $BACKENDS; do
        case $be in
          seq)    (( n <= SKIP_SEQ_ABOVE )) && run_cfg $algo seq $n 1 1 $leaf ;;
          omp)    for t in $THREADS; do run_cfg $algo omp $n 1 $t $leaf; done ;;
          mpi)    for p in $PROCS;   do run_cfg $algo mpi $n $p 1 $leaf; done ;;
          hybrid) for p in $HYBRID_PROCS; do for t in $HYBRID_THREADS; do
                    run_cfg $algo hybrid $n $p $t $leaf; done; done ;;
        esac
      done
    done
  done
done
echo "Done -> $CSV"
