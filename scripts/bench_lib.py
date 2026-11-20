#!/usr/bin/env python3
"""Library baseline (NumPy -> OpenBLAS/MKL), written in the same CSV format as csvLogger.h.
  MACHINE=laptop-A THREADS="1 2 4 8" SIZES="100 1000 4000" python3 scripts/bench_lib.py
Re-launches itself once per thread count (BLAS reads the thread env vars at import)."""
import os, sys, time, subprocess

machine = os.environ.get("MACHINE", os.uname().nodename)
out = os.environ.get("CSV", "results/results.csv")
sizes = [int(x) for x in os.environ.get("SIZES", "100 500 1000 2000 4000").split()]
threads = [int(x) for x in os.environ.get("THREADS", "1 2 4 8").split()]
trials = int(os.environ.get("TRIALS", "3"))

if os.environ.get("_BENCH_CHILD") != "1":
    for t in threads:
        env = dict(os.environ, _BENCH_CHILD="1", _T=str(t), OMP_NUM_THREADS=str(t),
                   OPENBLAS_NUM_THREADS=str(t), MKL_NUM_THREADS=str(t))
        subprocess.run([sys.executable, __file__], env=env, check=True)
    sys.exit(0)

import numpy as np
t_cnt = int(os.environ["_T"])
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
new = not os.path.exists(out) or os.path.getsize(out) == 0
rng = np.random.default_rng(0)
with open(out, "a") as f:
    if new:
        f.write("Machine,Algorithm,Backend,MatrixSize,Threads,Processes,LeafSize,Trial,ExecutionTime_s,GFLOPS,Status\n")
    for n in sizes:
        A, B = rng.uniform(-10, 10, (n, n)), rng.uniform(-10, 10, (n, n))
        for tr in range(1, trials + 1):
            t0 = time.perf_counter(); C = A @ B; dt = time.perf_counter() - t0
            f.write(f"{machine},library,blas,{n},{t_cnt},1,0,{tr},{dt:.9g},{2*n**3/dt/1e9:.6g},PASSED\n")
            print(f"library n={n} threads={t_cnt} trial={tr} {dt:.4f}s")
