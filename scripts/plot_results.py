#!/usr/bin/env python3
"""Reads CSV file(s) written by bin/bench_* and scripts/bench_lib.py, writes charts + summary.csv.
  python3 scripts/plot_results.py results/*.csv --out charts
Needs: pip install pandas matplotlib numpy"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--out", default="charts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

df = pd.concat([pd.read_csv(f) for f in args.csv], ignore_index=True)
df = df.rename(columns={"MatrixSize": "N", "ExecutionTime_s": "Time"})
bad = df[df.Status != "PASSED"]
if len(bad):
    print(f"WARNING: {len(bad)} FAILED rows excluded:")
    print(bad.groupby(["Machine", "Algorithm", "Backend", "N"]).size())
df = df[df.Status == "PASSED"].copy()
df["Workers"] = df.Threads * df.Processes

grp = ["Machine", "Algorithm", "Backend", "N", "Threads", "Processes", "Workers", "LeafSize"]
agg = df.groupby(grp).agg(Time=("Time", "median"), Std=("Time", "std"),
                          GFLOPS=("GFLOPS", "median"), Trials=("Time", "count")).reset_index()

# for Strassen keep the best leaf size per configuration
g2 = [c for c in grp if c != "LeafSize"]
cfg = agg.loc[agg.groupby(g2)["Time"].idxmin()].reset_index(drop=True)

# baseline = sequential run of the same algorithm; fallback = 1-worker omp / mpi run
def baseline(r):
    s = cfg[(cfg.Machine == r.Machine) & (cfg.Algorithm == r.Algorithm) & (cfg.N == r.N)]
    for be in ("seq", "omp", "mpi"):
        b = s[(s.Backend == be) & (s.Workers == 1)]
        if len(b): return b.Time.iloc[0]
    return np.nan
cfg["Speedup"] = cfg.apply(lambda r: baseline(r) / r.Time, axis=1)
cfg["Efficiency"] = cfg.Speedup / cfg.Workers
agg.to_csv(os.path.join(args.out, "summary_all.csv"), index=False)
cfg.to_csv(os.path.join(args.out, "summary_best_leaf.csv"), index=False)
print("summaries ->", args.out)

MARK = {"seq": "o", "omp": "s", "mpi": "^", "hybrid": "D", "blas": "x"}
def style(a): return "--" if a == "strassen" else (":" if a == "library" else "-")
def save(fig, name):
    fig.tight_layout(); p = os.path.join(args.out, name); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

for m, dm in cfg.groupby("Machine"):
    # 1. time + GFLOPS vs N (best config per algorithm/backend)
    best = dm.loc[dm.groupby(["Algorithm", "Backend", "N"])["Time"].idxmin()]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for (a, b), g in best.groupby(["Algorithm", "Backend"]):
        g = g.sort_values("N")
        for ax, col in zip(axs, ("Time", "GFLOPS")):
            ax.plot(g.N, g[col], marker=MARK.get(b, "o"), ls=style(a), label=f"{a}-{b}")
    axs[0].set(xscale="log", yscale="log", xlabel="Matrix size N", ylabel="Time (s)",
               title=f"Execution time, best config [{m}]")
    axs[1].set(xscale="log", yscale="log", xlabel="Matrix size N", ylabel="GFLOPS (2N^3/t)",
               title=f"Throughput [{m}]")
    for ax in axs: ax.grid(True, which="both", alpha=.3)
    axs[0].legend(fontsize=8, ncol=2)
    save(fig, f"time_gflops_vs_N_{m}.png")

    par = dm[dm.Backend.isin(["omp", "mpi", "hybrid"])]
    if par.empty: continue
    nmax = par.N.max()

    # 2/3. speedup & efficiency vs workers at largest N
    for col in ("Speedup", "Efficiency"):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (a, b), g in par[par.N == nmax].groupby(["Algorithm", "Backend"]):
            g = g.sort_values("Workers")
            ax.plot(g.Workers, g[col], marker=MARK[b], ls=style(a), label=f"{a}-{b}")
        w = sorted(par.Workers.unique())
        ax.plot(w, w if col == "Speedup" else [1] * len(w), "k:", label="ideal")
        ax.set(xlabel="Workers (processes x threads)", ylabel=col, xscale="log", title=f"{col} at N={nmax} [{m}]")
        ax.set_xticks(w, [str(x) for x in w]); ax.grid(True, alpha=.3); ax.legend(fontsize=8, ncol=2)
        save(fig, f"{col.lower()}_{m}_N{nmax}.png")

    # 4. speedup vs N at max workers
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (a, b), g in par.groupby(["Algorithm", "Backend"]):
        gm = g[g.Workers == g.Workers.max()].sort_values("N")
        ax.plot(gm.N, gm.Speedup, marker=MARK[b], ls=style(a), label=f"{a}-{b} ({gm.Workers.max()}w)")
    ax.set(xscale="log", xlabel="Matrix size N", ylabel="Speedup", title=f"Speedup vs N, max workers [{m}]")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8, ncol=2)
    save(fig, f"speedup_vs_N_{m}.png")

    # 5. hybrid heatmap procs x threads
    hy = dm[dm.Backend == "hybrid"]
    if len(hy):
        hy = hy[hy.N == hy.N.max()]
        algos = sorted(hy.Algorithm.unique())
        fig, axs = plt.subplots(1, len(algos), figsize=(5.5 * len(algos), 4.5), squeeze=False)
        for ax, a in zip(axs[0], algos):
            pv = hy[hy.Algorithm == a].pivot_table(index="Processes", columns="Threads", values="Time")
            im = ax.imshow(pv.values, cmap="viridis_r", aspect="auto")
            ax.set_xticks(range(len(pv.columns)), pv.columns); ax.set_yticks(range(len(pv.index)), pv.index)
            ax.set(xlabel="OpenMP threads / process", ylabel="MPI processes", title=f"{a} hybrid N={hy.N.iloc[0]} time (s)")
            for (y, x), v in np.ndenumerate(pv.values):
                if v == v: ax.text(x, y, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
            fig.colorbar(im, ax=ax)
        save(fig, f"hybrid_heatmap_{m}.png")

    # 6. versus library
    if (dm.Backend == "blas").any():
        cmp_ = best.pivot_table(index="N", columns=["Algorithm", "Backend"], values="Time")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        cmp_.plot(kind="bar", ax=ax, logy=True)
        ax.set(xlabel="Matrix size N", ylabel="Time (s), best config", title=f"Implementations vs BLAS library [{m}]")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, axis="y", alpha=.3)
        save(fig, f"vs_library_{m}.png")

# 7. Strassen leaf-size sweep (uses all leaf values)
st = agg[(agg.Algorithm == "strassen") & (agg.LeafSize > 0)]
if st.LeafSize.nunique() > 1:
    for m, dm in st.groupby("Machine"):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (b, n), g in dm[dm.Workers == dm.groupby("Backend").Workers.transform("max")].groupby(["Backend", "N"]):
            g = g.sort_values("LeafSize"); ax.plot(g.LeafSize, g.Time, marker="o", label=f"{b} N={n}")
        ax.set(xscale="log", yscale="log", xlabel="Leaf size", ylabel="Time (s)", title=f"Strassen leaf-size sweep [{m}]")
        ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
        save(fig, f"strassen_leaf_{m}.png")

# 8. OpenMP across machines (assignment: >= group-size shared-memory machines)
om = cfg[cfg.Backend == "omp"]
if om.Machine.nunique() > 1:
    nmax = om.N.max(); fig, ax = plt.subplots(figsize=(8, 5.5))
    for (m, a), g in om[om.N == nmax].groupby(["Machine", "Algorithm"]):
        g = g.sort_values("Threads"); ax.plot(g.Threads, g.Time, marker="s", ls=style(a), label=f"{m} {a}")
    ax.set(yscale="log", xlabel="OpenMP threads", ylabel="Time (s)", title=f"OpenMP across machines, N={nmax}")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    save(fig, "omp_machines.png")
