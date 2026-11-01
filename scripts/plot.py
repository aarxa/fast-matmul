"""Plot results.csv (from ./build/bench results.csv) into results.png."""
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
data = defaultdict(list)
with open(path) as f:
    for row in csv.DictReader(f):
        data[row["kernel"]].append((int(row["n"]), float(row["gflops"])))

fig, ax = plt.subplots(figsize=(8, 5))
for kernel, points in data.items():
    ns, gf = zip(*sorted(points))
    style = "--" if kernel == "accelerate" else "-"
    ax.plot(ns, gf, style, marker="o", label=kernel)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Matrix size N (N x N x N)")
ax.set_ylabel("GFLOP/s (log scale)")
ax.set_title("SGEMM on Apple M4 Pro: each optimization step")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results.png", dpi=150)
print("wrote results.png")
