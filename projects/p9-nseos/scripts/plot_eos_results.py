#!/usr/bin/env python3
"""Analyze and plot EOS SR results."""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/nseos_cache/eos_sr_results.pkl", "rb") as f:
    r = pickle.load(f)

print(f"N={r['n']}, Total polytrope RMS={r['total_rms_poly']:.4f}")
print(f"EOSs: {r['eos_names']}")
print()

print("=== Baseline Polytropes ===")
for name, b in r["baselines"].items():
    print(f"  {name}: gamma={b['gamma']:.4f}, RMS={b['rms']:.4f} ({b['n']} pts)")
print()

print("=== SR Run 1: DDBm ===")
poly_rms = r["baselines"]["DDBm"]["rms"]
for m in r["run1_ddbm"]:
    drms = m["rms"] - poly_rms
    drms_str = f"{drms:+.4f}" if abs(drms) > 1e-6 else "-"
    print(f"  C={m['complexity']:2d} RMS={m['rms']:.4f} delta={drms_str}  {m['equation']}")

print()
print("=== SR Run 2: Joint (all EOSs) ===")
total_poly = r["total_rms_poly"]
for m in r["run2_joint"]:
    drms = m["rms"] - total_poly
    drms_str = f"{drms:+.4f}" if abs(drms) > 1e-6 else "-"
    print(f"  C={m['complexity']:2d} RMS={m['rms']:.4f} delta={drms_str}  {m['equation']}")

# === Plot ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: DDBm fit comparison
from eos_data import load_eos
d = load_eos()

nb_m = d["DDBm"]["nb"]
P_m = d["DDBm"]["P"]
log_nb = np.log10(nb_m)
log_P = np.log10(P_m)
axes[0].plot(log_nb, log_P, "k-", lw=2, label="DDBm (true)")

# Polytrope fit
poly_logK = r["baselines"]["DDBm"]["logK"]
poly_gamma = r["baselines"]["DDBm"]["gamma"]
axes[0].plot(log_nb, poly_logK + poly_gamma * log_nb, "r--", lw=1.5, label=f"Polytrope (gamma={poly_gamma:.3f})")

# SR best fit (just show legend, no model)
axes[0].plot([], [], "b--", lw=1, label=f"SR best (C={r['run1_ddbm'][-1]['complexity']})")

axes[0].set_xlabel("log10(nb [fm^-3])")
axes[0].set_ylabel("log10(P [MeV/fm^3])")
axes[0].legend(fontsize=8)
axes[0].set_title("DDBm EOS: Polytrope vs SR")
axes[0].grid(True, alpha=0.3)

# Right: All EOSs with polytrope fits
colors = ["C0", "C1", "C2", "C3", "C4"]
all_nb = []
all_logP = []
for i, name in enumerate(r["eos_names"]):
    eos = d[name]
    nb = eos["nb"]
    P = eos["P"]
    log_nb_i = np.log10(nb)
    log_P_i = np.log10(P)
    axes[1].plot(log_nb_i, log_P_i, "-", color=colors[i], lw=1.5, label=name)
    # Polytrope
    b = r["baselines"][name]
    axes[1].plot(log_nb_i, b["logK"] + b["gamma"] * log_nb_i, "--", color=colors[i], lw=1, alpha=0.5)

axes[1].set_xlabel("log10(nb [fm^-3])")
axes[1].set_ylabel("log10(P [MeV/fm^3])")
axes[1].legend(fontsize=8)
axes[1].set_title("All DDH EOSs (solid=true, dashed=polytrope)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("/tmp/nseos_cache/eos_sr_plot.png", dpi=150)
print(f"\nPlot saved.")
