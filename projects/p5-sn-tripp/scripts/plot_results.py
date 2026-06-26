#!/usr/bin/env python3
"""Plot SR discovery results vs linear Tripp baseline."""

import numpy as np
import pickle, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = "/tmp/sn_tripp_cache"

def main():
    # Load data and SR results
    with open(os.path.join(CACHE, "tripp_residual.pkl"), "rb") as f:
        data = pickle.load(f)
    
    sr_file = os.path.join(CACHE, "tripp_sr_results.pkl")
    if os.path.exists(sr_file):
        with open(sr_file, "rb") as f:
            sr = pickle.load(f)
    else:
        print("SR results not found — run tripp_sr_remote.py first")
        sr = None

    z = data["z"]; x1 = data["x1"]; c = data["c"]
    resid = data["residual"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("SN Ia Tripp Standardization: Linear vs SR", fontsize=14)

    # 1. Hubble diagram
    ax = axes[0, 0]
    ax.scatter(z, data["mu_shoes"], s=1, alpha=0.5, c='k')
    ax.scatter(z, data["mu_ref"], s=1, alpha=0.3, c='r')
    ax.set_xlabel("z"); ax.set_ylabel("μ")
    ax.set_title("Hubble Diagram (MU_SH0ES)")

    # 2. Residual vs z
    ax = axes[0, 1]
    ax.scatter(z, resid, s=1, alpha=0.5, c='k')
    ax.axhline(0, c='r', ls='--')
    ax.set_xlabel("z"); ax.set_ylabel("δμ")
    ax.set_title(f"Residual (RMS={np.std(resid):.4f})")

    # 3. Residual vs x1
    ax = axes[0, 2]
    ax.scatter(x1, resid, s=1, alpha=0.5, c='k')
    ax.axhline(0, c='r', ls='--')
    # Binned
    x1_bins = np.percentile(x1, [5, 25, 50, 75, 95])
    for lo, hi in zip(x1_bins[:-1], x1_bins[1:]):
        bx = (x1 >= lo) & (x1 < hi)
        ax.errorbar((lo+hi)/2, np.mean(resid[bx]), np.std(resid[bx])/np.sqrt(bx.sum()),
                    fmt='ro', capsize=3)
    ax.set_xlabel("x1 (stretch)"); ax.set_ylabel("δμ")

    # 4. Residual vs c
    ax = axes[1, 0]
    ax.scatter(c, resid, s=1, alpha=0.5, c='k')
    ax.axhline(0, c='r', ls='--')
    c_bins = np.percentile(c, [5, 25, 50, 75, 95])
    for lo, hi in zip(c_bins[:-1], c_bins[1:]):
        bc = (c >= lo) & (c < hi)
        ax.errorbar((lo+hi)/2, np.mean(resid[bc]), np.std(resid[bc])/np.sqrt(bc.sum()),
                    fmt='ro', capsize=3)
    ax.set_xlabel("c (color)"); ax.set_ylabel("δμ")

    # 5. SR model prediction (if available)
    ax = axes[1, 1]
    if sr and len(sr["run1"]) > 0:
        best_eq = sr["run1"][0]["equation"]
        best_loss = sr["run1"][0]["loss"]
        ax.text(0.1, 0.5, f"Best SR model (x1,c):\n{best_eq}\n\nLoss: {best_loss:.6f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    else:
        ax.text(0.3, 0.5, "SR results pending", transform=ax.transAxes, fontsize=12)
    ax.set_title("SR Discovery")
    ax.axis('off')

    # 6. Legend / summary
    ax = axes[1, 2]
    text = f"Pantheon+ (1624 SNe)\n"
    text += f"Linear Tripp RMS: {data['rms']:.4f}\n"
    text += f"RMS after linear refit: {data.get('rms_linear', 0):.4f}\n"
    text += f"α (residual): {0:.4f}\nβ (residual): {0:.4f}"
    if sr:
        text += f"\n\nSR Run 1 best C={sr['run1'][0]['complexity']}"
        text += f"\nSR Run 2 best C={sr['run2'][0]['complexity']}"
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace')
    ax.axis('off')

    plt.tight_layout()
    outfile = os.path.join(os.path.dirname(__file__), "..", "tripp_results.png")
    fig.savefig(outfile, dpi=150)
    print(f"Saved to {outfile}")

if __name__ == "__main__":
    main()
