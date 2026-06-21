"""Symbolic regression for the Radial Acceleration Relation (RAR).

Reproduces and extends Desmond+2023 (arXiv:2301.04368) using PySR.
Searches for g_obs = F(g_bar) from SPARC data.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from parse_sparc import parse_mass_models, compute_radial_accelerations


def prepare_rar_data(df, min_gbar=1e-13):
    """Prepare RAR data for symbolic regression.

    Filters points with valid gbar > min_gbar, takes log10 of both.
    Returns arrays for fitting: log10(gbar) -> log10(gobs).
    """
    valid = (df["gbar"].values > min_gbar) & (df["gobs"].values > 0)
    log_gbar = np.log10(df["gbar"].values[valid])
    log_gobs = np.log10(df["gobs"].values[valid])
    print(f"Prepared RAR data: {len(log_gbar)} points (filtered from {len(df)})")
    print(f"  log_gbar: [{log_gbar.min():.2f}, {log_gbar.max():.2f}]")
    print(f"  log_gobs: [{log_gobs.min():.2f}, {log_gobs.max():.2f}]")
    return log_gbar.reshape(-1, 1), log_gobs


def run_pysr(X, y, output_dir="output", seed=42, n_cycles=100):
    """Run PySR on RAR data."""
    model = PySRRegressor(
        niterations=n_cycles,
        populations=20,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sqrt", "square", "cube",
            "exp", "log", "log10",
            "tanh", "erf",
        ],
        maxsize=20,
        complexity_of_operators={
            "+": 1, "-": 1, "*": 2, "/": 2,
            "sqrt": 2, "square": 1, "cube": 2,
            "exp": 3, "log": 3, "log10": 3,
            "tanh": 3, "erf": 4,
        },
        parsimony=0.001,
        elementwise_loss="L2DistLoss()",
        seed=seed,
        procs=8,
        multithreading=True,
        model_selection="accuracy",
        tempdir=output_dir,
        extra_julia_deps=["LossFunctions"],
    )
    model.fit(X, y)
    return model


def print_best_models(model, n=5):
    """Print the best equations found."""
    print(f"\n{'='*60}")
    print(f"Best RAR equations from PySR")
    print(f"{'='*60}")
    eqs = model.equations_
    for i in range(min(n, len(eqs))):
        row = eqs.iloc[i]
        print(f"\n  Score: {row.get('score', 'N/A'):.4f}")
        print(f"  Loss:  {row['loss']:.6e}")
        print(f"  Complexity: {row['complexity']}")
        print(f"  Equation: {row['equation']}")
    return eqs


def plot_rar(data_x, data_y, model, output="rar_sr.png"):
    """Plot the data and best SR fit."""
    log_gbar = data_x[:, 0]
    log_gobs = data_y
    gbar = 10 ** log_gbar
    gobs = 10 ** log_gobs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1.scatter(gbar, gobs, s=1, alpha=0.3, color="k", label="SPARC data")
    gbar_grid = np.logspace(-12, -8, 200)
    gbar_grid_log = np.log10(gbar_grid).reshape(-1, 1)
    gobs_pred = 10 ** model.predict(gbar_grid_log)
    ax1.plot(gbar_grid, gobs_pred, "r-", lw=2, label="PySR best")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$g_{\rm bar}$ (m/s$^2$)")
    ax1.set_ylabel(r"$g_{\rm obs}$ (m/s$^2$)")
    ax1.legend()

    # Log scale
    ax2.scatter(log_gbar, log_gobs, s=1, alpha=0.3, color="k")
    log_gbar_grid = np.linspace(-12, -8, 200).reshape(-1, 1)
    log_gobs_pred = model.predict(log_gbar_grid)
    ax2.plot(log_gbar_grid, log_gobs_pred, "r-", lw=2, label="PySR best")
    ax2.set_xlabel(r"$\log_{10} g_{\rm bar}$")
    ax2.set_ylabel(r"$\log_{10} g_{\rm obs}$")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"\nSaved {output}")
    plt.close()


if __name__ == "__main__":
    import sys
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print("=" * 60)
    print("SPARC RAR — Symbolic Regression with PySR")
    print(f"Iterations: {n_cycles}, Seed: {seed}")
    print("=" * 60)

    # 1. Load and prepare data
    df = parse_mass_models()
    acc = compute_radial_accelerations(df)
    X, y = prepare_rar_data(acc)

    # 2. Run PySR
    print(f"\nRunning PySR ({n_cycles} iterations)...")
    output_dir = f"output/rar_sr_s{seed}"
    import os
    os.makedirs(output_dir, exist_ok=True)
    model = run_pysr(X, y, output_dir=output_dir, seed=seed, n_cycles=n_cycles)

    # 3. Show results
    eqs = print_best_models(model)

    # 4. Plot
    plot_rar(X, y, model, output=f"{output_dir}/rar_fit.png")

    # 5. Save equations
    eqs.to_csv(f"{output_dir}/equations.csv", index=False)
    print(f"\nEquations saved to {output_dir}/equations.csv")
