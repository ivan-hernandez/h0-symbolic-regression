#!/usr/bin/env python3
"""
Ringdown null test with symbolic regression.
Fit δω/ω and δτ/τ as functions of (χ_f, M_f, z) using simple models.
Complements PySR by testing explicit hypotheses.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys, os

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "ringdown_data.csv")

def load_data():
    df = pd.read_csv(DATA_FILE)
    df["domega_err"] = (df["domega_hi"] - df["domega_lo"]) / 2
    df["dtau_err"] = (df["dtau_hi"] - df["dtau_lo"]) / 2
    return df

def chi2(y, y_pred, y_err):
    return np.sum(((y - y_pred) / y_err) ** 2)

def fit_model(X, y, y_err, model_func, p0):
    """Fit a model y = f(X; params), minimize chi^2."""
    def negloglike(p):
        pred = model_func(X, p)
        return 0.5 * chi2(y, pred, y_err)

    result = minimize(negloglike, p0, method="Nelder-Mead")
    best = result.x
    pred = model_func(X, best)
    chi2_val = chi2(y, pred, y_err)
    dof = len(y) - len(best)
    return best, chi2_val, dof

def models():
    """Define model functions and their parameter labels."""
    return {
        "Null (δ=0)": (
            lambda X, p: np.zeros(len(X)),
            [],
        ),
        "Const": (
            lambda X, p: p[0] * np.ones(len(X)),
            ["c"],
        ),
        "χ_f": (
            lambda X, p: p[0] * X[:, 0],
            ["a_χ"],
        ),
        "M_f": (
            lambda X, p: p[0] * X[:, 1],
            ["a_M"],
        ),
        "Const + χ_f": (
            lambda X, p: p[0] + p[1] * X[:, 0],
            ["c", "a_χ"],
        ),
        "Const + M_f": (
            lambda X, p: p[0] + p[1] * X[:, 1],
            ["c", "a_M"],
        ),
        "χ_f + M_f": (
            lambda X, p: p[0] * X[:, 0] + p[1] * X[:, 1],
            ["a_χ", "a_M"],
        ),
        "Const + χ_f + M_f": (
            lambda X, p: p[0] + p[1] * X[:, 0] + p[2] * X[:, 1],
            ["c", "a_χ", "a_M"],
        ),
        "χ_f²": (
            lambda X, p: p[0] * X[:, 0]**2,
            ["a_χ²"],
        ),
        "log M_f": (
            lambda X, p: p[0] * np.log(X[:, 1]),
            ["a_logM"],
        ),
    }

def run_test(df, target="domega", exclude_outliers=False):
    df_use = df.copy()
    if exclude_outliers:
        df_use = df_use[df_use["domega_med"].abs() < 1.0].copy()

    X_names = ["chif", "Mfz"]
    X = df_use[X_names].values
    y = df_use[f"{target}_med"].values
    y_err = df_use[f"{target}_err"].values
    events = df_use["event"].values
    n = len(events)

    print(f"\n{'='*70}")
    print(f"Null test: {target}")
    print(f"N={n} events: {list(events)}")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Params':<20} {'χ²':<10} {'dof':<6} {'χ²/dof':<10} {'Δχ²':<8} {'p(GR)':<8}")
    print("-"*70)

    # Reference: Null model (δ=0)
    chi2_null = chi2(y, np.zeros(n), y_err)

    results = []
    for name, (func, param_names) in models().items():
        if name == "Null (δ=0)":
            best_p = []
            chi2_val = chi2_null
            dof = n
        else:
            p0 = np.zeros(len(param_names))
            best_p, chi2_val, dof = fit_model(X, y, y_err, func, p0)

        delta_chi2 = chi2_val - chi2_null
        # p-value from chi2 distribution (GR is δ=0)
        from scipy.stats import chi2 as chi2_dist
        p_gr = 1.0 - chi2_dist.cdf(chi2_null - chi2_val, dof - n)  # improvement over null
        
        # Format params
        if len(best_p) == 0:
            param_str = "—"
        else:
            param_str = ", ".join([f"{pn}={pv:.4f}" for pn, pv in zip(param_names, best_p)])

        print(f"{name:<20} {param_str:<20} {chi2_val:<10.2f} {dof:<6} {chi2_val/dof:<10.2f} {delta_chi2:<+8.2f}")
        results.append((name, best_p, chi2_val, dof, delta_chi2))

    # Best model
    best = min(results, key=lambda r: r[2])
    print(f"\nBest model: {best[0]} (χ²={best[2]:.2f}, dof={best[3]})")

    # If no model improves significantly over null, report null result
    chi2_models = [r[2] for r in results if r[0] != "Null (δ=0)"]
    min_chi2 = min(chi2_models)
    if min_chi2 >= chi2_null:
        print(f"→ No model beats null (δ=0). GR consistent.")
    else:
        print(f"→ Best model χ²={min_chi2:.2f} vs null χ²={chi2_null:.2f} (Δχ²={min_chi2-chi2_null:.2f})")

    return results

def main():
    df = load_data()

    print("Ringdown Null Test")
    print("=" * 70)
    print(f"Data: {len(df)} events")
    print(df[["event", "domega_med", "domega_lo", "domega_hi", "dtau_med", "chif", "Mfz"]].to_string(index=False))

    # Full sample
    run_test(df, target="domega", exclude_outliers=False)
    run_test(df, target="dtau", exclude_outliers=False)

    # Exclude outliers (S191109d, S200208q)
    print("\n" + "█"*70)
    print("  EXCLUDING OUTLIERS (S191109d, S200208q)")
    print("█"*70)
    run_test(df, target="domega", exclude_outliers=True)
    run_test(df, target="dtau", exclude_outliers=True)

    # Also run combined hierarchical p-value
    print("\n" + "="*70)
    print("Combined hierarchical analysis (from data release)")
    print("="*70)
    import pickle
    comb_dir = "/tmp/rin_test/release_data_products/rin/pseob/combined_samples"
    for fname in ["domega_220_comb.dat.gz", "dtau_220_comb.dat.gz"]:
        fp = os.path.join(comb_dir, fname)
        if os.path.exists(fp):
            samples = np.loadtxt(fp)
            med = np.median(samples)
            lo = np.percentile(samples, 16)
            hi = np.percentile(samples, 84)
            p_zero = np.mean(samples > 0) if med > 0 else np.mean(samples < 0)
            sigma = (hi - lo) / 2
            print(f"  {fname}: {med:.4f} ± {sigma:.4f} (68% CL), {p_zero:.1%} posterior > 0")

if __name__ == "__main__":
    main()
