#!/usr/bin/env python3
"""
Run PySR on ringdown deviations data on the remote machine.
Usage: python3 remote_run.py
"""

import numpy as np
import pandas as pd
import sys, os, json, pickle, csv

DATA_FILE = os.path.join(os.path.dirname(__file__), "ringdown_data.csv")

def load_data():
    df = pd.read_csv(DATA_FILE)
    df["domega_err"] = (df["domega_hi"] - df["domega_lo"]) / 2
    df["dtau_err"] = (df["dtau_hi"] - df["dtau_lo"]) / 2
    df["chif_err"] = (df["chif_hi"] + df["chif_lo"]) / 2
    df["Mfz_err"] = (df["Mfz_hi"] + df["Mfz_lo"]) / 2
    return df

def run_sr(df, target="domega", exclude_outliers=False, seed=42):
    """Run PySR on domega or dtau."""
    df_use = df.dropna(subset=["redshift"]).copy()
    if exclude_outliers:
        df_use = df_use[df_use["domega_med"].abs() < 1.0].copy()

    X_names = ["chif", "Mfz", "redshift"]
    X = df_use[X_names].values
    y = df_use[f"{target}_med"].values
    y_err = df_use[f"{target}_err"].values
    events = df_use["event"].values

    n = len(events)
    print(f"\n=== PySR: {target}, {n} events, seed={seed} ===")
    print(f"Events: {list(events)}")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")

    import pysr

    model = pysr.PySRRegressor(
        niterations=200,
        populations=20,
        population_size=50,
        maxsize=20,
        parsimony=0.01,
        model_selection="accuracy",
        loss="L2",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "square"],
        batching=False,
        warm_start=False,
        random_state=seed,
        verbosity=0,
        progress=True,
    )

    weights = 1.0 / y_err
    # Normalize weights
    weights = weights / weights.mean()
    model.fit(X, y, weights=weights)

    results = []
    for _, eq in model.equations_.iterrows():
        if eq["score"] > 0:
            results.append({
                "complexity": int(eq["complexity"]),
                "loss": float(eq["loss"]),
                "score": float(eq["score"]),
                "equation": eq["equation"],
            })
            print(f"  CPX {eq['complexity']:2d} | loss={eq['loss']:8.4f} | score={eq['score']:6.3f} | {eq['equation']}")

    return {
        "target": target,
        "n_events": n,
        "exclude_outliers": exclude_outliers,
        "seed": seed,
        "results": results,
        "best_equation": results[0]["equation"] if results else None,
    }

def main():
    df = load_data()

    print("=" * 70)
    print("Ringdown Null Test — Symbolic Regression")
    print("=" * 70)
    print(f"\nData: {len(df)} events")
    print(df[["event", "domega_med", "domega_lo", "domega_hi", "dtau_med", "chif", "Mfz"]].to_string(index=False))

    all_results = []

    for target in ["domega", "dtau"]:
        for exclude in [False, True]:
            for seed in [42, 123, 456]:
                try:
                    r = run_sr(df, target=target, exclude_outliers=exclude, seed=seed)
                    all_results.append(r)
                except Exception as e:
                    print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SR Summary")
    print("=" * 70)
    for r in all_results:
        label = f"{r['target']}_n{r['n_events']}_s{r['seed']}"
        best = r.get("best_equation", "None")
        print(f"  {label}: {best}")

    # Save results
    out_file = os.path.join(os.path.dirname(__file__), "..", "sr_results.json")
    json.dump = lambda obj, **kw: json.JSONEncoder().encode(obj)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
