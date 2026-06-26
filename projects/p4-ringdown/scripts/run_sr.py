#!/usr/bin/env python3
"""
Symbolic regression on ringdown deviations.
Target: δω/ω and δτ/τ as functions of (χ_f, M_f, q, z).
"""

import numpy as np
import pandas as pd
import pkgutil, importlib, sys, os, json

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # domega error = half of 68% interval
    df["domega_err"] = (df["domega_hi"] - df["domega_lo"]) / 2
    df["dtau_err"] = (df["dtau_hi"] - df["dtau_lo"]) / 2
    # chif error = asymmetric, use mean
    df["chif_err"] = (df["chif_hi"] + df["chif_lo"]) / 2
    # Mfz error
    df["Mfz_err"] = (df["Mfz_hi"] + df["Mfz_lo"]) / 2
    return df

def run_pysr(df, target="domega", exclude_outliers=False):
    """Run PySR on domega or dtau."""
    if target == "domega":
        y = df["domega_med"].values
        y_err = df["domega_err"].values
    else:
        y = df["dtau_med"].values
        y_err = df["dtau_err"].values

    df_use = df[~df["redshift"].isna()].copy() if exclude_outliers else df.copy()
    if exclude_outliers:
        # Remove extreme outliers
        mask = df_use["domega_med"].abs() < 1.0
        df_use = df_use[mask]

    X_names = ["chif", "Mfz", "redshift"]
    X = df_use[X_names].values
    y = df_use["domega_med"].values if target == "domega" else df_use["dtau_med"].values
    y_err = df_use["domega_err"].values if target == "domega" else df_use["dtau_err"].values
    events = df_use["event"].values
    n_events = len(events)

    print(f"SR input: {len(X)} events")
    print(f"Target: {target}")
    print(f"Features: {X_names}")
    print(f"Events: {list(events)}")

    if n_events < 5:
        print("Too few events, skipping SR")
        return None

    try:
        import pysr
    except ImportError:
        print("PySR not installed locally. Use remote machine.")
        return None

    # Simple chi^2 fitness: minimize sum((y - f(X))^2 / y_err^2)
    # Use PySR with custom loss
    model = pysr.PySRRegressor(
        niterations=100,
        populations=15,
        population_size=50,
        maxsize=15,
        parsimony=0.01,
        model_selection="accuracy",
        loss="L2",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "square"],
        batching=False,
        warm_start=False,
        random_state=42,
        verbosity=0,
        progress=False,
    )

    # Weight samples by 1/error
    weights = 1.0 / y_err
    model.fit(X, y, weights=weights)

    print(f"\n--- PySR results for {target} ---")
    print(f"Complexity | Loss | Equation")
    print("-" * 60)
    for eq in model.equations_:
        if eq["score"] > 0:
            print(f"CPX {eq['complexity']:2d} | {eq['loss']:8.4f} | {eq['equation']}")

    return model

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "ringdown_data.csv")
    df = load_data(csv_path)

    print("=== Ringdown Data Summary ===")
    print(df[["event", "domega_med", "domega_lo", "domega_hi",
              "dtau_med", "dtau_lo", "dtau_hi",
              "chif", "Mfz", "redshift"]].to_string(index=False))

    print("\n" + "="*60)
    print("Combined hierarchical analysis:")
    print(f"  Combined domega: from CSV above")
    print(f"  Combined dtau: from CSV above")

    print("\nHow many events have domega inconsistent with zero at 68% CL?")
    signif = df[(df["domega_lo"] > 0) | (df["domega_hi"] < 0)]
    print(f"  {len(signif)}/11 events: {list(signif['event'])}")

    print("\nHow many events have dtau inconsistent with zero at 68% CL?")
    signif_t = df[(df["dtau_lo"] > 0) | (df["dtau_hi"] < 0)]
    print(f"  {len(signif_t)}/11 events: {list(signif_t['event'])}")

    # Check if PySR is available
    try:
        import pysr
        print("\nPySR available locally, running SR...")
        run_pysr(df, target="domega")
        print()
        run_pysr(df, target="dtau")
    except ImportError:
        print("\nPySR not available locally. Data prepared for remote run.")
        print("SCP to remote and run there.")

if __name__ == "__main__":
    main()
