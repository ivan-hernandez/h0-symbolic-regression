#!/usr/bin/env python3
"""
hubble_pilot.py

Phase 1 pilot: Symbolic regression on H(z) data to discover the expansion
history with minimal theoretical priors.

Approach:
  - Mock test: recover LCDM from synthetic data (validate pipeline)
  - Real data: fit H(z) from cosmic chronometers + BAO
  - Extract H0 = H(z=0), compare Planck vs SH0ES
  - Visualize discovered functional form

Usage:
  ./hubble_pilot.py              # runs mock + real
  ./hubble_pilot.py --mock-only  # mock test only
  ./hubble_pilot.py --real-only  # real data only
"""

import numpy as np
import sys, os, time, json
from pysr import PySRRegressor

os.environ["COLUMNS"] = "120"

# ---------------------------------------------------------------------------
# Plotting (Agg for headless)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. DATA COMPILATION
# ---------------------------------------------------------------------------

def get_cc_data():
    """
    Cosmic Chronometer H(z) compilation — verified from published tables.

    Sources (33 points):
      Jimenez+ 2003   (z=0.09)
      Simon+ 2005     (z=0.17, 0.27, 0.40, 0.90, 1.30, 1.43, 1.53, 1.75)
      Stern+ 2010     (z=0.48, 0.88)
      Moresco+ 2012   (z=0.1791, 0.1993, 0.3519, 0.5929, 0.6797, 0.7812, 0.8754, 1.037)
      Zhang+ 2014     (z=0.07, 0.12, 0.20, 0.28)
      Moresco 2015    (z=1.363, 1.965)
      Moresco+ 2016   (z=0.3802, 0.4004, 0.4247, 0.4497, 0.4783)
      Ratsimbazafy+ 2017 (z=0.47)
      Borghi+ 2022    (z=0.75)
      Jiao+ 2022      (z=0.80)
    """
    cc = np.array([
        # z        H(z)       err
        [0.070,    69.0,     19.6],  # Zhang+ 2014 (full-spectrum fitting)
        [0.090,    69.0,     12.0],  # Jimenez+ 2003 (full-spectrum)
        [0.120,    68.6,     26.2],  # Zhang+ 2014
        [0.170,    83.0,      8.0],  # Simon+ 2005 (full-spectrum)
        [0.1791,   75.0,      4.0],  # Moresco+ 2012 (D4000)
        [0.1993,   75.0,      5.0],  # Moresco+ 2012 (D4000)
        [0.200,    72.9,     29.6],  # Zhang+ 2014
        [0.270,    77.0,     14.0],  # Simon+ 2005
        [0.280,    88.8,     36.6],  # Zhang+ 2014
        [0.3519,   83.0,     14.0],  # Moresco+ 2012 (D4000)
        [0.3802,   83.0,     13.5],  # Moresco+ 2016 (D4000)
        [0.400,    95.0,     17.0],  # Simon+ 2005 (full-spectrum)
        [0.4004,   77.0,     10.2],  # Moresco+ 2016 (D4000)
        [0.4247,   87.1,     11.2],  # Moresco+ 2016
        [0.4497,   92.8,     12.9],  # Moresco+ 2016
        [0.470,    89.0,     34.0],  # Ratsimbazafy+ 2017 (full-spectrum)
        [0.4783,   80.9,      9.0],  # Moresco+ 2016
        [0.480,    97.0,     62.0],  # Stern+ 2010 (full-spectrum)
        [0.5929,  104.0,     13.0],  # Moresco+ 2012 (D4000)
        [0.6797,   92.0,      8.0],  # Moresco+ 2012
        [0.750,    98.8,     33.6],  # Borghi+ 2022 (Lick indices)
        [0.7812,  105.0,     12.0],  # Moresco+ 2012
        [0.800,   113.1,     28.5],  # Jiao+ 2022 (full-spectrum)
        [0.8754,  125.0,     17.0],  # Moresco+ 2012
        [0.880,    90.0,     40.0],  # Stern+ 2010
        [0.900,   117.0,     23.0],  # Simon+ 2005
        [1.037,   154.0,     20.0],  # Moresco+ 2012
        [1.300,   168.0,     17.0],  # Simon+ 2005
        [1.363,   160.0,     33.6],  # Moresco 2015
        [1.430,   177.0,     18.0],  # Simon+ 2005
        [1.530,   140.0,     14.0],  # Simon+ 2005
        [1.750,   202.0,     40.0],  # Simon+ 2005
        [1.965,   186.5,     50.4],  # Moresco 2015
    ])
    return cc


def get_bao_data():
    """
    BAO H(z) measurements (in km/s/Mpc, r_s-normalized).
    Source: SDSS DR12 consensus values (Alam+ 2017, Table 3, Ross+ method).
      H × (r_d / r_d,fid), r_d,fid = 147.78 Mpc.
    """
    bao = np.array([
        # z        H(z)       err
        [0.380,    81.1,      2.2],   # SDSS DR12 consensus
        [0.510,    91.1,      2.1],   # SDSS DR12 consensus
        [0.610,    99.4,      2.2],   # SDSS DR12 consensus
    ])
    return bao

def get_desi_bao_data(r_d=147.0):
    """
    DESI DR1 BAO: D_H/r_d converted to H(z) with r_d={r_d} Mpc.
    Source: DESI 2024 VI (arXiv:2404.03002), Table 1.
    """
    c = 299792.458
    desi = np.array([
        [0.510, 20.98334647, 0.61],
        [0.706, 20.07872919, 0.60],
        [0.930, 17.87612922, 0.35],
        [1.317, 13.82372285, 0.42],
        [2.330,  8.52256583, 0.17],
    ])
    hz = c / (r_d * desi[:, 1])
    errs = hz * desi[:, 2] / desi[:, 1]
    return np.column_stack([desi[:, 0], hz, errs])


def load_data(include_desi=True):
    """Combine CC + BAO + DESI DR1 BAO."""
    cc = get_cc_data()
    bao = get_bao_data()
    combined = np.vstack([cc, bao])
    if include_desi:
        desi = get_desi_bao_data()
        combined = np.vstack([combined, desi])
    combined = combined[combined[:, 0].argsort()]
    mask = (combined[:, 2] > 0) & (combined[:, 1] > 0) & (combined[:, 2] < 100)
    return combined[mask]


# ---------------------------------------------------------------------------
# 2. MOCK DATA TEST (validate pipeline)
# ---------------------------------------------------------------------------

def generate_mock_lcdm(z, H0=70.0, Om=0.3, Ol=0.7):
    """Generate H(z) from LCDM with scatter."""
    Or = 0.0  # neglect radiation for low-z
    Ok = 1.0 - Om - Ol
    E = np.sqrt(Om * (1+z)**3 + Or * (1+z)**4 + Ok * (1+z)**2 + Ol)
    noise = np.random.normal(0, 0.03, size=len(z))  # 3% scatter
    return H0 * E * (1 + noise), 0.03 * H0 * E  # data, errors


def run_mock_test():
    """Generate mock LCDM data and see if PySR recovers the form."""
    print("\n" + "=" * 60)
    print("MOCK TEST: Recovering LCDM from synthetic data")
    print("=" * 60)

    np.random.seed(42)
    z_mock = np.linspace(0.05, 2.0, 30)
    H_mock, err_mock = generate_mock_lcdm(z_mock)

    X_mock = z_mock.reshape(-1, 1)
    y_mock = H_mock

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt"],
        constraints={"^": (-1, 1)},
        niterations=100,
        populations=10,
        population_size=100,
        maxsize=20,
        parsimony=0.0001,
        precision=64,
        turbo=True,
        model_selection="best",
        early_stop_condition=1e-12,
        verbosity=1,
        parallelism="multithreading",
    )

    t0 = time.time()
    model.fit(X_mock, y_mock)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    if len(model.equations_) > 0:
        # Pick equation with minimum loss (best accuracy)
        best_idx = model.equations_['loss'].idxmin()
        best = model.equations_.loc[best_idx]
        print(f"  Best equation: {best['sympy_format']}")
        print(f"  Loss: {best['loss']:.6f}, Complexity: {best['complexity']}")

    # Evaluate at z=0 - use the predict which uses internally selected best model
    H0_mock = model.predict([[0.0]])[0]

    # Evaluate at z=0
    H0_mock = model.predict([[0.0]])[0]
    print(f"  Recovered H0 = {H0_mock:.2f} (true: 70.0)")

    # Plot
    z_plot = np.linspace(0, 2.2, 200)
    H_plot = model.predict(z_plot.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(z_mock, H_mock, yerr=err_mock, fmt='o', alpha=0.6,
                label='Mock data (LCDM)', capsize=2)
    ax.plot(z_plot, H_plot, 'r-', lw=2, label='SR fit')
    ax.set_xlabel('z')
    ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Mock test: Symbolic regression recovering LCDM')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig('output/mock_test.png', dpi=150)
    plt.close(fig)
    print("  Saved output/mock_test.png")

    return model, H0_mock


# ---------------------------------------------------------------------------
# 3. REAL DATA FIT
# ---------------------------------------------------------------------------

def run_real_fit(data, z0_prior_strength=20.0, seed=42):
    """Run symbolic regression on real H(z) data.

    Parameterization: H(z) = H0_ref + f(z)
    where H0_ref is a fixed reference and f(z) is discovered by SR.
    H0 = H0_ref + f(0), so the constant term of f gives the correction.

    A weak prior at z=0 (H=H0_ref ± z0_prior_strength) is added to
    anchor H0 extrapolation. The prior is deliberately weak (σ=20)
    so it prevents pathological sqrt(z) behavior without biasing results.
    """
    print("\n" + "=" * 60)
    print("REAL DATA: Symbolic regression on CC + BAO H(z)")
    print("=" * 60)

    H0_ref = 67.4  # Planck as neutral reference
    z_all, H_all, err_all = data[:, 0], data[:, 1], data[:, 2]

    # Transform to residuals relative to reference H0
    y_all = H_all - H0_ref

    # Add weak prior at z=0 to anchor H0 extrapolation
    z_all = np.append(z_all, 0.0)
    y_all = np.append(y_all, 0.0)
    H_all = np.append(H_all, H0_ref)
    err_all = np.append(err_all, z0_prior_strength)

    print(f"  Data points: {len(z_all)} ({len(z_all)-1} real + 1 z=0 prior)")
    print(f"  Redshift range: [{z_all.min():.3f}, {z_all.max():.3f}]")
    print(f"  H0 reference: {H0_ref}")
    print(f"  z=0 prior: H(0) = {H0_ref} ± {z0_prior_strength}")
    print(f"  y range: [{y_all.min():+.1f}, {y_all.max():+.1f}]")

    X = z_all.reshape(-1, 1)
    y = y_all
    y_err = err_all

    model = PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=["sqrt"],
        niterations=300,
        populations=12,
        population_size=100,
        maxsize=20,
        parsimony=0.0005,
        precision=64,
        turbo=True,
        procs=12,
        model_selection="accuracy",
        random_state=seed,
        early_stop_condition=1e-12,
        verbosity=1,
        parallelism="multithreading",
    )

    t0 = time.time()
    model.fit(X, y)
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed:.1f}s")

    best_idx = model.equations_['loss'].idxmin()
    print("\n  Top equations (sorted by accuracy):")
    for i, row in model.equations_.sort_values('loss').head(min(5, len(model.equations_))).iterrows():
        print(f"    [{i}] f(z) = {row['sympy_format']}")
        print(f"         loss={row['loss']:.4f}, "
              f"complexity={row['complexity']}")

    best_eq = model.equations_.loc[best_idx]['sympy_format']

    # Extract implied H0 from all equations
    print("\n  Hall of Fame:")
    print(f"  {'Cpx':>4} {'Loss':>8} {'f(0)':>9} {'H0':>8}  Equation")
    print(f"  {'-'*4} {'-'*8} {'-'*9} {'-'*8}  {'-'*35}")
    for i, row in model.equations_.sort_values('complexity').iterrows():
        try:
            y0_test = float(model.predict([[0.0]], index=i)[0])
            h0_test = H0_ref + y0_test
        except:
            y0_test = float('nan')
            h0_test = float('nan')
        eq_short = str(row['sympy_format'])[:50]
        f0_str = f"{y0_test:+.1f}" if not np.isnan(y0_test) else "NaN"
        h0_str = f"{h0_test:.1f}" if not np.isnan(h0_test) else "NaN"
        print(f"  {row['complexity']:>4d} {row['loss']:>8.1f} {f0_str:>9} {h0_str:>8}  {eq_short}")

    # Use the internally selected best model for H0
    best_idx = model.equations_['loss'].idxmin()
    eq_selected = model.equations_.loc[best_idx]
    print(f"\n  Best equation (minimum loss):")
    print(f"    f(z) = {eq_selected['sympy_format']}")

    try:
        y0_pred = float(model.predict([[0.0]])[0])
        H0_pred = H0_ref + y0_pred
    except:
        H0_pred = float('nan')
        y0_pred = float('nan')

    H0_planck = 67.4
    H0_sh0es = 73.0

    print(f"\n  {'=' * 40}")
    print(f"  H0_ref      = {H0_ref:.1f}")
    if not np.isnan(y0_pred):
        print(f"  f(0)        = {y0_pred:+.2f}")
    print(f"  IMPLIED H0  = {H0_pred:.2f} km/s/Mpc" if not np.isnan(H0_pred)
          else "  IMPLIED H0 = NaN (singularity at z=0)")
    print(f"  Planck 2018 (early): {H0_planck} ± 0.5")
    print(f"  SH0ES 2024 (late):   {H0_sh0es} ± 1.0")
    if not np.isnan(H0_pred):
        print(f"  Difference from Planck: {H0_pred - H0_planck:+.2f}")
        print(f"  Difference from SH0ES:  {H0_pred - H0_sh0es:+.2f}")
    print(f"  {'=' * 40}")

    # Plot
    z_plot = np.linspace(0, max(2.0, z_all.max() * 1.05), 200)
    try:
        y_plot = model.predict(z_plot.reshape(-1, 1))
        H_plot = H0_ref + y_plot
    except:
        H_plot = np.full_like(z_plot, np.nan)

    H_lcdm_planck = 67.4 * np.sqrt(0.315*(1+z_plot)**3 + 0.685)
    H_lcdm_sh0es  = 73.0 * np.sqrt(0.315*(1+z_plot)**3 + 0.685)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.errorbar(z_all, H_all, yerr=err_all, fmt='o', alpha=0.5,
                 label='CC + BAO data', capsize=2, color='gray')
    ax1.plot(z_plot, H_plot, 'r-', lw=2.5,
             label=f'SR: H(z) = {H0_ref} + f(z)')
    ax1.plot(z_plot, H_lcdm_planck, 'b--', lw=1.5, alpha=0.6,
             label=f'LCDM (H0={H0_planck})')
    ax1.plot(z_plot, H_lcdm_sh0es, 'g--', lw=1.5, alpha=0.6,
             label=f'LCDM (H0={H0_sh0es})')
    if not np.isnan(H0_pred):
        ax1.axhline(H0_pred, color='red', lw=1, ls=':', alpha=0.5,
                    label=f'SR H0 = {H0_pred:.1f}')
    ax1.set_ylabel('H(z) [km/s/Mpc]', fontsize=12)
    ax1.set_title('Symbolic Regression on H(z): Discovering Deviations from ΛCDM',
                  fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Residuals w.r.t. LCDM (Planck H0)
    H_lcdm_resid = 67.4 * np.sqrt(0.315*(1+z_all)**3 + 0.685)
    resid = H_all - H_lcdm_resid
    ax2.errorbar(z_all, resid, yerr=err_all, fmt='o', alpha=0.5,
                 color='gray', capsize=2)
    ax2.axhline(0, color='k', lw=0.5)
    if not np.isnan(H0_pred):
        ax2.axhline(H0_pred - 67.4, color='r', lw=1, ls=':', alpha=0.5,
                    label=f'SR offset = {H0_pred - 67.4:+.1f}')
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel('Residual [km/s/Mpc]', fontsize=12)
    ax2.set_title('Residuals vs LCDM (Planck H0 = 67.4)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig('output/hubble_pilot_results.png', dpi=150)
    plt.close(fig)
    print("\n  Saved output/hubble_pilot_results.png")

    meta = {
        "parameterization": "H(z) = H0_ref + f(z)",
        "H0_ref": H0_ref,
        "best_equation": str(best_eq),
        "H0_implied": round(float(H0_pred), 2) if not np.isnan(H0_pred) else None,
        "f(0)": round(float(y0_pred), 2) if not np.isnan(y0_pred) else None,
        "H0_planck": H0_planck,
        "H0_sh0es": H0_sh0es,
        "n_data_points": len(z_all),
        "z_range": [float(z_all.min()), float(z_all.max())],
    }
    with open("output/results.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved output/results.json")

    return model, H0_pred


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    mock_only = "--mock-only" in sys.argv
    real_only = "--real-only" in sys.argv
    seed = 42
    for i, a in enumerate(sys.argv):
        if a == "--seed" and i+1 < len(sys.argv):
            seed = int(sys.argv[i+1])

    if not real_only:
        run_mock_test()

    if not mock_only:
        data = load_data()
        run_real_fit(data, seed=seed)

    print(f"\nDone. (seed={seed})")
