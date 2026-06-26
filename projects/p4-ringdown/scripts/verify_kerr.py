#!/usr/bin/env python3
"""
Compare pSEOBNR GR predictions to exact Kerr QNM frequencies.
Test whether the dtau bias is explained by NR fitting formula errors.
"""

import numpy as np
import qnm, os, csv, sys

RIN_BASE = "/tmp/rin_test/release_data_products/rin/pseob"

# Event data from the pSEOB table (Tab. XIII)
# Mfz = (1+z)M_f [Msun], chif = final spin
EVENTS = {
    "GW150914": {"Mfz": 71.6, "chif": 0.76},
    "GW170104": {"Mfz": 69.4, "chif": 0.84},
    "S190519bj": {"Mfz": 155.5, "chif": 0.81, "z": 0.49},
    "S190521r": {"Mfz": 86.4, "chif": 0.73, "z": 0.25},
    "S190630ag": {"Mfz": 65.7, "chif": 0.62, "z": 0.19},
    "S190910s": {"Mfz": 123.5, "chif": 0.90, "z": 0.29},
    "S191109d": {"Mfz": 170.4, "chif": 0.94},
    "S200129m": {"Mfz": 74.2, "chif": 0.76},
    "S200208q": {"Mfz": 71.5, "chif": 1.00},
    "S200224ca": {"Mfz": 101.6, "chif": 0.85},
    "S200311bg": {"Mfz": 75.3, "chif": 0.76},
}

# Redshifts from parameters.pkl or literature
REDSHIFTS = {
    "GW150914": 0.09,
    "GW170104": 0.19,
    "S191109d": 0.60,
    "S200129m": 0.07,
    "S200208q": 0.12,
    "S200224ca": 0.14,
    "S200311bg": 0.30,
}

M_SUN_S = 4.925490947641267e-06

def kerr_frequency(chi, M_source):
    """Compute Kerr QNM frequency for s=-2,l=2,m=2,n=0 in Hz."""
    grav_220 = qnm.modes_cache(s=-2, l=2, m=2, n=0)
    try:
        omega, A, C = grav_220(a=min(chi, 0.999))
        freq = omega.real / (2 * np.pi * M_source * M_SUN_S)
        return freq, omega.real
    except Exception as e:
        return None, None

def main():
    grav_220 = qnm.modes_cache(s=-2, l=2, m=2, n=0)

    print(f"{'Event':<12} {'χ_f':<6} {'M_f':<8} {'f_Kerr':<10} {'f_GR(pSEOB)':<12} {'Δf/f_Kerr':<10} {'dω/ω':<10} {'Match?':<8}")
    print("-" * 80)

    for event, ep in EVENTS.items():
        z = ep.get("z", REDSHIFTS.get(event, 0.1))
        Mfz = ep["Mfz"]
        chi = ep["chif"]
        M_source = Mfz / (1 + z)

        # Kerr prediction
        try:
            omega, A, C = grav_220(a=min(chi, 0.999))
            f_kerr = omega.real / (2 * np.pi * M_source * M_SUN_S)
        except Exception as e:
            print(f"{event:<12} {chi:<6.2f} {M_source:<8.1f} ERR: {e}")
            continue

        # pSEOBNR GR prediction from the freq_220_modGR samples
        f_pseob_file = os.path.join(RIN_BASE, event, f"rin_{event}_pseobnrv4hm_freq_220_modGR.dat.gz")
        if os.path.exists(f_pseob_file):
            f_pseob_samples = np.loadtxt(f_pseob_file)
            f_pseob = np.median(f_pseob_samples)
            f_pseob_lo = np.percentile(f_pseob_samples, 16)
            f_pseob_hi = np.percentile(f_pseob_samples, 84)
        else:
            print(f"{event:<12} {chi:<6.2f} {M_source:<8.1f} No pSEOB file")
            continue

        # pSEOBNR domega
        domega_file = os.path.join(RIN_BASE, event, f"rin_{event}_pseobnrv4hm_domega_220.dat.gz")
        domega_samples = np.loadtxt(domega_file)
        domega = np.median(domega_samples)

        # Fractional difference between Kerr and pSEOB GR prediction
        delta_f = (f_pseob - f_kerr) / f_kerr

        match = "✓" if abs(delta_f) < 0.03 else ("~" if abs(delta_f) < 0.10 else "✗")
        print(f"{event:<12} {chi:<6.2f} {M_source:<8.1f} {f_kerr:<10.1f} "
              f"{f_pseob:<12.1f} {delta_f:<+10.3f} {domega:<+10.4f} {match:<8}")

    # Now test: is domega correlated with (f_pseob - f_kerr)/f_kerr?
    print("\n\nCorrelation: domega vs NR-fit error")
    print("If domega ≈ -(f_pseob - f_kerr)/f_kerr, then deviations are explained by NR fit error.")
    print(f"{'Event':<12} {'domega':<10} {'Δf/f':<10} {'Ratio':<10}")
    print("-" * 45)

    domega_vals = []
    delta_f_vals = []
    for event, ep in EVENTS.items():
        z = ep.get("z", REDSHIFTS.get(event, 0.1))
        M_source = ep["Mfz"] / (1 + z)
        chi = ep["chif"]

        try:
            omega, A, C = grav_220(a=min(chi, 0.999))
            f_kerr = omega.real / (2 * np.pi * M_source * M_SUN_S)
        except:
            continue

        f_pseob_file = os.path.join(RIN_BASE, event, f"rin_{event}_pseobnrv4hm_freq_220_modGR.dat.gz")
        if not os.path.exists(f_pseob_file):
            continue

        f_pseob = np.median(np.loadtxt(f_pseob_file))
        domega = np.median(np.loadtxt(
            os.path.join(RIN_BASE, event, f"rin_{event}_pseobnrv4hm_domega_220.dat.gz")))

        delta_f = (f_pseob - f_kerr) / f_kerr
        ratio = domega / (-delta_f) if abs(delta_f) > 0.001 else float('inf')

        domega_vals.append(domega)
        delta_f_vals.append(delta_f)
        print(f"{event:<12} {domega:<+10.4f} {delta_f:<+10.3f} {ratio:<10.2f}")

    # Correlation coefficient
    if len(domega_vals) > 2:
        corr = np.corrcoef(domega_vals, delta_f_vals)[0, 1]
        print(f"\nPearson r = {corr:.3f}")
        if abs(corr) > 0.5:
            print("→ Moderate-strong correlation: domega may be partly driven by NR fit error")
        else:
            print("→ Weak correlation: domega is NOT driven by NR fit error")

if __name__ == "__main__":
    main()
