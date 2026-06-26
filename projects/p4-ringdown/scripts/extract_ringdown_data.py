#!/usr/bin/env python3
"""
Extract per-event ringdown deviation posteriors and binary parameters
for the symbolic regression null test.

Data sources:
  - pSEOBNR domega_220, dtau_220 posterior samples (from DCC tar.gz)
  - pSEOBNR parameters.pkl (per-event binary params)
  - pSEOB table (Tab. XIII, hardcoded for M_f, chi_f)
"""

import numpy as np
import pickle, json, os, sys

RIN_BASE = "/tmp/rin_test/release_data_products/rin/pseob"

EVENTS = [
    "GW150914", "GW170104",
    "S190519bj", "S190521r", "S190630ag", "S190910s",
    "S191109d", "S200129m", "S200208q", "S200224ca", "S200311bg",
]

# Map S-names to GW-names from pseob_table_lvc_events.tex
GW_NAMES = {
    "S190519bj": "GW190519A",
    "S190521r":  "GW190521B",
    "S190630ag": "GW190630A",
    "S190910s":  "GW190910A",
    "S191109d":  "GW191109A",
    "S200129m":  "GW200129A",
    "S200208q":  "GW200208A",
    "S200224ca": "GW200224A",
    "S200311bg": "GW200311B",
}

# Per-event parameters from pseob_table_lvc_events.tex
# (1+z)M_f [M_sun], chi_f  — median +- 90% CI bounds (symmetricized)
EVENT_PARAMS = {
    "GW150914": {"Mfz": 71.6, "Mfz_lo": 11.0, "Mfz_hi": 8.6, "chif": 0.76, "chif_lo": 0.20, "chif_hi": 0.10},
    "GW170104": {"Mfz": 69.4, "Mfz_lo": 28.1, "Mfz_hi": 13.6, "chif": 0.84, "chif_lo": 0.57, "chif_hi": 0.12},
    "GW190519A": {"Mfz": 155.5, "Mfz_lo": 29.9, "Mfz_hi": 24.0, "chif": 0.81, "chif_lo": 0.28, "chif_hi": 0.10},
    "GW190521B": {"Mfz": 86.4, "Mfz_lo": 14.3, "Mfz_hi": 12.2, "chif": 0.73, "chif_lo": 0.26, "chif_hi": 0.12},
    "GW190630A": {"Mfz": 65.7, "Mfz_lo": 39.2, "Mfz_hi": 18.3, "chif": 0.62, "chif_lo": 0.62, "chif_hi": 0.26},
    "GW190728A": {"Mfz": 83.1, "Mfz_lo": 18.2, "Mfz_hi": 11.1, "chif": 0.89, "chif_lo": 0.25, "chif_hi": 0.06},
    "GW190910A": {"Mfz": 123.5, "Mfz_lo": 18.1, "Mfz_hi": 14.7, "chif": 0.90, "chif_lo": 0.11, "chif_hi": 0.05},
    "GW191109A": {"Mfz": 170.4, "Mfz_lo": 15.1, "Mfz_hi": 25.3, "chif": 0.94, "chif_lo": 0.04, "chif_hi": 0.02},
    "GW200129A": {"Mfz": 74.2, "Mfz_lo": 10.0, "Mfz_hi": 7.4, "chif": 0.76, "chif_lo": 0.22, "chif_hi": 0.10},
    "GW200208A": {"Mfz": 71.5, "Mfz_lo": 11.1, "Mfz_hi": 23.8, "chif": 1.00, "chif_lo": 0.45, "chif_hi": 0.00},
    "GW200224A": {"Mfz": 101.6, "Mfz_lo": 14.0, "Mfz_hi": 10.4, "chif": 0.85, "chif_lo": 0.16, "chif_hi": 0.07},
    "GW200311B": {"Mfz": 75.3, "Mfz_lo": 17.4, "Mfz_hi": 12.4, "chif": 0.76, "chif_lo": 0.39, "chif_hi": 0.13},
}

def get_gw_name(event):
    return GW_NAMES.get(event, event)

def extract_posterior(event):
    """Extract domega and dtau posterior samples for an event."""
    base = os.path.join(RIN_BASE, event)
    domega_file = os.path.join(base, f"rin_{event}_pseobnrv4hm_domega_220.dat.gz")
    dtau_file = os.path.join(base, f"rin_{event}_pseobnrv4hm_dtau_220.dat.gz")
    freq_gr_file = os.path.join(base, f"rin_{event}_pseobnrv4hm_freq_220_modGR.dat.gz")
    tau_gr_file = os.path.join(base, f"rin_{event}_pseobnrv4hm_tau_220_modGR.dat.gz")

    if not all(os.path.exists(f) for f in [domega_file, dtau_file, freq_gr_file, tau_gr_file]):
        print(f"  WARNING: missing files for {event}", file=sys.stderr)
        return None

    domega = np.loadtxt(domega_file)
    dtau = np.loadtxt(dtau_file)
    freq_gr = np.loadtxt(freq_gr_file)
    tau_gr = np.loadtxt(tau_gr_file)

    return {
        "domega": domega,
        "dtau": dtau,
        "freq_gr": freq_gr,
        "tau_gr": tau_gr,
    }

def summarize(arr):
    """Compute median and 68% CL (16th/84th percentiles)."""
    return {
        "median": float(np.median(arr)),
        "lo16": float(np.percentile(arr, 16)),
        "hi84": float(np.percentile(arr, 84)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }

def main():
    # Load parameters.pkl
    pkl_path = os.path.join(RIN_BASE, "parameters.pkl")
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)

    rows = []
    for event in EVENTS:
        print(f"Processing {event}...")
        gw_name = get_gw_name(event)
        if gw_name not in EVENT_PARAMS:
            print(f"  SKIP: no params for {gw_name}")
            continue

        data = extract_posterior(event)
        if data is None:
            continue

        domega_sum = summarize(data["domega"])
        dtau_sum = summarize(data["dtau"])
        freq_gr_sum = summarize(data["freq_gr"])
        tau_gr_sum = summarize(data["tau_gr"])
        ep = EVENT_PARAMS[gw_name]

        # Add redshift from parameters.pkl if available
        redshift = params.get("redshift", {}).get(gw_name, np.nan)
        
        rows.append({
            "event": event,
            "gw_name": gw_name,
            "domega_med": domega_sum["median"],
            "domega_lo": domega_sum["lo16"],
            "domega_hi": domega_sum["hi84"],
            "domega_mean": domega_sum["mean"],
            "domega_std": domega_sum["std"],
            "dtau_med": dtau_sum["median"],
            "dtau_lo": dtau_sum["lo16"],
            "dtau_hi": dtau_sum["hi84"],
            "dtau_mean": dtau_sum["mean"],
            "dtau_std": dtau_sum["std"],
            "freq_gr_med": freq_gr_sum["median"],
            "tau_gr_med": tau_gr_sum["median"],
            "Mfz": ep["Mfz"],
            "Mfz_lo": ep["Mfz_lo"],
            "Mfz_hi": ep["Mfz_hi"],
            "chif": ep["chif"],
            "chif_lo": ep["chif_lo"],
            "chif_hi": ep["chif_hi"],
            "redshift": redshift,
        })

    # Print summary
    print("\n" + "="*100)
    print(f"{'Event':<12} {'domega':>10} {'domega_lo':>10} {'domega_hi':>10} "
          f"{'dtau':>10} {'dtau_lo':>10} {'dtau_hi':>10} "
          f"{'chif':>6} {'Mfz':>6} {'z':>5}")
    print("-"*100)
    for r in rows:
        d_sigma = (r["domega_hi"] - r["domega_lo"]) / 2
        print(f"{r['event']:<12} {r['domega_med']:>10.4f} {r['domega_lo']:>10.4f} {r['domega_hi']:>10.4f} "
              f"{r['dtau_med']:>10.4f} {r['dtau_lo']:>10.4f} {r['dtau_hi']:>10.4f} "
              f"{r['chif']:>6.2f} {r['Mfz']:>6.1f} {r['redshift']:>5.2f}")

    # Save CSV
    save_path = os.path.join(os.path.dirname(__file__), "..", "ringdown_data.csv")
    import csv
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {save_path}")

    # Also save combined hierarchical results
    print("\n=== Combined hierarchical results ===")
    comb_dir = os.path.join(RIN_BASE, "combined_samples")
    for fname in ["domega_220_comb.dat.gz", "domega_220_comb_pseobO3a.dat.gz",
                   "dtau_220_comb.dat.gz", "dtau_220_comb_pseobO3a.dat.gz"]:
        fp = os.path.join(comb_dir, fname)
        if os.path.exists(fp):
            samples = np.loadtxt(fp)
            s = summarize(samples)
            print(f"{fname}: median={s['median']:.4f} 68%=[{s['lo16']:.4f},{s['hi84']:.4f}]")

if __name__ == "__main__":
    main()
