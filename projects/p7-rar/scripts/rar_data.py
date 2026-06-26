"""Load SPARC data and compute gbar/gobs for RAR SR."""
import numpy as np
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/general-conversation/rotation_curves'))
from parse_sparc import parse_mass_models, compute_radial_accelerations

CACHE = "/tmp/rar_cache"
DATA_DIR = os.path.expanduser("~/general-conversation/rotation_curves")

def load_rar(full_cov=True, log_space=True):
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, "rar_data.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    mrt_path = os.path.join(DATA_DIR, "MassModels_Lelli2016c.mrt")
    df = parse_mass_models(mrt_path)
    acc = compute_radial_accelerations(df, Upsilon_disk=0.5, Upsilon_bul=0.7)

    good = acc["gbar"].values > 0
    result = {
        "gbar": acc["gbar"].values[good],
        "gobs": acc["gobs"].values[good],
        "log_gbar": np.log10(acc["gbar"].values[good]),
        "log_gobs": np.log10(acc["gobs"].values[good]),
        "R": acc["R"].values[good],
        "ID": acc["ID"].values[good],
        "D": acc["D"].values[good],
        "N": good.sum(),
        "N_gal": acc["ID"].nunique(),
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Loaded RAR: {result['N']} points, {result['N_gal']} galaxies")
    print(f"  log gbar: [{result['log_gbar'].min():.3f}, {result['log_gbar'].max():.3f}]")
    print(f"  log gobs: [{result['log_gobs'].min():.3f}, {result['log_gobs'].max():.3f}]")
    print(f"  (removed {acc['gbar'].values.size - good.sum()} zero/negative gbar points)")
    return result

def load_rar_log():
    d = load_rar()
    return {
        "x": d["log_gbar"],
        "y": d["log_gobs"],
        "n": len(d["log_gbar"]),
        "desc": "RAR: log gobs = f(log gbar)"
    }

if __name__ == "__main__":
    d = load_rar()
    ld = load_rar_log()
    print(f"\nFeatures: x={ld['desc']}, N={ld['n']}")
