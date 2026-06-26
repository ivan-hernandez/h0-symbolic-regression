"""Load NS EOS tables (P, n_b, eps) from various models and prepare for SR."""
import numpy as np
import os, pickle

DATA_DIR = os.path.expanduser("~/general-conversation/projects/p9-nseos/data")
CACHE = "/tmp/nseos_cache"

# Standard reference: n_sat = 0.16 fm^-3, P_sat ~ 0 (by definition for symmetric matter)
# For beta-equilibrated NS matter, P_sat ≈ few MeV/fm^3

def load_eos():
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, "eos_data.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    eos_data = {}

    # DDH models (clean format: nb[fm^-3], e[MeV/fm^3], p[MeV/fm^3])
    # These are RMF-based EOSs covering different stiffness
    for name in ["DDBl", "DDBm", "DDBu1", "DDBu2", "DDBx"]:
        path = os.path.join(DATA_DIR, f"ddh_{name}.txt")
        if not os.path.exists(path): continue
        data = np.loadtxt(path, comments="#")
        nb = data[:, 0]
        eps = data[:, 1]
        P = data[:, 2]
        eos_data[name] = {"nb": nb, "eps": eps, "P": P, "n": len(nb)}
        print(f"  {name}: {len(nb)} pts, nb=[{nb[0]:.4f},{nb[-1]:.4f}], P=[{P[0]:.3e},{P[-1]:.3e}]")

    print(f"Loaded {len(eos_data)} EOS models")
    with open(cache_path, "wb") as f:
        pickle.dump(eos_data, f)
    return eos_data

def load_for_sr():
    """Return arrays for SR training. Use log-log space."""
    d = load_eos()
    all_nb = []
    all_logP = []
    all_labels = []

    for name, eos in d.items():
        nb = eos["nb"]
        P = eos["P"]
        # Skip near-zero pressure points (numerical issues in log)
        mask = P > 1e-6
        all_nb.append(nb[mask])
        all_logP.append(np.log10(P[mask]))
        all_labels.append(np.full(mask.sum(), name))

    result = {
        "nb": np.concatenate(all_nb),
        "log_nb": np.log10(np.concatenate(all_nb)),
        "logP": np.concatenate(all_logP),
        "label": np.concatenate(all_labels),
        "n": sum(len(x) for x in all_nb),
        "eos_names": list(d.keys()),
    }
    print(f"SR data: {result['n']} points")
    return result

if __name__ == "__main__":
    d = load_eos()
    sr = load_for_sr()
    print(f"Total: {sr['n']} points")
    print(f"  log nb: [{sr['log_nb'].min():.3f}, {sr['log_nb'].max():.3f}]")
    print(f"  log P: [{sr['logP'].min():.3f}, {sr['logP'].max():.3f}]")
