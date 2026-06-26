"""Load SMHR data from Behroozi+2013 and generate baseline fits."""
import numpy as np
import os, pickle, warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser("~/general-conversation/projects/p8-smhr/data/bwc2013/release-sfh_z0_z8_052913/smmr")
CACHE = "/tmp/smhr_cache"
FB = 0.157  # cosmic baryon fraction Ωb/Ωm

def load_smhr():
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, "smhr_data.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    redshifts = [0.10, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00]
    all_x, all_y, all_z, all_yerr_lo, all_yerr_hi = [], [], [], [], []

    for z in redshifts:
        fname = f"c_smmr_z{z:.2f}_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat"
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  Missing: {fname}")
            continue

        data = np.loadtxt(path)
        logMh = data[:, 0]
        log_ratio = data[:, 1]  # log(M*/Mh)
        yerr_lo = data[:, 2]
        yerr_hi = data[:, 3]
        log_eps = log_ratio + np.log10(1/FB)  # log(ε) = log(M*/Mh / fb)

        all_x.append(logMh)
        all_y.append(log_eps)
        all_z.append(np.full_like(logMh, z))
        all_yerr_lo.append(yerr_lo)
        all_yerr_hi.append(yerr_hi)

    result = {
        "logMh": np.concatenate(all_x),
        "log_eps": np.concatenate(all_y),
        "z": np.concatenate(all_z),
        "yerr_lo": np.concatenate(all_yerr_lo),
        "yerr_hi": np.concatenate(all_yerr_hi),
        "n": sum(len(x) for x in all_x),
        "redshifts": redshifts,
    }

    print(f"Loaded SMHR: {result['n']} points, {len(redshifts)} redshifts")
    print(f"  logMh: [{result['logMh'].min():.2f}, {result['logMh'].max():.2f}]")
    print(f"  log_eps: [{result['log_eps'].min():.3f}, {result['log_eps'].max():.3f}]")

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result

def load_smhr_arrays():
    """Return (x, y, z) arrays for SR."""
    d = load_smhr()
    return d["logMh"], d["log_eps"], d["z"]

def double_power_law(logMh, logM1, logN, beta, gamma):
    """Moster+2013 form: log ε = log(2N / ((Mh/M1)^(-β) + (Mh/M1)^γ))."""
    logMh = np.asarray(logMh)
    term1 = 10 ** (-beta * (logMh - logM1))
    term2 = 10 ** (gamma * (logMh - logM1))
    eps = 2 * 10**logN / (term1 + term2)
    return np.log10(eps)

def fit_double_power_law(logMh, log_eps, z):
    """Fit Moster+2013 double power law to SMHR at given z."""
    from scipy.optimize import minimize

    a = 1 / (1 + z)
    # priors from Moster+2013 Table 1
    p0 = {
        "logM1": 11.590 + 1.195 * (1 - a),
        "logN": np.log10(0.0351) + np.log10(0.0247) * (1 - a),
        "beta": 1.376 - 0.826 * (1 - a),
        "gamma": 0.608 + 0.329 * (1 - a),
    }

    def loss(p):
        pred = double_power_law(logMh, *p)
        return np.mean((log_eps - pred)**2)

    res = minimize(loss, [p0["logM1"], p0["logN"], p0["beta"], p0["gamma"]],
                   method="Nelder-Mead", options={"maxiter": 5000})
    return res.x, np.sqrt(res.fun)

if __name__ == "__main__":
    d = load_smhr()
    logMh, log_eps, z_arr = load_smhr_arrays()
    print(f"\nBaseline double-power-law fits at each z:")
    for z in d["redshifts"]:
        mask = np.abs(z_arr - z) < 0.01
        xz = logMh[mask]
        yz = log_eps[mask]
        params, rms = fit_double_power_law(xz, yz, z)
        logM1, logN, beta, gamma = params
        print(f"  z={z:.2f}: logM1={logM1:.3f}, logN={logN:.4f}, beta={beta:.3f}, gamma={gamma:.3f}, RMS={rms:.4f}")
