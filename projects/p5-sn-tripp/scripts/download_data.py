#!/usr/bin/env python3
"""Download Pantheon+ and DES-SN5YR data with raw lightcurve params."""

import numpy as np
import urllib.request, os, gzip, pickle

CACHE = "/tmp/sn_tripp_cache"
os.makedirs(CACHE, exist_ok=True)

# Pantheon+ data (raw lightcurve params)
PANTHEON_URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
    "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES.dat")
PANTHEON_COV_URL = ("https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES_STAT%2BSYS.cov")

# DES-SN5YR data
DES_URL = ("https://raw.githubusercontent.com/des-science/DES-SN5YR/"
    "main/DES-SN5YR_public_full_cov/DES-SN5YR_dataset.dat")

def download_pantheon():
    """Download and parse Pantheon+ data with lightcurve params.
    
    Columns: 0=CID, 1=IDSURVEY, 2=zHD, 3=zHDERR, 4=zCMB, 5=zCMBERR,
    6=zHEL, 7=zHELERR, 8=m_b_corr, 9=m_b_corr_err_DIAG,
    10=MU_SH0ES, 11=MU_SH0ES_ERR_DIAG,
    12=CEPH_DIST, 13=IS_CALIBRATOR, 14=USED_IN_SH0ES_HF,
    15=c, 16=cERR, 17=x1, 18=x1ERR, ...
    """
    cache_file = os.path.join(CACHE, "pantheon_raw.pkl")
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))

    print("Downloading Pantheon+...")
    raw = np.loadtxt(urllib.request.urlopen(PANTHEON_URL), dtype=str,
                     skiprows=1, delimiter=None)
    print(f"  {len(raw)} entries")

    z = raw[:, 2].astype(float)
    m_b = raw[:, 8].astype(float)
    m_b_err = raw[:, 9].astype(float)
    mu_shoes = raw[:, 10].astype(float)
    is_cal = raw[:, 13].astype(int)
    c_val = raw[:, 15].astype(float)
    c_err = raw[:, 16].astype(float)
    x1 = raw[:, 17].astype(float)
    x1_err = raw[:, 18].astype(float)

    data = {
        "z": z, "m_b": m_b, "m_b_err": m_b_err,
        "mu_shoes": mu_shoes, "is_cal": is_cal,
        "c": c_val, "c_err": c_err,
        "x1": x1, "x1_err": x1_err,
    }
    print(f"  Cosmological SNe: {(is_cal == 0).sum()}")
    pickle.dump(data, open(cache_file, "wb"))
    return data

def download_des():
    """Download DES-SN5YR dataset."""
    cache_file = os.path.join(CACHE, "des_raw.pkl")
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))

    print("Downloading DES-SN5YR...")
    raw = np.loadtxt(urllib.request.urlopen(DES_URL), dtype=str,
                     skiprows=1, delimiter=None)
    print(f"  {len(raw)} entries")
    
    z = raw[:, 0].astype(float)
    m_b = raw[:, 1].astype(float)
    x1 = raw[:, 2].astype(float)
    c = raw[:, 3].astype(float)
    
    data = {"z": z, "m_b": m_b, "x1": x1, "c": c}
    pickle.dump(data, open(cache_file, "wb"))
    return data

def main():
    pp = download_pantheon()
    print(f"Pantheon+: {pp['z'].shape[0]} SNe, "
          f"z=[{pp['z'].min():.3f},{pp['z'].max():.2f}]")
    try:
        des = download_des()
        print(f"DES-SN5YR: {des['z'].shape[0]} SNe, "
              f"z=[{des['z'].min():.3f},{des['z'].max():.2f}]")
    except Exception as e:
        print(f"DES-SN5YR download failed (non-critical): {e}")
    
    # Quick check: linear Tripp on Pantheon+ cosmology SNe
    mask = pp["is_cal"] == 0
    z = pp["z"][mask]
    m_b = pp["m_b"][mask]
    x1 = pp["x1"][mask]
    c_val = pp["c"][mask]

    from scipy.integrate import quad
    H0_ref, Om_ref = 70.0, 0.3
    c_light = 299792.458
    def mu_lcdm(z_arr):
        Ok = 1 - Om_ref
        dH = c_light / H0_ref
        result = []
        for zi in z_arr:
            def E(zp): return np.sqrt(Om_ref*(1+zp)**3 + Ok*(1+zp)**2 + (1-Om_ref-Ok))
            dc = dH * quad(E, 0, zi)[0]
            if Ok > 0:
                dm = dH / np.sqrt(Ok) * np.sinh(np.sqrt(Ok) * dc / dH)
            elif Ok < 0:
                dm = dH / np.sqrt(-Ok) * np.sin(np.sqrt(-Ok) * dc / dH)
            else:
                dm = dc
            result.append(5 * np.log10(dm * (1 + zi)) + 25)
        return np.array(result)

    mu_ref = mu_lcdm(z)
    y = m_b - mu_ref

    A = np.column_stack([np.ones_like(x1), x1, c_val])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coeff
    rms = np.sqrt(np.mean((y - pred)**2))
    print(f"\nBaseline linear Tripp (reference ΛCDM):")
    print(f"  M_B = {coeff[0]:.3f}")
    print(f"  α (x1 coeff) = {-coeff[1]:.4f}")
    print(f"  β (c coeff)  = {coeff[2]:.4f}")
    print(f"  RMS = {rms:.4f} mag")
    print(f"  N = {len(z)}")

    ref_data = {
        "z": z, "m_b": m_b, "x1": x1, "c": c_val,
        "mu_ref": mu_ref, "y": y,
        "coeff_linear": coeff,
        "rms_linear": rms,
    }
    pickle.dump(ref_data, open(os.path.join(CACHE, "pantheon_ref.pkl"), "wb"))
    print("  Saved to cache: pantheon_ref.pkl")

if __name__ == "__main__":
    main()
