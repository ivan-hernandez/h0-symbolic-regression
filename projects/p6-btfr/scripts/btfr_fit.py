#!/usr/bin/env python3
"""
Baryonic Tully-Fisher Relation analysis.
Data: SPARC catalog (Lelli+2016, AJ 152, 157).
Columns: Galaxy, T, D, e_D, f_D, Inc, e_Inc, L[3.6], e_L[3.6],
         Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, e_Vflat, Q, Ref
"""

import numpy as np
import urllib.request, pickle, os, sys, warnings
warnings.filterwarnings('ignore')

CACHE = "/tmp/btfr_cache"
os.makedirs(CACHE, exist_ok=True)

DATA_URL = ("https://content.cld.iop.org/journals/1538-3881/"
    "152/6/157/revision1/ajaa3207t1_mrt.txt")

def load_sparc():
    """Download and parse SPARC galaxy sample."""
    cache_file = os.path.join(CACHE, "sparc.pkl")
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))

    print("Downloading SPARC data...")
    raw = urllib.request.urlopen(DATA_URL, timeout=20).read().decode()
    lines = raw.strip().split('\n')

    # Find data start after the last --- separator
    # The last --- line before data is at line with '-------' separator
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('---'):
            data_start = i
    # Data starts after the final separator, ends before the notes
    # Notes start with "Note (" or empty lines after data
    galaxies = []
    for line in lines[data_start+1:]:
        if not line.strip():
            continue
        first = line[:11].strip()
        # Notes and references start with patterns that aren't galaxy names
        if first.startswith('Note'):
            break
        # Valid galaxy names: start with uppercase letter or digit
        if not first[0].isupper() and not first[0].isdigit():
            continue
        if len(first) < 2:
            continue
        # Reference lines are at the end and start with a ref code like "Ba05"
        if len(first) <= 5 and first[0].isalpha() and first[-1].isdigit():
            # Could be a reference code like Ba05, BC04, etc.
            # Skip if the 2nd char is lower case (most refs: Ba05, Be87, etc.)
            if len(first) >= 2 and first[1].islower():
                break
        galaxies.append(line)

    print(f"  Parsed {len(galaxies)} galaxies")

    # Parse fixed-width columns — atomic per galaxy
    records = []
    for line in galaxies:
        err = None
        try:
            name = line[0:11].strip()
            T = int(line[12:14])
            D = float(line[15:21])
            e_D = float(line[22:27])
            f_D = int(line[28:30])
            Inc = float(line[31:35])
            e_Inc = float(line[36:40])
            L3_6 = float(line[41:48])
            e_L3_6 = float(line[49:56])
            Reff = float(line[57:62])
            SBeff = float(line[63:71])
            Rdisk = float(line[72:77])
            SBdisk = float(line[78:86])
            MHI_str = line[87:94].strip()
            MHI = float(MHI_str) if MHI_str else 0.0
            RHI_str = line[95:100].strip()
            RHI = float(RHI_str) if RHI_str else 0.0
            Vflat = float(line[101:106])
            if Vflat <= 0:
                raise ValueError(f"Vflat={Vflat}")
            e_Vflat = float(line[107:112])
            Q_str = line[113:116].strip()
            Q = int(Q_str) if Q_str else 3
        except Exception as e:
            err = str(e)

        if err is None:
            records.append(dict(Galaxy=name, T=T, D=D, e_D=e_D, Inc=Inc,
                                e_Inc=e_Inc, L3_6=L3_6, e_L3_6=e_L3_6,
                                Reff=Reff, SBeff=SBeff, Rdisk=Rdisk,
                                SBdisk=SBdisk, MHI=MHI, RHI=RHI,
                                Vflat=Vflat, e_Vflat=e_Vflat, Q=Q))
        else:
            print(f"  SKIP {name or '?'}: {err}")

    data = {k: np.array([r[k] for r in records]) for k in records[0].keys()}

    pickle.dump(data, open(cache_file, "wb"))
    return data

def main():
    d = load_sparc()
    n = len(d['Galaxy'])
    print(f"Galaxies: {n}")

    # === Compute baryonic mass ===
    # Mstars = Υ[3.6] × L[3.6], standard Υ = 0.5 M⊙/L⊙
    # Mgas = 1.33 × MHI (helium correction)
    # Mbary = Mstars + Mgas
    upsilon = 0.5  # standard M/L at 3.6um
    Mstars = upsilon * d['L3_6'] * 1e9  # convert to Msun
    Mgas = 1.33 * d['MHI'] * 1e9        # convert to Msun
    Mbary = Mstars + Mgas               # Msun

    Vflat = d['Vflat']
    e_Vflat = d['e_Vflat']

    # === Quality cuts ===
    # Q=1 (high), Q=2 (medium), Q=3 (low)
    # Standard approach: use Q=1 and Q=2
    good = d['Q'] <= 2

    logMb = np.log10(Mbary[good])
    logV = np.log10(Vflat[good])
    e_logV = e_Vflat[good] / (Vflat[good] * np.log(10))

    print(f"\nGood galaxies (Q<=2): {good.sum()}")
    print(f"  log Mbary range: [{logMb.min():.2f}, {logMb.max():.2f}]")
    print(f"  log Vflat range: [{logV.min():.2f}, {logV.max():.2f}]")

    # === Baseline BTFR: log Mbary = a·log Vflat + b ===
    # Orthogonal distance regression (errors in both x and y)
    # For simplicity: standard least squares on y
    A = np.column_stack([np.ones_like(logV), logV])
    coeff, *_ = np.linalg.lstsq(A, logMb, rcond=None)
    b, a = coeff  # log Mb = a*log V + b

    pred = A @ coeff
    resid = logMb - pred
    rms = np.sqrt(np.mean(resid**2))

    print(f"\nBaseline BTFR (log-log linear):")
    print(f"  log Mbary = {a:.3f}·log Vflat + {b:.3f}")
    print(f"  Mbary ∝ Vflat^{a:.2f}")
    print(f"  RMS = {rms:.4f} dex")
    print(f"  Predicted by MOND: a = 4.0")
    print(f"  Predicted by ΛCDM: a ≈ 3.0-3.5")
    if abs(a - 4) < 0.3:
        print(f"  → Consistent with MOND")
    elif a < 3.8:
        print(f"  → Steeper than MOND prediction")
    
    # === MOND prediction ===
    a0 = 1.2e-10  # m/s^2
    # MOND: Vflat^4 = a0 * G * Mbary
    # log Mbary = 4*log Vflat - log(a0*G)
    G = 4.3009e-6  # kpc (km/s)^2 / Msun
    logV_mond = np.linspace(logV.min(), logV.max(), 100)
    logMb_mond = 4*logV_mond - np.log10(a0 * G * 3.086e16/1e3)
    # Note: a0 in m/s^2, G in kpc (km/s)^2 / Msun, need unit conversion
    # Vflat^4 = a0 * G * Mbary
    # log(Mbary) = 4*log(Vflat) - log(a0*G)
    # a0*G = 1.2e-10 * 4.3009e-6 * (3.086e19/1e3) ≈ 1.2e-10 * 4.3e-6 * 3.086e16 ≈ 1.59e-2
    a0G = a0 * G * (3.086e19 / 1e3)  # convert a0 from m/s^2 to km/s/kpc?
    # Actually: 1 m/s^2 = 3.086e16 km/s/kpc (roughly)
    # Vflat^4 = a0_G * G * Mbary, where a0_G = a0 in galactic units
    # a0 = 1.2e-10 m/s^2 = 1.2e-10 * 3.086e16 km/s/kpc ≈ 3.7e-6 (km/s)^2/kpc
    a0_gal = 1.2e-10 * 3.086e16  # ≈ 3.7e-6 (km/s)^2/kpc (actually km/s/kpc / s)
    # Hmm, the units are tricky. Let me compute MOND prediction directly.
    # Vflat^4 = a0 * G * Mbary (with a0 in galactic units)
    # a0 = 1.2e-10 m/s^2 = 3700 (km/s)^2/kpc   (1 m/s^2 = 3.086e16 km s^-2 kpc^-1)
    # Actually: acceleration in galactic units is km/s/Gyr, etc.
    # The standard MOND prediction: Mbary = Vflat^4 / (a0 * G)
    # where a0 = 1.2e-10 m/s^2 and G = 4.3e-6 kpc (km/s)^2 / Msun
    # a0 in kpc (km/s)^2 / (kpc^2) ... let me just use: 
    # a0_G = a0 * (1 m / 3.086e19 kpc) * (1 s^2 / ... )
    # This is getting complicated. Let me just report the expected slope.
    print(f"\nMOND prediction: slope = 4.0")
    print(f"  Best-fit slope: a = {a:.3f}")
    
    # === Save for SR ===
    data = {
        "galaxy": d['Galaxy'][good],
        "logMb": logMb, "logV": logV,
        "Mbary": Mbary[good], "Vflat": Vflat[good],
        "Mstars": Mstars[good], "Mgas": Mgas[good],
        "L3_6": d['L3_6'][good],
        "T": d['T'][good], "D": d['D'][good],
        "Inc": d['Inc'][good], "Q": d['Q'][good],
        "slope": float(a), "intercept": float(b),
        "rms": float(rms),
    }
    outfile = os.path.join(CACHE, "btfr_data.pkl")
    with open(outfile, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to {outfile}")

    # === Quick sensitivity tests ===
    print(f"\nSensitivity tests:")
    for ups in [0.3, 0.5, 0.7]:
        Mstars_t = ups * d['L3_6'] * 1e9
        Mbary_t = Mstars_t + Mgas
        logMb_t = np.log10(Mbary_t[good])
        coeff_t, *_ = np.linalg.lstsq(A, logMb_t, rcond=None)
        print(f"  Υ={ups}: a={coeff_t[1]:.3f}")

    for q_cut in [1, 2, 3]:
        g = d['Q'] <= q_cut
        if g.sum() < 3: continue
        logMb_q = np.log10(Mbary[g])
        logV_q = np.log10(Vflat[g])
        Aq = np.column_stack([np.ones_like(logV_q), logV_q])
        coeff_q, *_ = np.linalg.lstsq(Aq, logMb_q, rcond=None)
        print(f"  Q<={q_cut} ({g.sum()}): a={coeff_q[1]:.3f}")

if __name__ == "__main__":
    main()
