#!/usr/bin/env python3
"""Load compact object mass measurements from all available sources."""

import numpy as np, os

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_gwtc3():
    """GWTC-3 confident events (primary + secondary masses)."""
    f = os.path.join(DATA, 'gwtc3.npz')
    d = np.load(f, allow_pickle=True)
    ev = d['events']
    masses = []
    for e in ev:
        # Primary mass (heavier)
        masses.append((e['m1'], e['m1_low'], e['m1_high'], 'gwtc3_pri', e['name']))
        # Secondary mass (lighter)
        masses.append((e['m2'], e['m2_low'], e['m2_high'], 'gwtc3_sec', e['name']))
    arr = np.array(masses, dtype=[('mass','f8'), ('mlo','f8'), ('mhi','f8'),
                                  ('source','U20'), ('name','U30')])
    return arr

def load_radio_ns():
    """Radio pulsar masses from literature compilation."""
    # Antoniadis+2016, Fonseca+2021, etc.
    # fmt: mass, err_low, err_high, source, name
    raw = [
        # From Antoniadis+2016 (reliable DNS masses)
        (1.338, 0.001, 0.001, 'radio_dns', 'J0737-3039A'),
        (1.250, 0.001, 0.001, 'radio_dns', 'J0737-3039B'),
        (1.365, 0.018, 0.018, 'radio_dns', 'B1534+12'),
        (1.345, 0.003, 0.003, 'radio_dns', 'B1913+16'),
        (1.440, 0.002, 0.002, 'radio_dns', 'B2127+11C'),
        (1.714, 0.006, 0.022, 'radio_dns', 'J0453+1559'),
        (1.366, 0.021, 0.021, 'radio_dns', 'J1756-2251'),
        (1.54, 0.02, 0.02, 'radio_dns', 'J1811-1736'),
        (2.01, 0.04, 0.04, 'radio_dns', 'J0348+0432'),
        (1.53, 0.08, 0.08, 'radio_dns', 'J1946+2052'),
        (1.25, 0.09, 0.09, 'radio_dns', 'J1946+2052c'),
        (1.47, 0.03, 0.03, 'radio_dns', 'J0509+3801'),
        (1.48, 0.06, 0.06, 'radio_dns', 'J0509+3801c'),
        # From Fonseca+2021 (NANOGrav)
        (2.08, 0.07, 0.07, 'radio_nano', 'J0740+6620'),
        (1.928, 0.017, 0.017, 'radio_nano', 'J1614-2230'),
        (1.44, 0.07, 0.07, 'radio_nano', 'B1913+16_ng'),
        # From Özel+2016 review
        (1.37, 0.22, 0.22, 'radio_other', 'Cen X-3'),
        (1.49, 0.08, 0.08, 'radio_other', 'Her X-1'),
        (1.44, 0.12, 0.12, 'radio_other', 'LMC X-4'),
        (1.03, 0.12, 0.12, 'radio_other', 'SMC X-1'),
        (1.25, 0.08, 0.08, 'radio_other', '4U 1538-52'),
        (1.86, 0.19, 0.19, 'radio_other', 'Vela X-1'),
        # X-ray binary dynamical masses
        (1.40, 0.30, 0.30, 'xrb', 'Cen X-3_xrb'),
        (1.50, 0.30, 0.30, 'xrb', 'LMC X-4_xrb'),
    ]
    arr = np.array(raw, dtype=[('mass','f8'), ('mlo','f8'), ('mhi','f8'),
                                ('source','U20'), ('name','U30')])
    return arr

def load_nicer():
    """NICER mass measurements."""
    raw = [
        (1.44, 0.15, 0.15, 'nicer', 'J0030+0451'),
        (2.08, 0.07, 0.07, 'nicer', 'J0740+6620'),
    ]
    return np.array(raw, dtype=[('mass','f8'), ('mlo','f8'), ('mhi','f8'),
                                 ('source','U20'), ('name','U30')])

def load_all():
    gw = load_gwtc3()
    ns = load_radio_ns()
    ni = load_nicer()
    combined = np.concatenate([gw, ns, ni])
    # Sort by mass
    combined = combined[np.argsort(combined['mass'])]
    return combined

if __name__ == '__main__':
    d = load_all()
    print(f"Total objects: {len(d)}")
    for s in set(d['source']):
        sub = d[d['source'] == s]
        print(f"  {s}: {len(sub)} (mass range {sub['mass'].min():.2f}-{sub['mass'].max():.2f})")
