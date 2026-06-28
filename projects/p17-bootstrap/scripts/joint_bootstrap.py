"""Joint bootstrap: CC+BAO+DESI+SNe with Cpx 13 fixed form.

Resamples all data jointly, refits Cpx 13 parameters,
computes H0 distribution. 200 iterations.
"""
import csv, os, sys, math, random
import numpy as np
from scipy import stats, optimize

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import data as D

random.seed(42)
np.random.seed(42)

N_BOOT = 200
C = 299792.458  # km/s

# ---------- Integration ----------
def mu_from_H(H_func, z):
    n = 500
    h = z / (2*n)
    xs = [i*h for i in range(2*n+1)]
    fx = [1.0/H_func(x) for x in xs]
    integral = h/3 * (fx[0] + fx[-1] + 4*sum(fx[1::2]) + 2*sum(fx[2:-1:2]))
    Dc = C * integral
    return 5.0 * math.log10(max((1+z)*Dc, 1e-10)) + 25.0

# ---------- Load data ----------
print("Loading data...")
z_cc, h_cc, e_cc = D.fetch_cc()
z_b, h_b, e_b = D.fetch_sdss()
z_d, h_d, e_d = D.fetch_desi_dr2()
z_s, mu_s, e_s = D.fetch_pantheon()

print(f"CC: {len(z_cc)}, SDSS BAO: {len(z_b)}, DESI DR2: {len(z_d)}, Pantheon+: {len(z_s)}")

# ---------- Model ----------
def H_model(z, H0, A, B, C):
    return H0 + A * z * (z - B) * (z*z + C)

def chi2_H(params, z, h, e):
    H0, A, B, Cp = params
    pred = np.array([H_model(zi, H0, A, B, Cp) for zi in z])
    return np.sum(((h - pred) / e)**2)

def chi2_SN(params, z, mu, err):
    H0, A, B, Cp = params
    pred = np.array([mu_from_H(lambda zi: H_model(zi, H0, A, B, Cp), zi) for zi in z])
    # Marginalize over M analytically
    resid = mu - pred
    n = len(resid)
    sum_resid = np.sum(resid / err**2)
    sum_inv = np.sum(1.0 / err**2)
    dm = sum_resid / sum_inv  # best-fit M shift
    chi2 = np.sum(((resid - dm) / err)**2)
    return chi2, dm

def chi2_total(params):
    H0, A, B, Cp = params
    chi2 = chi2_H(params, z_cc, h_cc, e_cc)
    if len(z_b) > 0:
        chi2 += chi2_H(params, z_b, h_b, e_b)
    if len(z_d) > 0:
        chi2 += chi2_H(params, z_d, h_d, e_d)
    chi2_sn, dm = chi2_SN(params, z_s, mu_s, e_s)
    chi2 += chi2_sn
    # Weak z=0 prior
    chi2 += ((H0 - 67.4) / 20.0)**2
    return chi2

# ---------- Initial fit ----------
print("Running initial fit...")
result = optimize.minimize(chi2_total, [68, -3, 4, 1], method='Nelder-Mead',
                          options={'maxiter':10000, 'xatol':1e-8, 'fatol':1e-8})
H0_best = result.x[0]
print(f"Best fit: H0={H0_best:.2f}, A={result.x[1]:.2f}, B={result.x[2]:.2f}, C={result.x[3]:.2f}")
print(f"χ²={result.fun:.1f}")

# ---------- Bootstrap ----------
print(f"\nRunning {N_BOOT} bootstrap iterations...")
H0_samples = []
A_samples = []
B_samples = []
C_samples = []

n_cc = len(z_cc)
n_b = len(z_b)
n_d = len(z_d)
n_s = len(z_s)

for i in range(N_BOOT):
    # Resample with replacement
    idx_cc = np.random.randint(0, n_cc, n_cc)
    idx_b = np.random.randint(0, n_b, n_b) if n_b > 0 else []
    idx_d = np.random.randint(0, n_d, n_d) if n_d > 0 else []
    idx_s = np.random.randint(0, n_s, n_s)
    
    z_cc_b = z_cc[idx_cc]; h_cc_b = h_cc[idx_cc]; e_cc_b = e_cc[idx_cc]
    z_b_b = z_b[idx_b] if n_b > 0 else []; h_b_b = h_b[idx_b] if n_b > 0 else []; e_b_b = e_b[idx_b] if n_b > 0 else []
    z_d_b = z_d[idx_d] if n_d > 0 else []; h_d_b = h_d[idx_d] if n_d > 0 else []; e_d_b = e_d[idx_d] if n_d > 0 else []
    z_s_b = z_s[idx_s]; mu_s_b = mu_s[idx_s]; e_s_b = e_s[idx_s]
    
    # Refit
    def chi2_boot(params):
        H0, A, B, Cp = params
        chi2 = np.sum(((h_cc_b - np.array([H_model(zi, H0, A, B, Cp) for zi in z_cc_b])) / e_cc_b)**2)
        if n_b > 0:
            chi2 += np.sum(((h_b_b - np.array([H_model(zi, H0, A, B, Cp) for zi in z_b_b])) / e_b_b)**2)
        if n_d > 0:
            chi2 += np.sum(((h_d_b - np.array([H_model(zi, H0, A, B, Cp) for zi in z_d_b])) / e_d_b)**2)
        pred = np.array([mu_from_H(lambda zi: H_model(zi, H0, A, B, Cp), zi) for zi in z_s_b])
        resid = mu_s_b - pred
        suma = np.sum(resid / e_s_b**2)
        sumb = np.sum(1.0 / e_s_b**2)
        dm = suma / sumb
        chi2 += np.sum(((resid - dm) / e_s_b)**2)
        chi2 += ((H0 - 67.4) / 20.0)**2
        return chi2
    
    try:
        res = optimize.minimize(chi2_boot, result.x, method='Nelder-Mead',
                               options={'maxiter':5000, 'xatol':1e-6, 'fatol':1e-6})
        H0_samples.append(res.x[0])
        A_samples.append(res.x[1])
        B_samples.append(res.x[2])
        C_samples.append(res.x[3])
    except:
        pass
    
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_BOOT}")

# ---------- Results ----------
H0_arr = np.array(H0_samples)
print(f"\n{'='*50}")
print(f"BOOTSTRAP RESULTS ({len(H0_arr)} successful / {N_BOOT})")
print(f"{'='*50}")
print(f"H0: mean={np.mean(H0_arr):.2f}, median={np.median(H0_arr):.2f}, std={np.std(H0_arr):.2f}")
print(f"  16th={np.percentile(H0_arr, 16):.2f}, 84th={np.percentile(H0_arr, 84):.2f}")
print(f"  range=[{np.min(H0_arr):.2f}, {np.max(H0_arr):.2f}]")
print(f"\nA: mean={np.mean(A_samples):.2f} +/- {np.std(A_samples):.2f}")
print(f"B: mean={np.mean(B_samples):.2f} +/- {np.std(B_samples):.2f}")
print(f"C: mean={np.mean(C_samples):.2f} +/- {np.std(C_samples):.2f}")

# Compare with profile-based error
print(f"\n{'='*50}")
print(f"COMPARISON")
print(f"{'='*50}")
print(f"Profile (reported):   H0 = 68.0 ± 0.8")
print(f"Bootstrap (joint):    H0 = {np.mean(H0_arr):.1f} ± {np.std(H0_arr):.1f}")
ratio = np.std(H0_arr) / 0.8
print(f"\nRatio bootstrap/profile: {ratio:.1f}x")
if ratio < 2:
    print("VERDICT: Bootstrap confirms profile error bar (±0.8 is honest)")
else:
    print(f"VERDICT: Bootstrap is {ratio:.0f}x profile. True error is ±{np.std(H0_arr):.1f}")
    print(f"  Tension with SH0ES (73±1): {(73 - np.mean(H0_arr))/np.sqrt(np.std(H0_arr)**2 + 1**2):.1f}σ")
