"""Joint bootstrap: CC+BAO+DESI+SNe with Cpx 13 fixed form."""
import sys, os, math, time, random
import numpy as np
from scipy.interpolate import interp1d
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D

random.seed(420)
np.random.seed(420)

N_BOOT = 200
C = 299792.458

print("Loading data...")
cc = D.get_cc(); z_cc, h_cc, e_cc = cc[:,0], cc[:,1], cc[:,2]
bao = D.get_bao_sdss(); z_b, h_b, e_b = bao[:,0], bao[:,1], bao[:,2]
desi = D.get_desi_dr2(); z_d, h_d, e_d = desi[:,0], desi[:,1], desi[:,2]
z_s, mu_s, e_s = D.fetch_pantheon()
print(f"CC: {len(z_cc)}, SDSS: {len(z_b)}, DESI: {len(z_d)}, SNe: {len(z_s)}")

def mu_batch(z_arr, H0, A, B, Cv):
    z_max = np.max(z_arr)
    n_grid = max(100, int(z_max * 200))
    grid = np.linspace(0, z_max, n_grid)
    inv_H = 1.0 / np.maximum(H0 + A * grid * (grid - B) * (grid*grid + Cv), 1e-10)
    dz = grid[1] - grid[0]
    cum = np.zeros(n_grid)
    cum[1:] = np.cumsum(dz * (inv_H[:-1] + inv_H[1:]) / 2)
    f = interp1d(grid, cum, bounds_error=False, fill_value=0.0)
    Dc = C * f(z_arr)
    return 5.0 * np.log10(np.maximum((1 + z_arr) * Dc, 1e-10)) + 25.0

def chi2_total(params, z_cc_b, h_cc_b, e_cc_b, z_b_b, h_b_b, e_b_b,
               z_d_b, h_d_b, e_d_b, z_s_b, mu_s_b, e_s_b):
    H0, A, B, Cv = params
    chi2v = np.sum(((h_cc_b - (H0 + A * z_cc_b * (z_cc_b - B) * (z_cc_b*z_cc_b + Cv))) / e_cc_b)**2)
    if len(z_b_b) > 0:
        chi2v += np.sum(((h_b_b - (H0 + A * z_b_b * (z_b_b - B) * (z_b_b*z_b_b + Cv))) / e_b_b)**2)
    if len(z_d_b) > 0:
        chi2v += np.sum(((h_d_b - (H0 + A * z_d_b * (z_d_b - B) * (z_d_b*z_d_b + Cv))) / e_d_b)**2)
    pred_mu = mu_batch(z_s_b, H0, A, B, Cv)
    resid = mu_s_b - pred_mu
    suma = np.sum(resid / e_s_b**2)
    chi2v += np.sum(((resid - suma/np.sum(1.0/e_s_b**2)) / e_s_b)**2)
    chi2v += ((H0 - 67.4) / 20.0)**2
    return chi2v

# Initial fit
print("\nInitial fit...")
t0 = time.time()
res0 = optimize.minimize(
    lambda p: chi2_total(p, z_cc, h_cc, e_cc, z_b, h_b, e_b, z_d, h_d, e_d, z_s, mu_s, e_s),
    [68, -7, 3.8, 1.7], method='L-BFGS-B',
    bounds=[(50,80), (-15,0), (0,8), (-2,5)],
    options={'maxiter':5000})
t1 = time.time()
print(f"H0={res0.x[0]:.2f} A={res0.x[1]:.2f} B={res0.x[2]:.2f} C={res0.x[3]:.2f}")
print(f"chi2={res0.fun:.1f}  [{t1-t0:.0f}s]")

# Bootstrap
print(f"\nBootstrap ({N_BOOT} iterations)...")
H0_samps, A_samps, B_samps, C_samps = [], [], [], []
n_cc, n_b, n_d, n_s = len(z_cc), len(z_b), len(z_d), len(z_s)

for i in range(N_BOOT):
    t0 = time.time()
    idx_cc = np.random.randint(0, n_cc, n_cc)
    idx_b = np.random.randint(0, n_b, n_b)
    idx_d = np.random.randint(0, n_d, n_d)
    idx_s = np.random.randint(0, n_s, n_s)
    try:
        res = optimize.minimize(
            lambda p: chi2_total(p, z_cc[idx_cc], h_cc[idx_cc], e_cc[idx_cc],
                                 z_b[idx_b], h_b[idx_b], e_b[idx_b],
                                 z_d[idx_d], h_d[idx_d], e_d[idx_d],
                                 z_s[idx_s], mu_s[idx_s], e_s[idx_s]),
            res0.x, method='L-BFGS-B',
            bounds=[(50,80), (-15,0), (0,8), (-2,5)],
            options={'maxiter':5000})
        H0_samps.append(res.x[0])
        A_samps.append(res.x[1])
        B_samps.append(res.x[2])
        C_samps.append(res.x[3])
    except:
        pass
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_BOOT}")

# Results
H0_arr = np.array(H0_samps)
print(f"\n{'='*60}")
print(f"  JOINT BOOTSTRAP RESULTS")
print(f"{'='*60}")
print(f"  Successful:          {len(H0_arr)}/{N_BOOT}")
print(f"  H0 mean:             {np.mean(H0_arr):.2f}")
print(f"  H0 median:           {np.median(H0_arr):.2f}")
print(f"  H0 std:              {np.std(H0_arr):.2f}")
print(f"  16-84th pct:         [{np.percentile(H0_arr, 16):.2f}, {np.percentile(H0_arr, 84):.2f}]")
print(f"  95% CI:              [{np.percentile(H0_arr, 2.5):.2f}, {np.percentile(H0_arr, 97.5):.2f}]")
print(f"\n{'='*60}")
print(f"  COMPARISON WITH REPORTED ±0.8")
print(f"{'='*60}")
ratio = np.std(H0_arr) / 0.8
print(f"  Bootstrap std:       ±{np.std(H0_arr):.2f}")
print(f"  Reported (profile):  ±0.80")
print(f"  Ratio:               {ratio:.1f}x")
if ratio < 1.5:
    print(f"  VERDICT: Bootstrap confirms the reported ±0.8.")
else:
    print(f"  VERDICT: Bootstrap is {ratio:.0f}x profile. True σ ≈ ±{np.std(H0_arr):.1f}.")
tension = (73.0 - np.mean(H0_arr)) / np.sqrt(np.std(H0_arr)**2 + 1.0**2)
print(f"\n  Tension with SH0ES (73±1): {tension:.1f}σ")

# Save
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(outdir, exist_ok=True)
np.savetxt(os.path.join(outdir, 'bootstrap_h0.txt'), H0_arr)
print(f"\n  Saved to output/bootstrap_h0.txt")
