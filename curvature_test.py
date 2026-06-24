"""Universe curvature from SNe + BAO + DESI DR2.

D_L from SNe, D_M from BAO (transverse), D_H from BAO (line-of-sight).
Non-flat Cpx 13: H(z) form + free Ω_k.
"""
import numpy as np
from scipy.optimize import minimize
import sys; sys.path.insert(0,".")
from data import C as c_light

# ── Data ──
# H(z): CC + BAO D_H (line-of-sight) — already in data.py
# D_M(z)/r_s: BAO transverse — DESI DR2 Table 1
# DESI DR2: D_M/r_s values
r_s = 147.0  # Mpc

# DESI DR2 BAO: z, D_M/r_s, error
DM_DATA = np.array([
    [0.510, 13.364, 0.071],
    [0.706, 16.259, 0.086],
    [0.934, 19.136, 0.094],
    [1.321, 22.529, 0.188],
    [1.484, 24.221, 0.474],
    [2.330, 35.251, 0.378],
])

# SDSS BAO: D_M/r_s (Alam+2017)
DM_SDSS = np.array([
    [0.380, 10.23, 0.17],
    [0.510, 13.36, 0.21],
    [0.610, 15.77, 0.28],
])

# Combine D_M data (use DESI where available, SDSS otherwise)
z_dm = np.concatenate([DM_SDSS[:,0], DM_DATA[:,0]])
dm_obs = np.concatenate([DM_SDSS[:,1], DM_DATA[:,1]])
dm_err = np.concatenate([DM_SDSS[:,2], DM_DATA[:,2]])

# Remove z=0.51 SDSS duplicate (keep DESI which has better precision)
mask = ~((z_dm == 0.51) & (dm_err > 0.2))
z_dm, dm_obs, dm_err = z_dm[mask], dm_obs[mask], dm_err[mask]

print(f"D_M(z) data: {len(z_dm)} points (SDSS + DESI DR2)")

# H(z) data
from data import load_hz
hz = load_hz(include_sdss=True, version='dr2')
z_h, H_obs, H_err = hz[:,0], hz[:,1], hz[:,2]
print(f"H(z) data: {len(z_h)} points")

# Pantheon+ (full cov)
from pantheon_cov import load_cov
z_sn, mu_sn, Cinv, Cinv_1sum = load_cov()
print(f"Pantheon+: {len(z_sn)} SNe")

# ── Model ──
def D_M(z, H0, A, B, C, Ok):
    """Transverse comoving distance in non-flat universe."""
    Z = np.linspace(0, np.max(z)+0.01, 5000) if isinstance(z, np.ndarray) else np.linspace(0, z+0.01, 5000)
    dz = Z[1] - Z[0]
    H_vals = H0 + A*Z*(Z-B)*(Z**2+C)
    integrand = 1.0 / H_vals
    D_c = c_light * np.cumsum(integrand) * dz  # line-of-sight comoving distance
    Dc_at_z = np.interp(z, Z, D_c)

    if abs(Ok) < 1e-10:
        return Dc_at_z

    sqrt_abs_Ok = np.sqrt(abs(Ok))
    x = sqrt_abs_Ok * Dc_at_z * H0 / c_light

    if Ok > 0:
        return (c_light / H0) * np.sin(x) / sqrt_abs_Ok
    else:
        return (c_light / H0) * np.sinh(x) / sqrt_abs_Ok

def log_likelihood(theta):
    H0, A, B, C, Ok = theta
    if H0 < 40 or H0 > 90 or B < 1 or B > 5 or C < 0 or C > 10 or abs(Ok) > 0.8:
        return -1e10

    # H(z)
    Hp = H0 + A*z_h*(z_h-B)*(z_h**2+C)
    chi2_hz = np.sum((H_obs - Hp)**2 / H_err**2)

    # D_M(z)
    dm_pred = D_M(z_dm, H0, A, B, C, Ok) / r_s
    chi2_dm = np.sum((dm_obs - dm_pred)**2 / dm_err**2)

    # SNe (analytic M marginalization, full covariance)
    D_M_sn = D_M(z_sn, H0, A, B, C, Ok)
    D_L_sn = (1 + z_sn) * D_M_sn
    mu0 = 5.0 * np.log10(D_L_sn) + 25.0
    r = mu_sn - mu0
    Mh = (Cinv @ r).sum() / Cinv_1sum
    chi2_sn = (r @ (Cinv @ r)) - Mh**2 * Cinv_1sum

    return -0.5 * (chi2_hz + chi2_dm + chi2_sn)

# ── Optimize ──
print("\nOptimizing...")
r = minimize(lambda t: -log_likelihood(t), [68, -7.7, 3.7, 1.6, 0.0],
             method="Nelder-Mead", options={"maxiter": 10000, "xatol": 1e-8})
H0_b, A_b, B_b, C_b, Ok_b = r.x

# Profile Ok
print(f"\nBest-fit: H0={H0_b:.1f}, Ok={Ok_b:.4f}")
print("Ok profile:")
for Ok_v in np.linspace(-0.1, 0.1, 11):
    def f_ok(theta):
        H0, A, B, C = theta
        return -log_likelihood([H0, A, B, C, Ok_v])
    r_ok = minimize(f_ok, [68, -7.7, 3.7, 1.6], method="Nelder-Mead", options={"maxiter": 5000})
    chi2 = -2 * log_likelihood([r_ok.x[0], r_ok.x[1], r_ok.x[2], r_ok.x[3], Ok_v])
    print(f"  Ω_k = {Ok_v:+.2f}: χ² = {chi2:.0f}")

# Compute dchi2
chi2_vals = []
for Ok_v in np.linspace(-0.15, 0.15, 20):
    def f_ok(theta):
        H0, A, B, C = theta
        return -log_likelihood([H0, A, B, C, Ok_v])
    r_ok = minimize(f_ok, [68, -7.7, 3.7, 1.6], method="Nelder-Mead", options={"maxiter": 5000})
    chi2_vals.append(-2 * log_likelihood([r_ok.x[0], r_ok.x[1], r_ok.x[2], r_ok.x[3], Ok_v]))
chi2_arr = np.array(chi2_vals)
dchi2 = chi2_arr - chi2_arr.min()
within1 = np.linspace(-0.15, 0.15, 20)[dchi2 <= 1]
if len(within1) > 1:
    print(f"\nΩ_k 68% CL: [{within1[0]:+.4f}, {within1[-1]:+.4f}]")
else:
    print(f"\nΩ_k 68% CL: Ok={Ok_b:.4f} (tightly constrained)")
# Compute sigma from dchi2 curvature at minimum
idx = np.argmin(dchi2)
if idx > 1 and idx < len(dchi2)-2:
    curv = (dchi2[idx+1] - 2*dchi2[idx] + dchi2[idx-1]) / (0.015**2)
    sig = np.sqrt(2/curv) if curv > 0 else 0.5
    print(f"Ω_k = {Ok_b:.4f} ± {sig:.4f}")
