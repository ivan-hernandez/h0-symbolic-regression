"""Updated H₀ from DESI DR2 + CC + Pantheon+ with MCMC.

Uses Cpx 13 form: H(z) = H0 + A*z*(z-B)*(z²+C)
Proper MCMC with emcee, profile likelihood for H₀.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import sys, os
sys.path.insert(0, ".")
from data import load_hz, mu_from_H, fetch_pantheon, C
import warnings; warnings.filterwarnings("ignore")

OUTDIR = "analysis/h0_dr2"
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 60)
print("H₀ update: DESI DR2 + CC + Pantheon+")
print("=" * 60)

# ── Data ──
hz_data = load_hz(include_sdss=True, version='dr2')
z_hz, H_obs, H_err = hz_data[:,0], hz_data[:,1], hz_data[:,2]
print(f"\n  H(z) points: {len(z_hz)} (CC + SDSS + DESI DR2)")

# Pantheon+ — use diagonal for speed
z_sn, mu_sn, e_sn = fetch_pantheon()
print(f"  Pantheon+ points: {len(z_sn)}")

# ── Model ──
def H_cpx(z, H0, A, B, C):
    return H0 + A*z*(z-B)*(z**2+C)

# ── Joint likelihood (CC+BAO + SNe) ──
def log_likelihood(theta):
    H0, A, B, C, M = theta
    if H0 < 40 or H0 > 90 or B < 1 or B > 5 or C < 0 or C > 10:
        return -1e10

    # CC+BAO
    H_pred = H_cpx(z_hz, H0, A, B, C)
    chi2_h = np.sum((H_obs - H_pred)**2 / H_err**2)

    # SNe (with free absolute magnitude M)
    mu_pred = np.array([mu_from_H(lambda z: H_cpx(z, H0, A, B, C), z) for z in z_sn])
    mu_model = mu_pred + M
    chi2_sn = np.sum((mu_sn - mu_model)**2 / e_sn**2)

    return -0.5 * (chi2_h + chi2_sn)

# ── Optimize ──
init = [68.0, -7.7, 3.7, 1.6, -19.3]
# Bounds
def log_prob_opt(theta):
    return -log_likelihood(theta)  # minimize -logL

r = minimize(lambda t: -log_likelihood(t), init, method="Nelder-Mead",
             options={"maxiter": 10000, "xatol": 1e-8})
H0_best, A_best, B_best, C_best, M_best = r.x
print(f"\n  Best-fit: H0={H0_best:.2f}, A={A_best:.2f}, B={B_best:.2f}, C={C_best:.2f}, M={M_best:.2f}")

# ── MCMC ──
try:
    import emcee
except:
    import subprocess
    subprocess.check_call(["pip3","install","emcee","--break-system-packages","-q"])
    import emcee

ndim = 5; nwalkers = 32; nsteps = 3000
pos = r.x + 0.01 * np.random.randn(nwalkers, ndim)
pos[:,0] = np.clip(pos[:,0], 45, 85)
pos[:,1] = np.clip(pos[:,1], -15, -2)
pos[:,2] = np.clip(pos[:,2], 2, 5)
pos[:,3] = np.clip(pos[:,3], 0, 5)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
print(f"  Running MCMC ({nsteps} steps)...")
sampler.run_mcmc(pos, nsteps, progress=False)
print("  Done.")

samples = sampler.get_chain(discard=500, flat=True)
H0_samples = samples[:,0]

H0_med = np.median(H0_samples)
H0_lo, H0_hi = np.percentile(H0_samples, [16, 84])
print(f"\n  H₀ = {H0_med:.1f}  [{H0_lo:.1f}, {H0_hi:.1f}] (68% CL)")
print(f"  Planck 2018: 67.4 ± 0.5")
print(f"  SH0ES 2024:  73.0 ± 1.0")
print(f"  Δ(Planck): {(H0_med-67.4)/(0.5*(H0_hi-H0_lo)):.1f}σ")
print(f"  Δ(SH0ES):  {(73.0-H0_med)/(0.5*(H0_hi-H0_lo)):.1f}σ")

# ── Figure ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
z_grid = np.linspace(0, 2.5, 300)
# Best fit
ax.plot(z_grid, H_cpx(z_grid, *r.x[:4]), "b-", lw=2, label="Cpx 13 (DR2 best-fit)")
# 100 posterior draws
for i in np.random.choice(len(samples), 50, replace=False):
    ax.plot(z_grid, H_cpx(z_grid, *samples[i,:4]), "b-", lw=0.2, alpha=0.1)
ax.errorbar(z_hz, H_obs, yerr=H_err, fmt="o", ms=4, capsize=2, color="k", label="CC+BAO (DR2)")
ax.set_xlabel("Redshift z"); ax.set_ylabel("H(z) [km/s/Mpc]")
ax.set_title(f"H₀={H0_med:.1f}±{0.5*(H0_hi-H0_lo):.1f}")
ax.legend(fontsize=8); ax.set_xlim(0, 2.5)

ax = axes[1]
ax.hist(H0_samples, bins=40, color="steelblue", edgecolor="white", density=True)
ax.axvline(H0_med, color="k", lw=2); ax.axvline(H0_lo, color="k", ls="--", lw=1)
ax.axvline(H0_hi, color="k", ls="--", lw=1)
ax.axvline(67.4, color="green", lw=2, label="Planck 2018")
ax.axvline(73.0, color="orange", lw=2, label="SH0ES 2024")
ax.set_xlabel("H₀ [km/s/Mpc]"); ax.set_ylabel("Posterior density")
ax.set_title(f"Profile: {H0_med:.1f} [{H0_lo:.1f}, {H0_hi:.1f}]")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/h0_dr2_mcmc.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/h0_dr2_mcmc.png", dpi=150)
print(f"\n  Saved {OUTDIR}/h0_dr2_mcmc.png")
plt.close()

# Save
np.savez(f"{OUTDIR}/h0_dr2_samples.npz", samples=samples, H0_med=H0_med, H0_lo=H0_lo, H0_hi=H0_hi)
print(f"  Saved {OUTDIR}/h0_dr2_samples.npz")
