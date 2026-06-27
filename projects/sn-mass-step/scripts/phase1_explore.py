"""
Phase 1: Compute Hubble residuals from Pantheon+, explore the mass step.
"""
import numpy as np
import urllib.request, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
    "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES.dat")
CACHE = "/tmp/sn_mass_step_data.npz"

def download_data():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d['z'], d['mu'], d['logM'], d['logM_err'], d['c'], d['x1'], d['mB'], d['used']

    print("Downloading Pantheon+ data...")
    req = urllib.request.urlopen(DATA_URL, timeout=30)
    lines = req.read().decode().strip().split('\n')
    header = lines[0].split()
    z_i = header.index('zHD')
    mu_i = header.index('MU_SH0ES')
    m_i = header.index('HOST_LOGMASS')
    me_i = header.index('HOST_LOGMASS_ERR')
    c_i = header.index('c')
    x1_i = header.index('x1')
    mB_i = header.index('mB')
    used_i = header.index('USED_IN_SH0ES_HF')

    zl, mul, ml, mel, cl, x1l, mBl, usedl = [], [], [], [], [], [], [], []
    for line in lines[1:]:
        cols = line.split()
        if len(cols) <= max(z_i, mu_i, m_i): continue
        try:
            z = float(cols[z_i])
            mu = float(cols[mu_i])
            logM = float(cols[m_i])
            logM_err = float(cols[me_i])
            c_val = float(cols[c_i])
            x1_val = float(cols[x1_i])
            mB_val = float(cols[mB_i])
            used = int(cols[used_i])
        except: continue
        if z <= 0.01: continue
        if logM < 5 or logM > 13: continue
        zl.append(z); mul.append(mu); ml.append(logM)
        mel.append(logM_err); cl.append(c_val); x1l.append(x1_val)
        mBl.append(mB_val); usedl.append(used)

    z = np.array(zl); mu = np.array(mul)
    logM = np.array(ml); logM_err = np.array(mel)
    c = np.array(cl); x1 = np.array(x1l); mB = np.array(mBl)
    used = np.array(usedl)
    np.savez(CACHE, z=z, mu=mu, logM=logM, logM_err=logM_err,
             c=c, x1=x1, mB=mB, used=used)
    print(f"  {len(z)} SNe with host mass data")
    return z, mu, logM, logM_err, c, x1, mB, used

def distance_modulus_lcdm(z, H0, Om=0.3):
    """Flat ΛCDM distance modulus."""
    c = 299792.458  # km/s
    n = len(z)
    DC = np.zeros(n)
    for i in range(n):
        def E(zp): return np.sqrt(Om*(1+zp)**3 + (1-Om))
        from scipy import integrate
        DC[i], _ = integrate.quad(lambda zp: c / E(zp), 0, z[i])
    DL = DC * (1 + z)
    mu = 5 * np.log10(DL) + 25  # DL in Mpc
    return mu

def main():
    z, mu, logM, logM_err, c, x1, mB, used = download_data()
    print(f"\nData summary:")
    print(f"  z range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"  mu range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  logM range: [{logM.min():.3f}, {logM.max():.3f}]")
    print(f"  logM < 10: {(logM < 10).sum()} ({100*(logM<10).sum()/len(logM):.0f}%)")
    print(f"  logM > 10: {(logM > 10).sum()} ({100*(logM>10).sum()/len(logM):.0f}%)")

    # Fit H0 in flat ΛCDM
    from scipy.optimize import minimize
    def chi2_H0(H0):
        mu_model = distance_modulus_lcdm(z, H0[0])
        # Marginalize over M (absolute magnitude offset)
        resid = mu - mu_model
        # Best-fit M is weighted mean residual
        w = np.ones_like(resid)
        M_best = np.average(resid, weights=w)
        chi2 = np.sum(((resid - M_best))**2)
        return chi2

    result = minimize(chi2_H0, [70], bounds=[(50, 90)], method='Nelder-Mead')
    H0_best = result.x[0]
    print(f"\nBest-fit H0 (ΛCDM, Ωm=0.3): H0 = {H0_best:.2f}")

    # Compute residuals
    mu_model = distance_modulus_lcdm(z, H0_best)
    resid = mu - mu_model
    M_best = np.average(resid)
    resid -= M_best  # Remove absolute magnitude offset
    print(f"  Best-fit M = {M_best:.3f} mag")

    # Simple step function fit
    threshold = 10.0
    low = logM < threshold
    high = logM >= threshold
    step_amp = np.mean(resid[high]) - np.mean(resid[low])
    print(f"\nStep function at logM=10:")
    print(f"  γ (step amplitude) = {step_amp:.4f} ± {np.std(resid[high])/np.sqrt(len(resid[high])):.4f} mag")

    # Welch's t-test for significance
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(resid[low], resid[high])
    print(f"  t-test: t={t_stat:.2f}, p={p_val:.2e}")
    print(f"  <Δμ> low-mass: {np.mean(resid[low]):.4f} ± {np.std(resid[low])/np.sqrt(len(resid[low])):.4f}")
    print(f"  <Δμ> high-mass: {np.mean(resid[high]):.4f} ± {np.std(resid[high])/np.sqrt(len(resid[high])):.4f}")

    # Binned residuals
    bins = np.linspace(7.5, 12.0, 20)
    bin_c = (bins[1:] + bins[:-1]) / 2
    bin_m = np.array([np.mean(resid[(logM >= bins[i]) & (logM < bins[i+1])]) for i in range(len(bins)-1)])
    bin_s = np.array([np.std(resid[(logM >= bins[i]) & (logM < bins[i+1])]) / np.sqrt(np.sum((logM >= bins[i]) & (logM < bins[i+1]))) for i in range(len(bins)-1)])
    keep = np.isfinite(bin_m)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(logM, resid, s=3, alpha=0.3, color='gray', label='Individual SNe')
    ax.errorbar(bin_c[keep], bin_m[keep], yerr=bin_s[keep], fmt='o-', color='black', lw=2, label='Binned avg')
    ax.axvline(x=10, color='red', ls='--', alpha=0.5, label=f'Step at logM=10 (γ={step_amp:.3f})')
    ax.axhline(y=step_amp/2, color='red', ls=':', alpha=0.3)
    ax.axhline(y=-step_amp/2, color='red', ls=':', alpha=0.3)
    ax.set_xlabel('log10(Host Stellar Mass / M☉)')
    ax.set_ylabel('Δμ (Hubble residual, mag)')
    ax.set_title(f'SN Host Mass Step (N={len(z)}, H0={H0_best:.1f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Histogram of logM split by residual sign
    pos_resid = resid > 0; neg_resid = resid <= 0
    ax.hist(logM[pos_resid], bins=30, alpha=0.5, label='Δμ > 0 (brighter)', density=True)
    ax.hist(logM[neg_resid], bins=30, alpha=0.5, label='Δμ < 0 (dimmer)', density=True)
    ax.axvline(x=10, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('log10(Host Stellar Mass / M☉)')
    ax.set_ylabel('Fraction')
    ax.set_title('Host Mass Distribution by Residual Sign')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'mass_step_explore.png')
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")

    # Save processed data for SR phase
    data_out = os.path.join(os.path.dirname(__file__), '..', 'data', 'residuals.npz')
    np.savez(data_out, z=z, mu=mu, logM=logM, logM_err=logM_err,
             resid=resid, H0=H0_best, c=c, x1=x1, mB=mB)
    print(f"Saved {data_out}")

if __name__ == "__main__":
    main()
