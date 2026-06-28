"""Phase 1: Explore SN distance modulus residuals vs x1, c, host mass.

The standard SALT model assumes linear corrections: μ = mB - M + α·x1 - β·c
We test whether non-linear forms improve the fit using ΛCDM as baseline.
"""
import csv, os, math
import numpy as np
from scipy import stats

DATAFILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'pantheon_plus.csv')

# Constants
C = 299792.458  # km/s

def mu_lcdm(z, H0=68.0, Om=0.321):
    """Flat ΛCDM distance modulus."""
    Ol = 1.0 - Om
    def integrand(zp):
        return 1.0 / math.sqrt(Om*(1+zp)**3 + Ol)
    # Simpson integration
    n = 500
    a, b = 0.0, z
    h = (b-a)/(2*n)
    xs = [a + i*h for i in range(2*n+1)]
    fx = [integrand(x) for x in xs]
    integral = h/3 * (fx[0] + fx[-1] + 4*sum(fx[1::2]) + 2*sum(fx[2:-1:2]))
    Dc = C / (100.0 * math.sqrt(Om*(1+z)**3 + Ol)) * integral
    DL = (1+z) * Dc
    return 5.0 * math.log10(max(DL, 1e-10)) + 25.0

# Load
rows = []
with open(DATAFILE) as f:
    for row in csv.DictReader(f):
        z = float(row['zHD'])
        mu_data = float(row['MU_SH0ES'])
        x1 = float(row['x1'])
        c_val = float(row['c'])
        host = float(row['HOST_LOGMASS']) if row['HOST_LOGMASS'] else np.nan
        mB = float(row['mB'])
        if z > 0.01:
            rows.append({'z': z, 'mu': mu_data, 'x1': x1, 'c': c_val, 'host': host, 'mB': mB})

print('=== Phase 1: SNIa Distance Modulus Residuals ===')
print('SNe with z > 0.01: %d' % len(rows))

# Compute residuals relative to ΛCDM
mu_model = np.array([mu_lcdm(r['z']) for r in rows])
mu_data = np.array([r['mu'] for r in rows])
residuals = mu_data - mu_model

print('Mean residual: %.4f' % np.mean(residuals))
print('Std residual: %.4f' % np.std(residuals))

# Get arrays
x1 = np.array([r['x1'] for r in rows])
c_vals = np.array([r['c'] for r in rows])
host = np.array([r['host'] for r in rows])
z_arr = np.array([r['z'] for r in rows])
valid_host = ~np.isnan(host)

# Test linear corrections: μ = a + α*x1 + β*c + γ*log(host_mass)
# First fit α, β
A = np.column_stack([np.ones_like(x1), x1, c_vals])
coeff = np.linalg.lstsq(A, residuals, rcond=None)[0]
resid_linear = residuals - A @ coeff
mse_lin = np.mean(resid_linear**2)
print('\n=== Linear SALT correction fit ===')
print('Intercept: %.4f' % coeff[0])
print('α (x1 coeff): %.4f' % coeff[1])
print('β (c coeff): %.4f' % coeff[2])
print('R: %.4f' % (1 - mse_lin/np.var(residuals)))

# Test quadratic in x1 and c
A2 = np.column_stack([np.ones_like(x1), x1, x1**2, c_vals, c_vals**2])
coeff2 = np.linalg.lstsq(A2, residuals, rcond=None)[0]
resid_quad = residuals - A2 @ coeff2
mse_quad = np.mean(resid_quad**2)
print('\n=== Quadratic SALT correction fit ===')
print('Intercept: %.4f' % coeff2[0])
print('x1: %.4f' % coeff2[1])
print('x1^2: %.4f' % coeff2[2])
print('c: %.4f' % coeff2[3])
print('c^2: %.4f' % coeff2[4])
print('R: %.4f' % (1 - mse_quad/np.var(residuals)))

# Host mass step
print('\n=== Host mass step ===')
host_valid = host[valid_host]
resid_valid = residuals[valid_host]
# Standard step at logM = 10
step = [r for r in rows if not np.isnan(r['host'])]
low_mass = [r for r in step if r['host'] < 10]
high_mass = [r for r in step if r['host'] >= 10]
if low_mass and high_mass:
    mu_low = np.array([mu_lcdm(r['z']) for r in low_mass])
    mu_high = np.array([mu_lcdm(r['z']) for r in high_mass])
    res_low = np.array([r['mu'] for r in low_mass]) - mu_low
    res_high = np.array([r['mu'] for r in high_mass]) - mu_high
    t_stat, p_val = stats.ttest_ind(res_low, res_high)
    print('Low mass (<10): mean=%.4f, n=%d' % (np.mean(res_low), len(res_low)))
    print('High mass (>=10): mean=%.4f, n=%d' % (np.mean(res_high), len(res_high)))
    print('Difference: %.4f (t=%.2f, p=%.4f)' % (np.mean(res_high)-np.mean(res_low), t_stat, p_val))

# Full model with all terms
print('\n=== Full model: linear + host step ===')
A_full = np.column_stack([np.ones_like(x1), x1, c_vals, (host >= 10).astype(float)])
mask = ~np.isnan(host)
coeff_full = np.linalg.lstsq(A_full[mask], residuals[mask], rcond=None)[0]
resid_full = residuals[mask] - A_full[mask] @ coeff_full
mse_full = np.mean(resid_full**2)
print('Terms: const, x1, c, host_step')
print('Coeffs:', coeff_full)
print('R: %.4f' % (1 - mse_full/np.var(residuals[mask])))

# Check for non-linearity: do x1 or c residuals show trend?
print('\n=== Residual trend check ===')
from scipy.stats import spearmanr
for name, val in [('x1', x1), ('c', c_vals), ('z', z_arr), ('host', host)]:
    mask_v = ~np.isnan(val)
    if np.sum(mask_v) > 10:
        rho, p = spearmanr(residuals[mask_v], val[mask_v])
        print('  %-10s: rho=%.3f, p=%.4f' % (name, rho, p))

# AIC comparison
n = len(residuals)
print('\n=== AIC comparison ===')
# Just compute for the linear model with host mass
mask = ~np.isnan(host)
for name, mse, k in [('Linear (x1,c)', mse_lin, 3), ('Quad (x1,c)', mse_quad, 5), ('Full+step', mse_full, 4)]:
    aic = n * math.log(mse) + 2 * k
    print('  %-20s AIC=%.1f' % (name, aic))
