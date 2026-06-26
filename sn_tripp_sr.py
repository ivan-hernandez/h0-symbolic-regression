"""
SR discovery on SN Ia Tripp correction: non-linear f(stretch, color).
Search for residual structure in distance modulus beyond linear α·x1 − β·c.
"""
import numpy as np
from pysr import PySRRegressor
import urllib.request
import warnings
warnings.filterwarnings('ignore')

SEED = 42
rng = np.random.RandomState(SEED)

# === Load Pantheon+ data ===
# Columns: 0=CID, 1=IDSURVEY, 2=zHD, 3=zHDERR, 4=zCMB, 5=zCMBERR,
# 6=zHEL, 7=zHELERR, 8=m_b_corr, 9=m_b_corr_err_DIAG,
# 10=MU_SH0ES, 11=MU_SH0ES_ERR_DIAG,
# 12=CEPH_DIST, 13=IS_CALIBRATOR, 14=USED_IN_SH0ES_HF,
# 15=c, 16=cERR, 17=x1, 18=x1ERR, ...
URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
    "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES.dat")
print("Downloading Pantheon+ data...")
data = np.loadtxt(urllib.request.urlopen(URL), dtype=str,
                  skiprows=1, delimiter=None)
print(f"Loaded {len(data)} entries")

# Parse columns (skip string columns 0, 1)
z = data[:, 2].astype(float)
mu_obs = data[:, 10].astype(float)
is_cal = data[:, 13].astype(int)
c = data[:, 15].astype(float)
x1 = data[:, 17].astype(float)

# === Select cosmology SNe (IS_CALIBRATOR=0) ===
sn_mask = is_cal == 0
z = z[sn_mask]
mu_obs = mu_obs[sn_mask]
x1 = x1[sn_mask]
c = c[sn_mask]
print(f"Cosmological SNe selected: {len(z)}")

# === Reference cosmology: flat ΛCDM, H0=70, Ωm=0.3 ===
H0_ref, Om_ref = 70.0, 0.3
c_light = 299792.458  # km/s
# Distance modulus μ = 5*log10(d_L/Mpc) + 25
def mu_lcdm(z, H0=H0_ref, Om=Om_ref):
    Ok = 1 - Om
    from scipy.integrate import quad
    def E(zp):
        return np.sqrt(Om*(1+zp)**3 + Ok*(1+zp)**2 + (1-Om-Ok))
    dH = c_light / H0
    dc = dH * np.array([quad(E, 0, zi)[0] for zi in z])
    if Ok > 0:
        dm = dH / np.sqrt(Ok) * np.sinh(np.sqrt(Ok) * dc / dH)
    elif Ok < 0:
        dm = dH / np.sqrt(-Ok) * np.sin(np.sqrt(-Ok) * dc / dH)
    else:
        dm = dc
    dl = dm * (1 + z)
    return 5 * np.log10(dl) + 25

mu_cosmo = mu_lcdm(z)

# === Tripp residual ===
# MU_SH0ES already has linear Tripp correction (α, β) applied
# residual = MU_SH0ES - μ_ΛCDM(Planck) should be ~0 if H0=70 matches
# But SH0ES uses different H0 and M_B, so there's an offset
residual = mu_obs - mu_cosmo
# Remove mean offset (M_B + H0 difference)
residual -= np.mean(residual)

print(f"\nResidual stats:")
print(f"  Mean residual: {np.mean(residual):.4f}")
print(f"  RMS residual: {np.std(residual):.4f}")
print(f"  x1 range: [{x1.min():.2f}, {x1.max():.2f}]")
print(f"  c range: [{c.min():.3f}, {c.max():.3f}]")

# === Baseline: linear Tripp (should be ~0 since already applied) ===
A = np.column_stack([np.ones_like(x1), x1, c])
coeff, *_ = np.linalg.lstsq(A, residual, rcond=None)
baseline_pred = A @ coeff
baseline_rms = np.sqrt(np.mean((residual - baseline_pred)**2))
print(f"\nBaseline residual after refitting (α, β): RMS={baseline_rms:.4f}")
print(f"  α (x1): {coeff[1]:.4f} (should be near 0 if already applied)")
print(f"  β (c): {coeff[2]:.4f} (should be near 0 if already applied)")

# === SR Discovery ===
# Features: x1 (stretch), c (color)
# Also add quadratic features as baselines
print("\n=== SR on Tripp residual ===\n")

X = np.column_stack([x1, c])
feature_names = ['x1', 'c']

model = PySRRegressor(
    niterations=1000,
    populations=20,
    population_size=50,
    ncycles_per_iteration=300,
    maxsize=20,
    parsimony=0.01,
    warm_start=False,
    turbo=True,
    timeout_in_seconds=36000,
    random_state=SEED,
    binary_operators=["+", "-", "*", "/", "pow"],
    unary_operators=["cos", "exp", "log", "sqrt", "square"],
    loss="L2DistLoss()",
    model_selection="accuracy",
    output_directory="./output",
    progress=True,
    verbosity=1,
)

model.fit(X, residual, variable_names=feature_names)

# === Evaluate ===
print("\n=== Best models ===")
for i in range(min(5, len(model.equations_))):
    row = model.equations_.iloc[i]
    print(f"  C={row['complexity']}, loss={row['loss']:.6f}: {row['sympy_format']}")

# === Compare with quadratic baseline ===
# residual = a*x1² + b*c² + d*x1*c + e*x1 + f*c + g
A2 = np.column_stack([np.ones_like(x1), x1, c, x1**2, c**2, x1*c])
coeff2, *_ = np.linalg.lstsq(A2, residual, rcond=None)
quad_pred = A2 @ coeff2
quad_rms = np.sqrt(np.mean((residual - quad_pred)**2))
print(f"\nQuadratic model RMS: {quad_rms:.4f}")
print(f"  Terms: {coeff2[1]:.4f}*x1 {coeff2[2]:.4f}*c {coeff2[3]:.4f}*x1² {coeff2[4]:.4f}*c² {coeff2[5]:.4f}*x1*c")

# Check for structure by binning in x1 and c
print("\n=== Binned residuals ===")
x1_bins = np.percentile(x1, [0, 25, 50, 75, 100])
c_bins = np.percentile(c, [33, 67])
print(f"x1 bins: {x1_bins}")
for i in range(4):
    bx = (x1 >= x1_bins[i]) & (x1 < x1_bins[i+1])
    print(f"  x1 in [{x1_bins[i]:.1f}, {x1_bins[i+1]:.1f}]: N={bx.sum()}, residual={np.mean(residual[bx]):.3f}±{np.std(residual[bx]):.3f}")

print(f"\nc bins: {c_bins}")
for i in range(2):
    bc = (c >= c_bins[i]) & (c < c_bins[i+1])
    print(f"  c in [{c_bins[i]:.3f}, {c_bins[i+1]:.3f}]: N={bc.sum()}, residual={np.mean(residual[bc]):.3f}±{np.std(residual[bc]):.3f}")

# === Summary ===
print("\n===== TRIPP SR SUMMARY =====")
print(f"N SN: {len(sn)} (non-calibrator, non-cepheid-host)")
print(f"Residual RMS (total): {np.std(residual):.4f}")
print(f"Linear Tripp refit RMS: {baseline_rms:.4f}")
print(f"Quadratic RMS: {quad_rms:.4f}")
print(f"SR best loss (C={model.equations_.iloc[0]['complexity']}): {model.equations_.iloc[0]['loss']:.6f}")
print(f"SR best model: {model.equations_.iloc[0]['sympy_format']}")
