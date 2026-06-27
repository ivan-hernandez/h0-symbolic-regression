"""
Phase 2: Symbolic regression on SN host mass step residual.
Target: Δμ (Hubble residual) as a function of logM (host stellar mass).

We use MU_SH0ES which already has the BBC step-function correction applied.
The residual Δμ = MU_SH0ES - μ_ΛCDM(z) - M_offset should be ~0 if the
step function correction is perfect. SR discovers any remaining pattern.
"""
import numpy as np, urllib.request, os, sys
from scipy import integrate
from pysr import PySRRegressor

# --- Data ---
CACHE = "/tmp/sn_mass_step_data.npz"
if not os.path.exists(CACHE):
    URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
        "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
        "Pantheon%2BSH0ES.dat")
    print("Downloading Pantheon+ data...")
    req = urllib.request.urlopen(URL, timeout=30)
    lines = req.read().decode().strip().split('\n')
    header = lines[0].split()
    zi = header.index('zHD'); mu_i = header.index('MU_SH0ES')
    lm_i = header.index('HOST_LOGMASS')
    zl, mul, lml = [], [], []
    for line in lines[1:]:
        cols = line.split()
        if len(cols) <= max(zi, mu_i, lm_i): continue
        try:
            z = float(cols[zi]); mu = float(cols[mu_i])
            logM = float(cols[lm_i])
        except: continue
        if z <= 0.01 or logM < 5: continue
        zl.append(z); mul.append(mu); lml.append(logM)
    z = np.array(zl); mu = np.array(mul); logM = np.array(lml)
    np.savez(CACHE, z=z, mu=mu, logM=logM)
else:
    d = np.load(CACHE); z = d['z']; mu = d['mu']; logM = d['logM']

print(f"{len(z)} SNe")

# --- Reference cosmology residuals ---
c = 299792.458
DC = np.array([integrate.quad(lambda zp: c/np.sqrt(0.3*(1+zp)**3+0.7), 0, zi)[0] for zi in z])
DL = DC * (1+z)
mu_lcdm = 5*np.log10(DL) + 25
resid = mu - mu_lcdm
M_best = np.mean(resid)
resid -= M_best  # remove absolute magnitude offset

# --- SR target ---
# y = Δμ(logM), x = logM
x = logM.copy()
y = resid.copy()

# Remove outliers (5-sigma clip)
mask = np.abs(y - np.median(y)) < 5 * np.std(y)
x, y = x[mask], y[mask]
print(f"SR target: N={len(x)}, y range=[{y.min():.4f}, {y.max():.4f}]")
print(f"Step function residual: γ=0.014 mag (baseline)")

# --- PySR ---
model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt", "square", "tanh"],
    model_selection="accuracy",
    parsimony=0.001,
    populations=20,
    population_size=50,
    ncycles_per_iteration=1000,
    maxsize=15,
    random_state=42,
    procs=12,
    parallelism="multithreading",
    output_directory=os.path.join(os.path.dirname(__file__), '..', 'analysis'),
    run_id="mass_step_sr",
    verbosity=1,
    progress=True,
)

print("Running PySR...")
model.fit(x.reshape(-1, 1), y)
print(f"Best: {model.sympy()}")

# Save best equation
best_eq = str(model.sympy())
best_loss = model.get_best()['loss']
best_complexity = model.get_best()['complexity']
print(f"Best equation: {best_eq} (loss={best_loss:.6f}, complexity={best_complexity})")
