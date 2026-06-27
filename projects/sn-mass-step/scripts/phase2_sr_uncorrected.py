"""
Phase 2 (alt): SR on UNCORRECTED mass step.
Reverse the BBC correction to recover the full γ=0.059 signal,
then discover f(logM) with SR.
"""
import numpy as np, urllib.request, os, sys
from scipy import integrate
from pysr import PySRRegressor

# Pantheon+ data
URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
    "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES.dat")
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
        logM_val = float(cols[lm_i])
    except: continue
    if z <= 0.01 or logM_val < 5: continue
    zl.append(z); mul.append(mu); lml.append(logM_val)

z = np.array(zl); mu = np.array(mul); logM = np.array(lml)
step = (logM >= 10).astype(float)

# Residual from reference cosmology
c = 299792.458
DC = np.array([integrate.quad(lambda zp: c/np.sqrt(0.3*(1+zp)**3+0.7), 0, zi)[0] for zi in z])
DL = DC * (1+z)
mu_lcdm = 5*np.log10(DL) + 25
resid_corr = mu - mu_lcdm  # corrected residual (γ=0.014)

# Reverse the BBC correction to get UNCORRECTED residual
# The BBC correction embedded in MU_SH0ES is ~-0.045*H(logM-10)
# Reversing: mu_uncorr = mu - 0.045*step (since correction SUBTRACTS from mu)
# Wait: correction = -0.045*H. So: mu_corrected = mu_uncorrected - 0.045*H
# To reverse: mu_uncorrected = mu_corrected + 0.045*H
gamma_bbc = 0.045  # correction amplitude embedded in MU_SH0ES
mu_uncorr = mu + gamma_bbc * step
resid_uncorr = mu_uncorr - mu_lcdm
M_best = np.mean(resid_uncorr)
resid_uncorr -= M_best

x = logM.copy()
y = resid_uncorr.copy()

print(f"Uncorrected residual: N={len(x)}, y range=[{y.min():.4f}, {y.max():.4f}]")
print(f"Step: γ={np.mean(y[logM>=10]) - np.mean(y[logM<10]):.4f} mag")

# PySR
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
    run_id="mass_step_sr_uncorr",
    verbosity=1,
    progress=True,
)

print("Running PySR on uncorrected residual...")
model.fit(x.reshape(-1, 1), y)
print(f"Best: {model.sympy()}")
