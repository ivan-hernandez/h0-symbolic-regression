"""
Cepheid Period-Luminosity relation discovery with symbolic regression.
Target: F160W Wesenheit magnitude = m_F160W - R_H * (V-I)
Features: log10(P), V-I, [O/H]
Nuisance: per-host distance moduli (removed via demeaning)
"""
import numpy as np
from pysr import PySRRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
rng = np.random.RandomState(SEED)

# === Load NIR Cepheid data ===
lines = open('data/R22_orig19_NIR.out').readlines()
records = []
for line in lines[2:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    host = parts[0]
    period = float(parts[4])
    vi_str = parts[5]
    f160w = float(parts[7])
    metal = float(parts[9])
    # Handle missing V-I
    if vi_str == '1' or vi_str == '':
        continue
    vi = float(vi_str)
    records.append((host, period, vi, f160w, metal))

print(f'Loaded {len(records)} Cepheids')

# === Compute Wesenheit magnitudes ===
# W = m_F160W - R_H * (V-I), R_H = 0.46 (standard for H-band)
R_H = 0.46
host_arr = np.array([r[0] for r in records])
logP = np.log10(np.array([r[1] for r in records]))
vi = np.array([r[2] for r in records])
W = np.array([r[3] - R_H * r[2] for r in records])
metal = np.array([r[4] for r in records])

# === Remove per-host means (distance moduli) ===
hosts = np.unique(host_arr)
host_idx = {h: i for i, h in enumerate(hosts)}
n_host = len(hosts)
print(f'Hosts: {n_host}')

W_demeaned = np.copy(W)
logP_demeaned = np.copy(logP)
vi_demeaned = np.copy(vi)
metal_demeaned = np.copy(metal)
for h in hosts:
    mask = host_arr == h
    W_demeaned[mask] -= np.mean(W[mask])
    logP_demeaned[mask] -= np.mean(logP[mask])
    vi_demeaned[mask] -= np.mean(vi[mask])
    metal_demeaned[mask] -= np.mean(metal[mask])

# === Train/test split ===
N = len(records)
idx = np.arange(N)
rng.shuffle(idx)
split = int(0.8 * N)
train_idx = idx[:split]
test_idx = idx[split:]

X_train = np.column_stack([logP_demeaned[train_idx],
                            vi_demeaned[train_idx],
                            metal_demeaned[train_idx]])
y_train = W_demeaned[train_idx]
X_test = np.column_stack([logP_demeaned[test_idx],
                           vi_demeaned[test_idx],
                           metal_demeaned[test_idx]])
y_test = W_demeaned[test_idx]

print(f'Train: {len(train_idx)}, Test: {len(test_idx)}')

# === Baseline linear fit ===
# W = a*log10(P) + b*(V-I) + c*[O/H] + const
A = np.column_stack([np.ones_like(y_train), X_train])
coeff, *_ = np.linalg.lstsq(A, y_train, rcond=None)
baseline_pred = A @ coeff
baseline_r2 = 1 - np.sum((y_train - baseline_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
baseline_rmse = np.sqrt(np.mean((y_train - baseline_pred)**2))
print(f'Baseline linear (a*logP + b*vi + c*metal + d): R2={baseline_r2:.4f}, RMSE={baseline_rmse:.4f}')
print(f'  Coefficients: a={coeff[1]:.4f}, b={coeff[2]:.4f}, c={coeff[3]:.4f}, d={coeff[0]:.4f}')

# Test baseline
A_test = np.column_stack([np.ones_like(y_test), X_test])
baseline_test_pred = A_test @ coeff
baseline_test_rmse = np.sqrt(np.mean((y_test - baseline_test_pred)**2))
baseline_test_r2 = 1 - np.sum((y_test - baseline_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
print(f'Baseline linear test: R2={baseline_test_r2:.4f}, RMSE={baseline_test_rmse:.4f}')

# === PySR Discovery ===
# Search for form: W = f(log10(P), V-I, [O/H])
# We add a constant term via the SR search
# Operators: +, -, *, /, ^2, ^3, sqrt, exp, log
feature_names = ['logP', 'VI', 'metal']

model = PySRRegressor(
    niterations=1000,
    populations=20,
    population_size=50,
    ncycles_per_iteration=300,
    maxsize=20,
    parsimony=0.005,
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

print("\n=== Starting PySR ===")
model.fit(X_train, y_train, variable_names=feature_names)

# === Results ===
print("\n=== Best models ===")
print(model)

# Evaluate on test set
for i, formula in enumerate(model.equations_.head(10)['sympy_format']):
    try:
        pred = model.predict(X_test, index=i)
        test_rmse = np.sqrt(np.mean((pred - y_test)**2))
        print(f"  Model {i}: {formula:.60s} -> test RMSE={test_rmse:.4f}")
    except:
        pass

# === Plot ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Period-luminosity (raw)
for h in hosts[:10]:
    mask = host_arr == h
    axes[0, 0].scatter(logP[mask], W[mask], s=3, alpha=0.6, label=h)
axes[0, 0].set_xlabel('log10(P) [days]')
axes[0, 0].set_ylabel('W (Wesenheit F160W)')
axes[0, 0].set_title('Raw PL relation (per host)')
axes[0, 0].invert_yaxis()
axes[0, 0].legend(fontsize=6, ncol=2)

# 2. Demeaned PL
for h in hosts[:10]:
    mask = host_arr == h
    axes[0, 1].scatter(logP_demeaned[mask], W_demeaned[mask], s=3, alpha=0.6, label=h)
axes[0, 1].set_xlabel('log10(P) - <log10(P)>_host')
axes[0, 1].set_ylabel('W - <W>_host')
axes[0, 1].set_title('Demeaned PL relation')
axes[0, 1].legend(fontsize=6, ncol=2)

# 3. Baseline fit residuals
residuals = y_train - baseline_pred
axes[1, 0].hist(residuals, bins=50, alpha=0.7)
axes[1, 0].set_xlabel('Residual (mag)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title(f'Baseline linear fit residuals (RMSE={baseline_rmse:.3f})')

# 4. SR model predictions
y_pred_sr = model.predict(X_test)
axes[1, 1].scatter(y_test, y_pred_sr, s=4, alpha=0.5)
axes[1, 1].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
axes[1, 1].set_xlabel('True (demeaned)')
axes[1, 1].set_ylabel('SR Predicted')
axes[1, 1].set_title(f'SR model (best)')
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.savefig('output/ceph_pl_sr.png', dpi=150)
print("\nSaved output/ceph_pl_sr.png")
