# Hubble Constant from Symbolic Regression

**DOI:** [10.5281/zenodo.20778035](https://zenodo.org/doi/10.5281/zenodo.20778035) (v3 — adversarially validated)

Discover the cosmic expansion history H(z) using symbolic regression (PySR) on
CC + BAO + DESI DR1 + Pantheon+ data, with minimal theoretical priors.

**Result:** H0 = 68.0 ± 0.8 km/s/Mpc (68% CL), consistent with Planck at 1.2σ.
The Hubble tension is a >8σ discrepancy in the Cepheid calibration of the
supernova absolute magnitude M, not the expansion history shape.

## Requirements

- Python ≥ 3.9 with numpy
- [PySR](https://github.com/MilesCranmer/PySR) (for SR discovery; the joint fit
  scripts do not require PySR)
- 15+ GiB RAM for the Pantheon+ full covariance inversion (cached after first run)
- Internet connection for data downloads (cached after first run)

## Quick Start

```bash
# Reproduce the main result
python3 joint_rank.py                     # SR ranking + best fit
python3 pantheon_cov.py                   # Pantheon+ full covariance fit
python3 des_sn5yr.py                      # DES-SN5YR cross-check
python3 profile_h0.py                     # H0 profile likelihood
python3 lcdm_fit.py                       # ΛCDM comparison fit
python3 reject_all.py                     # full objection test suite
python3 validate_all.py                   # validation checks
```

## Files

| Script | Purpose |
|--------|---------|
| `joint_rank.py` | Joint CC+BAO+DESI+SNe ranking |
| `pantheon_cov.py` | Pantheon+ data with full 1590×1590 covariance |
| `des_sn5yr.py` | DES-SN5YR data with full 1820×1820 precision matrix |
| `profile_h0.py` | H0 profile likelihood |
| `lcdm_fit.py` | ΛCDM fit for comparison |
| `reject_all.py` | Comprehensive objection test suite |
| `validate_all.py` | Validation checks |
| `sh0es_objections.py` | Targeted SH0ES objection responses |
| `bootstrap_h0.py` | Bootstrap H0 uncertainty |
| `marginalize_rd.py` | r_d marginalization |
| `hubble_pilot.py` | Main SR discovery script (requires PySR) |
| `paper/paper.tex` | LaTeX paper draft |

## Validation Summary

| Check | Result |
|-------|--------|
| 8 independent SR seeds | Same functional form |
| 4 SN samples | All converge to H0≈68 |
| 3 BAO configurations | Consistent |
| Full covariance (Pantheon+, DES-SN5YR) | H0 shifts <1 km/s |
| Fix M to SH0ES | Δχ²=+64–82 (8–9σ rejected) |
| Taylor expansion | H0=67.4 (identical) |
| No CC data | H0=68.2–68.6 |
| ΛCDM comparison | Joint χ² differs by Δχ²=1.2 |
| Binned residuals | Flat, no z-trend |
| Bootstrap | H0=66.2±3.1 (CC+BAO+DESI only) |

## Cached Data

- `pantheon_cov_cache.npz` — Pantheon+ covariance inverse (32 MB download, 5 s invert)
- `DES_SN5YR_cache.npz` — DES-SN5YR covariance inverse (6 MB download)
- Both cached at `/tmp/`
