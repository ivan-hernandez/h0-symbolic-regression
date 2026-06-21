# Galactic Rotation Curves — Symbolic Regression

Symbolic regression (PySR) on the SPARC Radial Acceleration Relation to discover the functional form \(g_{\rm obs} = F(g_{\rm bar})\) with minimal theoretical priors.

## Result

**CPX5:** \(\log g_{\rm obs} = a + b/\log g_{\rm bar}\) with \(a = -17.060 \pm 0.133\), \(b = -72.71 \pm 1.38\) (bootstrap).
Beats all MOND interpolating functions (\(\Delta\)AIC = 88, \(\Delta\chi^2 \approx 960\)).
MOND \(\sqrt{g_{\rm bar}}\) asymptote not required by kinematic data (\(c = 0.10 \pm 0.15\), \(\Delta\chi^2 = 0.18\), \(p = 0.67\)).

## Paper

- **PDF:** `paper/paper.pdf`
- **LaTeX source:** `paper/paper.tex`
- **Zenodo:** [10.5281/zenodo.20788781](https://zenodo.org/records/20788781)
- **Repo:** [github.com/ivan-hernandez/h0-symbolic-regression](https://github.com/ivan-hernandez/h0-symbolic-regression)

## Data
- SPARC: 175 late-type galaxies, 3,391 rotation curve points
- `MassModels_Lelli2016c.mrt`: Combined table (R, Vobs, Vgas, Vdisk, Vbul, SBdisk, SBbul, D)
- `RAR.mrt`: Pre-computed Radial Acceleration Relation (gbar, gobs)
- Mistele+2024 weak-lensing data (Table 1, 11 binned points, extends RAR by 2.5 dex)

## Approach
1. Parse SPARC mass models → compute gbar, gobs
2. PySR search for g_obs = F(g_bar) in log-log space
3. Multi-seed validation (3 independent seeds)
4. Compare with MOND interpolating functions (McGaugh RAR IF, Simple, Standard)
5. Validation: bootstrap (200 resamples), holdout (10-fold), M/L grid (16 combinations), blind MOND recovery
6. Joint SPARC + Mistele lensing PySR (6.5 dex dynamic range)
7. MOND asymptote test, EFE (3D SIMBAD), gas fraction systematic

## Key Files
| File | Purpose |
|------|---------|
| `rar_sr.py` | Main PySR search on SPARC RAR |
| `rar_analysis.py` | MOND fit, SR comparison, holdout, property dependence |
| `rar_linear.py` | Linear-space SR |
| `extend_rar_analysis.py` | EFE proxy, per-galaxy CPX5, bootstrap |
| `efe_coordinates.py` | SIMBAD 3D EFE analysis |
| `joint_pysr_lensing.py` | PySR on combined SPARC + lensing (6.5 dex) |
| `joint_sr_lensing.py` | Deterministic model comparison on joint data |
| `test_mond_asymptote.py` | MOND √gbar asymptote test |
| `gas_fraction_analysis.py` | Gas fraction systematic investigation |
| `hook_search.py` | FIRE-2 non-monotonic RAR track search |
| `literature_comparison.py` | Compare with Desmond+2023, McGaugh+2016, Mistele+2024 |
| `simulation_comparison.py` | Compare with EAGLE, IllustrisTNG, FIRE-2 |
| `parse_sparc.py` | SPARC data parser and acceleration computation |
| `paper_figures.py` | Generate publication-quality figures |
| `paper/paper.tex` | Full paper (aastex631.cls, two-column) |

## Key References
- Desmond+2023: MNRAS 521, 1817 — "On the functional form of the radial acceleration relation"
- McGaugh+2016: PRL 117, 201101 — Original RAR discovery
- Lelli+2016: AJ 152, 157 — SPARC data paper
- Mistele+2024: JCAP 04, 020 — Weak-lensing RAR extension
- Ardizzone+2023: A&A 672, A118 — FIRE-2 RAR hooks
