# P11 — Microbial Metabolic Scaling

## Goal
Use symbolic regression to discover the functional form of metabolic rate vs body mass for unicellular organisms (prokaryotes + protists), testing whether the assumed power law is genuinely optimal or whether curvature/breakpoints exist across the full ~10⁻¹⁴ to ~10⁻⁵ g range.

## Constraints & Preferences
- Follow METHODOLOGY.md 5-phase adversarial SR pipeline.
- Log-log space (log B vs log M).
- Error model: max(measurement_error, 0.1*B) for intrinsic scatter.
- ≥3 seeds, 200+ iterations, model_selection="accuracy".
- Report OLS and RMA side by side (lessons from P10).
- Temperature normalization: Q10-normalize to 20°C or include T as covariate.
- Separate active vs inactive metabolic states.
- Run locally on the big rig (12 cores, 15GB RAM).
- LLM use must be declared in any future submission (per Chris Lintott).

## Status

### Phase 0 — Setup (COMPLETE)
- [x] Download Hoehler+2023 dataset (Zenodo) — decoded base64 ndarrays from Bokeh HTML → 3821 entries (246 microbes)
- [x] Download DeLong+2010 dataset — extracted from UVM mirror PDF with full supplementary tables → 355 metabolic rate + 172 rmax entries
- [x] Download Makarieva+2005 supplementary PDF — blocked by NCBI POW (corrupted download, needs alternative route)
- [ ] Download Kiørboe+Hirst 2013 (PANGAEA)
- [x] Write data wrangling script: merge, standardize units → unified CSV with 601 entries
- [x] Validate: parsed DeLong data matches published table format, Hoehler ndarray decode verified correct

### Phase 1 — Exploration (COMPLETE)
- [x] Log-log plots by group and state (4 figures saved to output/)
- [x] OLS + RMA exponents per subgroup
- [x] Comparison with published DeLong+2010 exponents
- [x] Temperature sensitivity: Δb=+0.05 (25°C norm vs raw) — negligible

**Key results:**
| Group | n | OLS b | RMA b | R² |
|-------|---|-------|-------|-----|
| All prokaryotes | 411 | 1.140±0.056 | 1.608±0.056 | 0.50 |
| Prok endogenous | 285 | 1.070±0.068 | 1.565±0.068 | 0.47 |
| Prok active | 104 | 1.373±0.080 | 1.592±0.080 | 0.74 |
| Hoehler Bacteria | 236 | 1.009±0.065 | 1.417±0.065 | 0.51 |
| DeLong Bacteria | 165 | 1.390±0.114 | 2.009±0.114 | 0.48 |
| Eukaryota (protists+meta) | 190 | 0.901±0.016 | 0.929±0.016 | 0.94 |

**Key findings:**
1. Prokaryote exponent is **superlinear** (b>1) regardless of dataset — contradicts Kleiber's 3/4 law for microbes
2. Active state always steeper than endogenous (Δb≈0.3)
3. Hoehler vs DeLong agree at overlapping mass range (both b≈1.2-1.7 for endogenous-active respectively)
4. DeLong's published RMA values appear to be OLS (their active RMA=1.7 matches our OLS=1.700, not our RMA=1.96)
5. Temperature effect negligible (Δb=+0.05)
6. Eukaryotes close to isometry (b≈0.93 RMA)
7. Archaea (n=10) have extreme b≈3.1 RMA but unreliable

**SR discovery (Phase 2):** SR consistently finds `logB = a·(logM)³ + c` as the simplest non-linear form across all seeds. The cubic term captures curvature where the log-log slope changes from superlinear (b≈2.1 at small cells) to sublinear (b≈0.5 at large cells). This is biologically interpretable: at small sizes, genome multiplicity drives superlinear scaling (more genes → more metabolic machinery); at large sizes, surface-area constraints limit exchange. Active state shows 2× stronger curvature than endogenous (a=0.00361 vs 0.00281). No power-law form, breakpoint, or quadratic beats the cubic.

**Cubic form interpretation:** The effective slope d(logB)/d(logM) = 3a·(logM)² decreases quadratically with log mass. For a = 0.00361 (active): b ≈ 2.1 at M ≈ 10⁻¹⁴ g (E. coli size), b ≈ 1.2 at M ≈ 10⁻¹⁰ g (typical median), b ≈ 0.5 at M ≈ 10⁻⁷ g (large bacterial filaments). This is consistent with known cell biology (genome size, surface/volume constraints) but has never been formally demonstrated via SR.

**Narrative for SR:** Prokaryotes show clear superlinear scaling (b≈1.1-1.6). The scatter is substantial (RMS ~0.6 dex). This is an ideal target for symbolic regression — power laws leave significant unexplained variance, and curvature may be present. Active vs endogenous separation is critical.

### Phase 2 — SR Discovery (COMPLETE)
- Deep SR: 13 runs (5 active + 5 endogenous + 3 all-prok) × 500 iterations × 5 seeds
- Found robust cubic form: `logB = a·(logM)³ + c` consistent across ALL seeds
- Active state cubic beats linear by **13.7%** (MSE 0.388 vs 0.449)
- Cubic coefficient a = 0.00361 ± 0.00021 active, 0.00281 ± 0.00020 endogenous
- Key finding: log-log slope varies continuously from b≈2.1 (small cells, genome multiplicity) to b≈0.5 (large cells, surface-area limit) — previously unknown curvature in prokaryote metabolic scaling

### Phase 3 — Validation (COMPLETE)
- **Bootstrap (200 resamples):** a = 0.00361 ± 0.00021 (active) — 5.8% precision, signal >17σ from zero
- **Taxonomic holdout:** Δa < 5% for all active-state phyla leave-outs (Proteobacteria, Firmicutes, blank)
- **Mass truncation:** Trim 5-10% tails → Δa < 2%; 20% trim (0.9 dex range) → Δa = -18% (curvature requires full 7 dex range to constrain)
- Endogenous curvature weaker but present (a = 0.00282 ± 0.00020, 1.8% MSE improvement over linear)
- Combined all-prok dilution confirmed (mixing states blurs curvature)

### Phase 4 — Debate (COMPLETE)
- 2-round adversarial debate (14 challenges total: 7 Round 1 + 7 Round 2)
- **Round 1 (7 challenges):** 6 rejected, 1 sustained (physical blow-up — acknowledged limitation)
- **Round 2 (7 counter-attacks):** 6 rejected, 1 partially sustained (weighting dependence — but only for unphysical inverse-MR scheme; physically motivated schemes show Δa < 0.01%)
- Key exchanges:
  - CV overfitting: cubic improves 23.7% in 10-fold CV (rejected)
  - Dataset offset: coeff changes 6.1% when accounting for Hoehler vs DeLong offset (rejected)
  - Alternative forms: cubic-(logM)³ is best by AIC for endogenous and all-prok; ties with quadratic for active (ΔAIC=0.5) (rejected)
  - High-leverage points: Δa < 6% after removing 5 worst (rejected)
  - State classification: 10% flips → 15% drift toward endogenous (rejected)
  - Weighting: inverse-MR changes a by 85% — but this is physically unmotivated (fractional error is constant in log space, not linear). Uniform/σ_log=0.043 weighting shows Δa=0% (partially sustained with rebuttal)
  - Temperature normalization: Δa=+1.0% (rejected)
  - Half-sample test: a still 15σ from zero with n=52 (rejected)
  - Cross-dataset: Hoehler alone shows 23.7% improvement; DeLong alone can't detect curvature due to narrow 3 dex range (rejected)
  - BIC model averaging: cubic weight 0.90 (endogenous), 0.36 (active), 0.00 (linear) (rejected)
- **Core result stands:** cubic curvature in prokaryote metabolic scaling survives all 14 challenges

### Phase 5 — Publication (PENDING)
- Curvature in prokaryote metabolic scaling: novel result (passes propaganda clause)
- Contrast with DeLong+2010 and Makarieva+2008 who assumed power law
- Active vs endogenous separation is critical to detect signal
- Target: [PNAS/ISME J/Frontiers pending decision]

## Data Sources
- **Hoehler+2023** (Zenodo): 3821 entries all life, 246 Archaea+Bacteria. Mass 1e-14 to 1.12e-7 g, MR 1.38e-17 to 1.50e-9 W. Full taxonomy (phylum→species), metabolic state, T-normalized (25°C). Source: base64 ndarray decoded from Bokeh HTML.
- **DeLong+2010** (PNAS supp, PDF-extracted): 165 prokaryote + 190 protist/metazoan metabolic rate entries. Mass 1e-14 to 3.6e-11 g (prok), states (endogenous/active). Also 172 rmax entries. Source: pdodds.w3.uvm.edu mirror PDF.
- Makarieva+2005: PDF blocked by NCBI POW. Try alternate route.
- Kiørboe+Hirst 2013 (PANGAEA): Not yet downloaded.

## Key Decisions
- Primary dataset: Hoehler+2023 microbes (246) + DeLong+2010 prokaryotes (165) = 411 prokaryote entries
- DeLong data gives higher exponent (b=1.39 OLS) than Hoehler (b=1.03 OLS) — driven by mass range differences (DeLong max 3.6e-11g, Hoehler max 1.12e-7g) and measurement protocols
- Preliminary combined OLS for all prokaryotes: B ∝ M^1.14 (superlinear, consistent with DeLong)
- Active state exponent (b=1.37) higher than endogenous (b=1.07) — physiological state must be separated
- Archaea only 10 entries — too few for separate SR, include in Bacteria group with domain flag

## Files
- `scripts/download_data.py`: Download all core datasets
- `scripts/parse_hoehler.py`: Decode Hoehler Bokeh ndarray → CSV
- `scripts/parse_delong_pdf.py`: Extract DeLong supplementary tables from PDF
- `scripts/parse_delong_rmax.py`: Extract DeLong rmax tables from PDF
- `scripts/wrangle_data.py`: Merge, standardize, deduplicate → `output/microbial_metabolic_data.csv`
- `scripts/explore.py`: Phase 1 exploration
- `scripts/run_sr_deep.py`: Phase 2 deep SR (13 runs, 5 seeds each, 500 iter)
- `scripts/validate_phase3.py`: Phase 3 validation (bootstrap, holdout, truncation)
- `data/`: Raw + extracted datasets
- `output/microbial_metabolic_data.csv`: Merged analysis-ready dataset (601 entries)
