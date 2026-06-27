# P11 — Microbial Metabolic Scaling via Symbolic Regression

## Crap-or-Worthwhile Test

> If the conclusion changes what someone would assume or do, it's novel.
> If it says "consistent with literature," it's practice.

**Verdict: NOVEL.** The field universally assumes power-law scaling (B ∝ M^α) in log-log space and debates only the exponent value (0.6 to 1.7). No study has used symbolic regression to ask whether the true functional form is a power law at all. Growing evidence for curvature (Ritchie-Kempes 2024, Li & Wang 2019, Kolokotrones 2010) suggests breakpoints or non-power-law forms may exist. SR would either confirm the power-law assumption (novel test) or discover new functional forms (novel result). Either outcome changes the evidentiary basis of MTE theory.

## Hypothesis

The metabolic scaling of unicellular organisms (bacteria + protists) across their full mass range (~10⁻¹⁴ to ~10⁻⁵ g, 9 orders of magnitude) is not a pure power law. Either:

- **H1: Curvature.** Log-log scaling is nonlinear — the exponent varies with body mass (steeper for small prokaryotes, shallower for large protists), consistent with surface-to-volume constraints or diffusion-limitation transitions.
- **H2: Breakpoint.** A single continuous function describes both prokaryotes and protists but with a transition region near ~10⁻¹⁰ g where the dominant physical constraint changes (e.g., diffusion → bulk flow → transport).
- **H0: Power law suffices.** A single exponent across the full range describes the data as well as any more complex form.

## Data Sources (all open access)

| Dataset | n (microbes) | Mass range | Temperature? | Source |
|---------|-------------|------------|--------------|--------|
| DeLong+2010, PNAS | ~170 (prok+protist) | ~10⁻¹⁴–10⁻⁵ g | Some | pnas.org/suppl/doi:10.1073/pnas.1007783107 |
| Makarieva+2005, Proc B | ~80 (prokaryotes) | ~10⁻¹⁴–10⁻⁸ g | Yes | royalsocietypublishing.org/doi/10.1098/rspb.2005.3225 |
| Hoehler+2023, PNAS | ~500+ microbial subset | Full range | Yes (Q10) | zenodo.org/doi:10.5281/zenodo.7877885 |
| Kiørboe+Hirst 2013, PANGAEA | ~100+ marine microbes | ~10⁻¹⁰–10⁻⁵ g | Yes | doi.pangaea.de/10.1594/PANGAEA.819857 |
| Kempes+2012, PNAS | ~50 | ~10⁻¹⁴–10⁻⁸ g | Some | pnas.org |

## Pipeline

| Phase | Task | Deliverable |
|-------|------|-------------|
| 0 | Data wrangling | Clean merged dataset (n ≥ 300, standardized units, T-normalized) |
| 1 | Exploration | Log-log plots by group, OLS/RMA exponents, comparison with literature |
| 2 | SR Discovery | PySR on full range + per group, model selection by accuracy |
| 3 | Validation | Bootstrap (200×), holdout by taxon, temperature sensitivity |
| 4 | Debate | Adversarial challenge round |
| 5 | Publication | Paper + data release + Medium article |

## Key Methodological Decisions

- **SR in log-log space** (log B vs log M), consistent with P10
- **Error model:** max(measurement_error, 0.1 × B) for intrinsic scatter
- **Temperature normalization:** Include T as covariate OR Q10-normalize to 20°C
- **State separation:** Active vs inactive metabolic rates analyzed separately
- **Regression method:** Report both OLS and RMA side by side (lessons from P10)
- **PySR:** model_selection="accuracy", ≥3 seeds, 200+ iterations, procs=12
- **Weak boundary prior** on zero intercept to suppress pathology

## References (key)

- DeLong JP, et al. (2010) PNAS 107:12941. doi:10.1073/pnas.1007783107
- Makarieva AM, et al. (2005) Proc R Soc B 272:2327. doi:10.1098/rspb.2005.3225
- Hoehler TM, et al. (2023) PNAS 120:e2217836120. doi:10.1073/pnas.2217836120
- Ritchie ME, Kempes CP (2024) arXiv:2405.xxxxx. doi:10.1101/2024.05.xxx
- Li Y, Wang G (2019) Sci Rep 9:15418. doi:10.1038/s41598-019-51829-2
- Kolokotrones T, et al. (2010) Nature 464:753. doi:10.1038/nature08920
- Glazier DS (2005) Biol Rev 80:611. doi:10.1017/S1464793105006834
- López-Urrutia Á, et al. (2006) Nature 443:967. doi:10.1038/nature05226
