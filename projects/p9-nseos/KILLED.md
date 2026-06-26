# P9: Neutron Star Equation of State — KILLED (Practice)

**Date:** 2026-06-26
**Result:** Currently set up to fit RMF theory models (DDH, APR) with SR — circular re-parametrization of known functional forms. Would be worthwhile only if redirected to observational EOS constraints.

## Crap-or-Worthwhile Assessment
**Question:** Does this change what someone would assume or do?

**As-is (theory models):** No. Fitting a theory model (RMF parameterizations of nuclear matter) with SR is circular — you're just re-discovering the functional form that was used to generate the model. The DDH models already have known parametric forms within RMF theory. SR discovering a different parametrization doesn't change what anyone assumes about NS structure.

**If redirected (observational):** Possibly. Using NICER mass-radius posteriors, GW170817 tidal deformability, and pulsar mass measurements to discover P(ε) directly from data would be novel. But this requires:
- Full forward model (TOV solver)
- Bayesian inference (not chi2 minimization)
- This is a full MCMC pipeline, not a simple SR fit

**Verdict: PRACTICE** as-is. Kill. If observational constraints become available in a direct form (P-ε points with errors), reconsider.

## Scripts
- `eos_data.py`: Load DDH/APR/SLY/APS tables
- `eos_sr_remote.py`: SR on P(ε) or P(n_b)
- `plot_eos_results.py`: Comparison plots

## Data
- DDH models (DDBu1, DDBu2, DDBl, DDBm, DDBx): RMF parameterizations
- APR: Akmal-Pandharipande-Ravenhall EOS
- SLY: Skyrme EOS
- DDH posterior extraction (10k samples): training set for emulator
