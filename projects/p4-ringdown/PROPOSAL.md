# P4: Ringdown Null Test with Symbolic Regression

## Motivation
GR predicts Kerr black hole ringdown frequencies are uniquely determined by mass M and spin χ via the no-hair theorem. Any deviation δ = ω_obs/ω_Kerr - 1 ≠ 0 would falsify GR. With ~100 BBH events and growing, this is testable.

## Method
1. Extract posterior samples for each high-SNR event: (M_f, χ_f, ω_{220}, τ_{220})
2. Compute Kerr prediction M·ω_{220} = f(χ) using `qnm` Python package
3. Form dimensionless deviation δ = (ω_obs - ω_Kerr)/σ_ω
4. SR search: δ = F(M_f, χ_f, q, χ_eff, ...) — if F ≠ 0, evidence for GR deviation
5. Validation: multi-seed bootstrap, holdout, adversarial debate

## Data Sources
- **GWTC-3 Tests of GR** ringdown posteriors: Zenodo DOI 10.5281/zenodo.17461225
- **GWOSC API** for event metadata: https://gwosc.org
- **qnm** package for Kerr QNM frequencies

## Crap-or-Worthwhile Test
- If δ consistent with zero across all parameters → null result confirms GR, novel as first SR application to ringdown and first systematic search for parameter-dependent deviations
- If δ ≠ 0 → discovery of GR violation → extremely novel
- **Verdict: worthwhile in either outcome**

## Success Criteria
- At least 10 high-SNR events (ringdown SNR > 8) with well-measured (M_f, χ_f)
- SR finds no functional form for δ at >95% confidence → GR confirmed
- OR SR finds δ ≈ F(X) at >3σ → evidence for deviation
- Result robust to: seed variation, bootstrap, holdout fraction
