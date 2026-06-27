# SN Host Galaxy Mass Step — SR Discovery

## Hypothesis
The Type Ia supernova host galaxy mass step (a ~0.05 mag residual in
distance modulus correlated with host stellar mass) follows a smooth
functional form discoverable by symbolic regression, rather than the
ad-hoc step function currently used in all cosmological analyses.

## Crap-or-Worthwhile Test
**Current state:** SN Ia cosmology corrects for host galaxy environment
with a step function at log(M_host/M☉) = 10:
  μ_corr = μ_raw − γ × H(logM − 10)
where γ ≈ 0.05 mag. This is the most ad-hoc correction in the
standardization pipeline. The physical origin is debated (age,
metallicity, progenitor delay time distribution).

**If we succeed:** A discovered functional form ∆µ(logM_host) that:
- Is smoother than a step function (e.g., logistic, sigmoid, power law)
  would directly change every SN cosmology analysis (Pantheon+, DES,
  Union3, LSST)
- Correlates more strongly with sSFR, metallicity, or specific host
  properties than with mass alone would be a physical discovery about
  SN progenitors
- Would reduce systematic error in H0 and w measurements

**If we fail:** "Mass step is consistent with a step function at 10^10
M☉" → practice. Kill it.

**Verdict: WORTHWHILE** — the current step function is acknowledged as
ad-hoc by the community. A discovered functional form would change how
every SN collaboration handles this systematic.

## Data
- **Primary:** Pantheon+ (1590 SNe) with HOST_LOGMASS already included
  in the public data file
- **Secondary:** DES-SN5YR (1820 SNe) — host masses need cross-matching
  with DES host catalog
- **Variables:** HOST_LOGMASS, zHD, MU_SH0ES (or MU for DES), residuals
  from cosmological model
- **Selection:** z > 0.01, good quality cuts

## Method
1. Compute Hubble residuals from a reference cosmology
   (ΛCDM, Ωm=0.3, H0 fitted to the data)
2. Subtract best-fit absolute magnitude M (which absorbs the unknown
   H0 distance scale)
3. Fit residual ∆µ as a function of HOST_LOGMASS using SR:
   ∆µ = f(logM_host) + scatter
4. Baseline: step function with amplitude γ at logM = 10
5. Validation: bootstrap resampling, holdout by survey, split by
   low-z vs high-z

## Novelty
- Step function is universally used but known to be ad-hoc
- Any discovered smooth form would be a methodological improvement
- If form involves other host properties (sSFR, metallicity), it's a
   physical discovery about SN progenitors

## Phases
1. Compute residuals and explore the mass step
2. SR discovery of f(logM_host)
3. Validation (bootstrap, holdout, split samples)
4. Comparison with step function (ΔBIC, ΔAIC)
5. Extension to sSFR / metallicity / other host properties
6. Adversarial Debate
7. Publication
