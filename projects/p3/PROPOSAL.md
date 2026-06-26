# P3: Compact Object Mass Function — Do Observations Require Singularities?

## Question
GR singularity theorems say singularities are inevitable under mild energy
conditions.  But *real* observations may not require them.  If nature avoids
singularities, objects near the TOV limit (~2.3 M☉) might not collapse to
black holes.  Instead they could form non-singular configurations (quark
stars, boson stars, gravastars) with a different mass distribution than GR
predicts.

**Null hypothesis (GR):** The compact object mass function is a smooth
power law across the NS–BH boundary, with no sharp feature near 2–3 M☉
other than observational selection effects.

**Alternative (singularities not required):** There is a gap, pileup,
cutoff, or other feature near the TOV limit that GR alone does not explain.

## Data
- GWTC-3: 184 confident events (179 BBH, 1 BNS, 4 NSBH) — already downloaded
- Radio pulsar masses: compiled from literature (Özel+2016, Antoniadis+2016,
  Fonseca+2021, etc.)
- X-ray binary dynamical masses
- NICER mass measurements (J0030, J0740)

## Analysis
1. Build histogram / KDE of mass function from all compact objects
2. SR search for functional form P(m) = f(m; θ)
3. Test for features near 2–3 M☉ (gap, step, pileup)
4. Bootstrap uncertainty on feature location and significance

## Predictions
- **GR wins:** smooth power law + Gaussian component best fits the data,
  no significant feature near TOV limit
- **Alternatives win:** a gap, pileup, or cutoff near 2–3 M☉ is required
  by the data at >3σ

## Caveats
- Selection effects dominate the observed mass function (detector sensitivity,
  Malmquist bias, beaming for radio pulsars, etc.)
- The "mass gap" between NS (~2.1 M☉) and BH (~5 M☉ for XRBs, ~3 M☉ for GW)
  is already documented in the literature
- We are testing whether the *combined* mass function of ALL compact objects
  shows features consistent with alternatives to singularities
