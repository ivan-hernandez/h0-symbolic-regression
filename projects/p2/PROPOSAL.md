# P2: GW Compact Object Mass Distribution — SR Discovery

## Hypothesis
The mass distribution of merging compact objects (BBH, BNS, NSBH) follows
a functional form discoverable by symbolic regression, which may reveal
the underlying astrophysical processes (supernova engines, pair-instability
gap, neutron star maximum mass) without assuming ad-hoc broken power laws
or Gaussian peaks.

## Crap-or-Worthwhile Test
**Current state:** Mass distribution modeled as "Power Law + Peak" (Talbot
& Thrane 2018) with 7+ parameters: a power law with a Gaussian bump at
~35 M_sun and optional breaks. Form is phenomenological, not physically
derived.

**If we succeed:** A discovered functional form with fewer parameters that
fits equally well or better would:
- Provide a more parsimonious description of the mass distribution
- Suggest physical mechanisms (e.g., if form is a smooth Schechter-like
  function with a cutoff, it constrains pair-instability supernova physics;
  if it requires a Gaussian peak, supports the "pulsational pair-instability"
  scenario)
- Directly constrain population synthesis models

**If we fail:** "Mass distribution is consistent with Power Law + Peak" →
practice (do not publish).

**Verdict: WORTHWHILE** — a discovered form with physical interpretability
would change how GW data is analyzed and how population models are built.

## Data
- Source: GWTC-3 from GWOSC (Gravitational Wave Open Science Center)
- Columns: m1 (primary mass), m2 (secondary mass), SNR, FAR, etc.
- Cut: FAR < 0.01 (confident detections), m1 > 0
- Size: ~200 events (GWTC-3 confident BBH+BNS+NSBH)

## Phases
1. Discovery (SR on primary mass PDF)
2. Validation (bootstrap resampling, holdout by observing run)
3. Extension (test on GWTC-2 vs GWTC-3 separately)
4. Adversarial Debate
5. Publication

## Constraints
- Fit the PDF of m1, not the joint m1-m2 distribution
- Compare to Power Law + Peak baseline (from LVK publications)
- Propaganda clause: if result is "consistent with literature," kill it
