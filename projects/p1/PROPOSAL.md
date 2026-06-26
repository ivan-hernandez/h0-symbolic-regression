# P1: Exoplanet Mass-Radius Relation — Symbolic Regression Discovery

## Hypothesis
The exoplanet mass-radius relation follows a continuous functional form
discoverable by symbolic regression, which may reveal physically meaningful
transition regimes between rocky, volatile-rich, and gas giant planets.

## Crap-or-Worthwhile Test
**Current state:** M-R relation modeled as ad-hoc broken power laws
(Rogers 2015, Chen & Kipping 2017, Otegi+ 2020) with arbitrary transition
boundaries. Categories (rocky, super-Earth, sub-Neptune, gas giant) are
theory-motivated bins, not empirically discovered.

**If we succeed:** A discovered functional form provides the first
model-agnostic, data-driven M-R relation. Could reveal:
- Continuous vs discontinuous transitions
- Whether there are 2, 3, or 4 distinct populations
- Where the true transitions occur without assuming theory priors

**If we fail:** "M-R relation is consistent with literature" → practice
(do not publish as standalone).

**Verdict: WORTHWHILE** — a discovered functional form changes how
exoplanet surveys classify planets and plan follow-up.

## Data
- Source: NASA Exoplanet Archive confirmed planets table
- Columns: mass (M_Earth or M_Jup), radius (R_Earth or R_Jup), mass_err, radius_err, detection method
- Cut: planets with both mass AND radius measurement, mass > 0.1 M_Earth
- Size: ~1000+ planets

## Phases
1. Discovery (SR on log M vs log R)
2. Validation (bootstrap, holdout by detection method)
3. Extension (test on independent samples, compare to theory)
4. Adversarial Debate
5. Publication

## Constraints
- Log-log space for dynamic range (0.1 M_E to 10 M_Jup, ~0.1 R_E to 2 R_Jup)
- Intrinsic scatter model: error = max(measurement_error, 0.1*y)
- Multi-seed: ≥3 seeds, 200 iterations each
- Propaganda clause: if result is "consistent with literature," kill it
