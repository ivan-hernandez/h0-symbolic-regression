# P10: Kleiber's Law — Symbolic Regression on Metabolic Allometry

## Hypothesis
The 90-year debate over whether metabolic rate scales as M^2/3, M^3/4, or something more complex (curvature, clade-specific exponents, regime-dependent) is resolvable by symbolic regression on compiled public data, which can discover the functional form without assuming a power-law template.

## Background
**Kleiber's Law** (Kleiber 1932) states BMR ∝ M^3/4, taught as a biological law for 90 years. Three competing camps:
1. **Surface law (2/3):** Rubner (1883), White & Seymour (2003, PNAS) — heat dissipation through surface area
2. **Quarter-power (3/4):** Kleiber (1932), West, Brown & Enquist (1997, Science) — fractal nutrient networks
3. **Curvature:** Kolokotrones et al. (2010, Nature) — quadratic in log-log improves fit; curvature at high masses
4. **Clade-dependent:** Mammals, birds, reptiles, insects show systematically different exponents (McNab 2008, 2025)

A 2026 Annual Review on metabolic scaling was just published — the debate remains unresolved.

## Crap-or-Worthwhile Test
**Current state:** The field is split between 2/3, 3/4, curvature, and clade-specific models with no consensus. Each camp uses their preferred data subset and model template (power law with fixed exponent).

**If we succeed:**
- A discovered functional form with fewer assumptions than any existing model would:
  - End the 90-year debate if a simple universal form fits all data
  - Quantify whether curvature is real (Kolokotrones) or a phantom of phylogenetic covariance
  - Provide clade-specific forms with uncertainty, usable for paleobiology and climate modeling
  - Reveal whether endotherms and ectotherms share a common form with different parameters, or fundamentally different forms

**If we fail:** "Metabolic scaling is consistent with a power law" → practice (do not publish).

**Verdict: WORTHWHILE** — A discovered form that either resolves the exponent debate or proves the field needs clade-specific models would change how comparative physiology, paleobiology, and climate science use allometric scaling.

## Data
**Primary:** AnimalTraits database (Herberstein et al. 2022, Scientific Data, CC0 license)
- ~2000 terrestrial animal species, ~600 with metabolic rate
- Taxonomy phylum→species, thermoregulation group (endotherm/ectotherm), measurement method
- Body mass (g), metabolic rate (W), brain size (g)
- Zenodo DOI: 10.5281/zenodo.6468938

**Secondary (cross-validation):**
- Kolokotrones et al. (2010) mammal BMR data (Nature, supplementary info)
- FmrBT field metabolic rates (de Castro 2025, Scientific Data, CC0) — 700+ species
- McNab (2008) mammalian BMR compilation — 746 species

## Phases
1. Discovery (SR on log BMR vs log M)
   - All animals; mammals only; endotherms vs ectotherms; by class
2. Validation (bootstrap, holdout by class, phylogenetic signal in residuals)
3. Extension (cross-data validation, comparison to literature exponents)
4. Adversarial Debate
5. Publication

## Constraints
- Log-log space for dynamic range (mg to tonnes, ~10 orders)
- Error model: max(measurement_error, 0.1*y) for intrinsic scatter
- ≥3 seeds, 200+ iterations, model_selection="accuracy"
- Account for thermoregulation group (endotherm vs ectotherm) as potential bifurcation
- Propaganda clause: if result is "consistent with literature," kill it

## Literature
- Kleiber (1932) — M^3/4 origin
- West, Brown & Enquist (1997, Science) — fractal network theory
- White & Seymour (2003, PNAS) — M^2/3 with phylogenetic correction
- Kolokotrones et al. (2010, Nature) — curvature in log-log
- McNab (2008, Comp Biochem Physiol) — clade-specific exponents
- Herberstein et al. (2022, Scientific Data) — AnimalTraits database
- de Castro (2025, Scientific Data) — FmrBT field metabolic rates
