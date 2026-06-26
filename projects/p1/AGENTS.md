# P1: Exoplanet Mass-Radius Relation — SR Discovery

## Goal
Discover the functional form of the exoplanet mass-radius relation
using symbolic regression on NASA Exoplanet Archive data.

## Template
Follow METHODOLOGY.md (5-phase adversarial SR pipeline).

## Constraints
- Log-log space for SR (log R vs log M)
- Error model: max(measurement_error, 0.1 * y) for intrinsic scatter
- ≥3 seeds, 200 iterations, model_selection="accuracy"
- Propaganda clause: if result is "consistent with literature," do not publish

## Progress
### Phase 1 — Data & Discovery (COMPLETE)
- Downloaded 3398 confirmed planets from NASA Exoplanet Archive (ps table)
- Filtered to 3086 with mass > 0.1 M_E, positive radius, and error bars
- Data range: 0.065 M_E to 25426 M_E, 0.3 R_E to 87 R_E
- 3 SR seeds (42, 123, 456) each with 200 iterations

### Phase 2 — Discovery Results
All 3 seeds agree on single power law C3 (the only stable form):
**log R = 0.332 * log M** → **R ∝ M^0.33** (PySR with accuracy selection)
**log R = 0.388 * log M + 0.097** (OLS, R² = 0.79, RMSE = 0.183 dex)

| Model | Params | R² | RMSE | Δ from PL |
|-------|--------|-----|------|-----------|
| Power law (C3) | 2 | 0.79 | 0.183 dex | — |
| Broken PL | 4 | 0.79 | 0.182 dex | 0% |
| Triple PL | 6 | 0.87 | 0.147 dex | -19% |
| Quadratic | 3 | 0.86 | 0.152 dex | -17% |

Key findings:
- **Broken PL is NOT preferred over single PL** (break at 0.4 M_E = noise)
- **Triple PL improves R² from 0.79 to 0.87** — modest gain for 4 extra params
- **Intrinsic scatter dominates**: RMSE = 0.18 dex = ±50% at fixed mass
- Higher-complexity SR forms diverge between seeds — overfitting
- The C3 power law is the only robust, seed-independent form

### Phase 3 — Crap-or-Worthwhile Assessment
**Question:** Does this change what someone would assume or do?

**Negative finding:** The M-R relation is fundamentally a ~0.18 dex scatter relation
regardless of model complexity. Composition and evolution matter more than the
mean power-law index. This is underappreciated — most papers optimize the mean
relation without quantifying the irreducible scatter.

**But:** A single power law with ~0.3-0.4 slope is fully "consistent with literature."
The triple-PL improvement is real but modest. Neither result is novel enough to
change what people assume.

**Verdict: PRACTICE.** The power-law result confirms known literature. The
scatter dominance is mildly interesting but not actionable for the field.

**Do not publish as standalone.** Move to P2.

### Done
- Phase 1 (Data download, verification, exploration)
- Phase 2 (3-seed SR discovery, model comparison)
- Phase 3 (Crap-or-worthwhile assessment → practice, kill)

## Remote Compute
- Heavy computation on 100.121.64.70 (Tailscale SSH, 12 cores, 15 GB RAM)
- Julia 1.11.9 installed at ~/julia/
- PySR 1.5.10 via pip

## Key Decisions
- Use M_Earth and R_Earth units for interpretability
- Filter: confirmed planets with mass > 0.1 M_Earth and radius measured
- Log-log space to avoid high-mass domination of MSE loss
- Propaganda clause enforced: P1 result is practice, not novel — killed
