# P13: Species-Area Relationship — Optimal Functional Form via Symbolic Regression

## Hypothesis
The canonical power law S = c·A^z (Arrhenius 1921, Preston 1962) is either:
- **(A) Confirmed** — power law optimal across all scales → practice
- **(B) Rejected** — a different form (quadratic, logistic, triphasic, rational) fits better → novel

## Background
The species-area relationship is perhaps the most well-known pattern in ecology. For 100+ years the debate has raged over its functional form:
- Power law: S = c·A^z (Arrhenius 1921)
- Logistic: S = c·A^z/(1 + d·A^z) (He & Legendre 2002)
- Triphasic: different z at small/medium/large scales (Rosenzweig 1995)
- Quadratic in log-space: log S = log c + z·log A - d·(log A)^2 (Williams 1995)
- Breakpoint/piecewise: different z above/below threshold A

No one has applied symbolic regression to discover the optimal form from data.

## Crap-or-Worthwhile Test
| Finding | Verdict | Propaganda |
|---------|---------|------------|
| Power law best (z≈0.25) | Practice — 100 years of literature | KILLED |
| Breakpoint or triphasic | Moderate — known for specific taxa | Borderline |
| Novel rational/quadratic form | Novel — changes what people model | Proceed |

> If the conclusion is "power law z≈0.25" — this is practice.
> If SR discovers a new form with better fit — this changes how biodiversity is modeled.

## Methodology
Standard 5-phase adversarial SR pipeline.

**Predictors:** log10(Area), taxonomic group, habitat
**Target:** log10(Species Richness)
**Space:** log-log (standard for SAR)

## Data Sources
1. PREDICTS database (NHM data portal)
2. Drakare+2006 meta-analysis (794 SARs)
3. Triantis+2012 (oceanic island SARs)
4. BioTIME (species assemblages)
5. sars R package datasets

## Key References
- Arrhenius 1921. Species and area. J. Ecol.
- Preston 1962. The canonical distribution of commonness and rarity. Ecology
- Rosenzweig 1995. Species diversity in space and time.
- He & Legendre 2002. Species diversity patterns derived from species-area models. Ecology
- Drakare+2006. The imprint of the geographical, evolutionary and ecological context on species-area relationships. Ecol. Lett.
- Triantis+2012. The island species–area relationship. J. Biogeogr.

## Propaganda Clause
If SR recovers power law with z≈0.25 — KILL.
If SR discovers a better form — proceed.
