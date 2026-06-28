# P12: Temperature Dependence of Metabolic Rate Across Domains of Life

## Hypothesis
The Arrhenius-Boltzmann form log(B) = E/kT + c universally assumed by Metabolic Theory of Ecology (MTE, Gillooly+2001) is either:
- **(A) Confirmed** — linear across all domains, E ≈ 0.65 eV → practice
- **(B) Rejected** — curvature exists and varies systematically by domain → novel

## Crap-or-Worthwhile Test
| Condition | Verdict |
|-----------|---------|
| SR finds linear, E ≈ 0.65 eV universal | **Practice** (confirms MTE, publish as RNAAS note at most) |
| SR finds curvature, domain-dependent form | **Novel** (changes how people model thermal biology) |
| SR finds E varies with thermal niche (psychrophile vs thermophile) | **Novel** (systematic variation would refute universal E) |

> If activation energy is universal, this is consistent with literature.
> If activation energy varies predictably with growth temperature or domain, it changes what someone assumes.

## Methodology
Standard 5-phase adversarial SR pipeline.

**Predictors:** T (K), domain (Archaea/Bacteria/Eukarya), thermal class (psychrophile/mesophile/thermophile)
**Target:** log10(metabolic rate) or log10(growth rate)
**Log-log space:** log(rate) vs 1/kT (Arrhenius plot)

## Key References
- Gillooly+2001: "Effects of size and temperature on metabolic rate." Science 293:2248–2251
- Corkrey+2016: "Thermal adaptation of growth rate across the tree of life." Biol. Lett. 12:20160186
- Dell+2011: "Systematic variation in the temperature dependence of physiological and ecological traits." PNAS 108:10591–10596
- Knies & Kingsolver 2010: "Erroneous Arrhenius." Am. Nat. 176:E177–E185
- Pottier+2024: "OpdB: a global database on the optimal temperature of metabolic traits." Glob. Ecol. Biogeogr.

## Data Sources (targeted)
1. Corkrey+2016 — growth rate vs T for 469 species across all 3 domains (Dryad)
2. OpdB (Pottier+2024) — optimal temperature database (Zenodo/Figshare)
3. Hoehler+2023 — microbial metabolic rates (already have, needs temperature metadata)
4. Dell+2011 — thermal response curves for multiple traits (PNAS supplementary)

## Propaganda Clause
If SR recovers linear Arrhenius with universal E — KILL.
If SR finds curvature or domain-dependent E — proceed.
