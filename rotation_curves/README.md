# Galactic Rotation Curves — Symbolic Regression

Reproduce and extend Desmond et al. (2023) on the Radial Acceleration Relation using PySR (Class SR) for a unified functional form.

## Data
- SPARC: 175 late-type galaxies, ~2700 rotation curve points
- `MassModels_Lelli2016c.mrt`: Combined table (R, Vobs, Vgas, Vdisk, Vbul, SBdisk, SBbul, D)
- `RAR.mrt`: Pre-computed Radial Acceleration Relation (gbar, gobs)
- `Rotmod_LTG.zip`: Individual galaxy rotation curve files

## Approach
1. Parse SPARC mass models → compute gbar, gobs
2. PySR search for g_obs = F(g_bar) with minimal priors
3. Compare with:
   - Desmond+2023 ESR results
   - MOND interpolating functions
   - ΛCDM predictions
4. Extend: test for galaxy property dependence, EFE, etc.

## Key References
- Desmond+2023: arXiv:2301.04368 — "On the functional form of the radial acceleration relation"
- Lelli+2016: SPARC data paper — AJ 152, 157
- McGaugh+2016: Original RAR — PRL 117, 201101
