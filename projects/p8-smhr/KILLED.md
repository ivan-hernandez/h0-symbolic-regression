# P8: Stellar Mass–Halo Mass Relation — KILLED (Practice)

**Date:** 2026-06-26
**Result:** SMHR is already comprehensively characterized by Behroozi+2013, Moster+2013, and multiple independent groups. SR cannot improve on well-established double-power-law forms given irreducible scatter.

## Crap-or-Worthwhile Assessment
**Question:** Does this change what someone would assume or do?

**No.** The SMHR functional form (double power law with redshift evolution) has been established for over a decade by multiple independent groups using different methods (abundance matching, halo occupation, subhalo abundance matching). The intrinsic scatter (~0.2 dex) is larger than any functional form improvement SR could find.

SR discovering the same double-power-law form would be "consistent with literature." SR finding a different form would be attributed to the specific data/selection, not a genuine discovery.

**Verdict: PRACTICE.** Killed.

## Scripts
- `smhr_data.py`: Data loader (Behroozi+2013 SMMR tables)
- `smhr_sr_remote.py`: Per-z and joint 2D SR runner

## Data
- Behroozi+2013 release (bwc2013): SFE/SFR/SMHR at z=0.1-8
- UniverseMachine DR1: observational constraints
