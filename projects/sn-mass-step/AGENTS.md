# SN Host Galaxy Mass Step — SR Discovery

## Goal
Discover the functional form of the Type Ia SN host galaxy mass step
using symbolic regression on Pantheon+ data.

## Template
Follow METHODOLOGY.md (5-phase adversarial SR pipeline).

## Constraints
- Fit ∆µ (Hubble residual) as a function of HOST_LOGMASS
- Compare to step function at logM = 10 (the current standard)
- Propaganda clause: if result is "consistent with step function," kill it
- Data from Pantheon+ (has HOST_LOGMASS built in)
- DES-SN5YR as secondary cross-check (needs host catalog matching)

## Progress
### Done
- PROPOSAL.md written
- Phase 1: Residuals computed, mass step explored (γ=0.059 in uncorrected data)
- Phase 2: SR on corrected residual found nothing (noise)
- Phase 3: Validation — no slope, no smoothness, no redshift evolution
- KILLED as practice; see KILLED.md

## Key Decisions
- Use MU_SH0ES (Pantheon+) / MU (DES-SN5YR) as the distance modulus
- Use zHD as the redshift
- Use flat ΛCDM with H0 fitted to the data as reference cosmology
- Fit the residual ∆µ = µ_obs − µ_model(z) − M as f(logM_host)
- SR target: y = ∆µ (one-dimensional f(logM_host))
- Consider extension to 2D: f(logM_host, sSFR) or f(logM_host, z)
