# P2: GW Compact Object Mass Distribution — SR Discovery

## Goal
Discover the functional form of merging compact object mass distribution
using symbolic regression on GWTC-3 data.

## Template
Follow METHODOLOGY.md (5-phase adversarial SR pipeline).

## Constraints
- Fit PDF of primary mass m1
- Compare to Power Law + Peak baseline (LVK 2023)
- Propaganda clause: if result is "consistent with literature," kill it

## Progress
### In Progress
- Phase 1: Data download and parsing

### Done
- (none yet)

## Remote Compute
- Julia 1.11.9 at ~/julia/, PySR 1.5.10 ready
- 12 cores, 15 GB RAM at 100.121.64.70

## Key Decisions
- Use primary mass m1 (the more massive component)
- Use confident detections only (FAR < 0.01 yr^-1)
- Compare to Power Law + Peak as baseline
- Consider fitting histogram density rather than individual masses
