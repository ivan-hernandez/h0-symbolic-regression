# Pulsar Glitch Size Distribution: Evidence for Two Populations

## Finding
Pulsar glitch fractional sizes (Δν/ν) follow a **Weibull (stretched exponential) distribution**, not a power law.

## Data
- **724 glitches** from the Jodrell Bank catalogue (222 pulsars)
- Range: 2.5×10⁻¹² to 6.5×10⁻⁵ (7+ dex)
- Sources: Espinoza+2011 compilation + updated measurements to 2025

## Methods
- Maximum likelihood estimation for all models
- AIC comparison across Weibull, power law (Pareto), lognormal, exponential
- Breakpoint analysis for two-population model
- Bootstrap uncertainty (500 resamples)
- Jackknife stability test
- Per-pulsar fits for prolific glitchers

## Results

### Single-population comparison
| Model | Parameters | ΔAIC |
|-------|-----------|------|
| **Weibull** | k=0.349, λ=2.7×10⁻⁷ | **0** (best) |
| Lognormal | s=3.36 | +12.8 |
| Exponential | λ=1.15×10⁻⁶ | +2120 |
| Power law (Pareto) | α=1.10 | +11917 |

Weibull preferred decisively. ΔAIC > 10 vs lognormal (Kass & Raftery "decisive"), ΔAIC > 10000 vs power law.

### Two-population model
**Breakpoint detected at ~5×10⁻⁸** (ΔAIC = 1050 over single Weibull):

| Population | Size range | k | Interpretation |
|-----------|-----------|----|---------------|
| Small glitches | <5×10⁻⁸ | 0.65 | Crust cracking (weakest-link failure) |
| Large glitches | >5×10⁻⁸ | 0.75–1.0 | Near-exponential; consistent with superfluid unpinning |

### Per-pulsar variability
| Pulsar | Glitches | k | Dominant type |
|--------|---------|----|--------------|
| J0537-6910 | 65 | 1.39 | Large-glitch emitter |
| J0534+2200 (Crab) | 32 | 0.58 | Small-glitch emitter |
| J0835-4510 (Vela) | 26 | 0.73 | Mixed |
| J1740-3015 | 38 | 0.42 | Small-glitch dominated |

### Robustness
- Bootstrap: k = 0.349 ± 0.008, λ = 2.69×10⁻⁷ ± 2.9×10⁻⁸
- Jackknife (remove top 10 glitches): λ stable within 10%
- KS: Weibull KS=0.094 (best among all models; KS rejection at n=724 expected for tiny deviations)
- Truncation: k varies with threshold, explained by two-population mixture

## Conclusions
1. **Power law is ruled out** (ΔAIC = 11917) — glitch sizes have a characteristic scale
2. **Two populations exist**, likely with different physical origins
3. Small glitches (k≈0.65) consistent with **crust cracking**
4. Large glitches (k≈0.75–1.0) consistent with **superfluid unpinning**
5. The Weibull distribution outperforms the lognormal (ΔAIC = 12.8)

## References
- Espinoza+2011, A&A 533, A114
- Melatos+2018, MNRAS 477, L21
- Fuentes+2017, MNRAS 468, 1846
- Warszawski & Melatos 2011, MNRAS 415, 1611
