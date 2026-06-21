# The Hubble Constant Tension: Solved by Letting the Data Speak

*Ivan Hernandez · June 2026*

---

For the past decade, cosmology has faced a crisis.

Two different ways of measuring how fast the universe is expanding give answers that disagree by more than 5 standard deviations.

The Planck satellite, looking at the infant universe, says the expansion rate (the Hubble constant, or H₀) is about 67.4 km/s/Mpc.

The SH0ES project, using nearby stars and supernovae, says it's 73.0.

Neither budges. Neither is wrong — or so the argument goes.

**Our new analysis suggests a different conclusion. The expansion history itself isn't the problem. The problem is in the calibration of the supernovae.**

---

## What We Did

We used a technique called symbolic regression — essentially, a computer search that explores millions of possible mathematical formulas to find one that fits the data.

We fed it the best available measurements of the universe's expansion:

- **Cosmic chronometers**: The ages of old, quiet galaxies at different distances, giving us the expansion rate directly
- **Baryon acoustic oscillations (BAO)**: Frozen sound waves from the early universe, measured by the SDSS and DESI galaxy surveys
- **Three different supernova datasets**: Pantheon+, DES-SN5YR, and Union3 — nearly 2000 exploding stars spanning billions of light-years

The key is that we didn't assume a cosmological model. We let the data tell us what shape the expansion history has.

---

## What We Found

The search converged on a simple 4-parameter formula. Nothing exotic — just a polynomial. Eight independent computer runs, each exploring millions of possibilities, all found the same answer.

**The result: H₀ = 68.0 ± 0.8 km/s/Mpc.**

That's consistent with Planck (67.4) at only 1.2 standard deviations. It excludes the SH0ES value (73.0) at 5 standard deviations.

Then we did something crucial: we tested every plausible objection.

---

## The Test That Changed Everything

We ran 7 systematic tests.

Removing different datasets. Changing the mathematical form. Switching to full covariance matrices. Using three completely independent supernova samples.

Every single test gave H₀ between 67 and 69. The answer was rock solid.

**Except one.**

When we fixed the supernova absolute magnitude to the value SH0ES uses — their Cepheid calibration — H₀ jumped to 74.4, and the fit got dramatically worse. The statistical rejection was 8 to 9 sigma.

What this means is profound: **the supernovae data, by themselves, contain no preference for a high H₀.** The expansion history doesn't want H₀ = 73. The only way to get that number is to force the supernova calibration to the SH0ES Cepheid distance scale — and the data screams that this is wrong.

---

## Why This Matters

The Hubble tension has spawned hundreds of proposed solutions involving exotic new physics: early dark energy, modified gravity, extra relativistic species. Our results suggest none of that is necessary.

**The tension is not in the expansion of the universe. It's in the calibration of the distance ladder.**

Specifically, the Cepheid variable stars that SH0ES uses to anchor their measurement. This is consistent with a growing body of evidence that the Cepheid distance scale may harbor subtle systematic errors at the ~0.1 magnitude level — enough to explain the entire discrepancy.

---

## How Robust Is This?

We checked every way we could think to break the result:

**Different supernova samples:** Pantheon+ full covariance (1590 SNe), DES-SN5YR full covariance (1820 SNe), Union3 (22 binned points) — all give H₀ ≈ 67–68.

**Different BAO datasets:** SDSS only, DESI only, both — all consistent.

**No cosmic chronometers:** Using only BAO + DESI + SNe — H₀ = 68.6.

**Different functional forms:** Third-order Taylor expansion — H₀ = 67.4 (identical).

**Full covariance:** Using the correct off-diagonal terms — H₀ shifts by less than 1 km/s.

**Comparison with standard cosmology:** A direct ΛCDM fit gives H₀ = 67.9, with a joint fit that differs from our model by only Δχ² = 1.2.

The answer doesn't depend on which data you use or what functional form you assume. It only changes when you force the Cepheid calibration.

---

## What This Means

If confirmed, this result has a simple implication: the universe is expanding at about 68 km/s/Mpc, consistent with the Planck measurement from the cosmic microwave background.

The Hubble tension is not a crack in the standard model of cosmology. It's a crack in the Cepheid distance ladder.

This doesn't mean SH0ES is "wrong." They've done extraordinary work refining the distance ladder over two decades. It means the systematic uncertainties in the Cepheid calibration may be larger than currently estimated.

---

## Update: What We've Done Since

We've now extended the analysis in several ways, and the answer hasn't budged.

**DESI DR2.** The Dark Energy Spectroscopic Instrument released its second data release in 2025 — three years of data instead of one, with twice the precision. The result: H₀ = 68.3, identical to our original finding. Every BAO point agrees with the first release within 1 standard deviation.

**M(z) evolution.** We tested whether the supernova brightness changes with cosmic time — a possible sign of systematic error or new physics. The answer: no evolution detected. The supernova absolute magnitude is constant across billions of years.

**External cross-checks.** Three completely independent techniques give the same answer:
- Gravitational waves: GW170817 says H₀ = 65.5 ± 4.4
- GW + galaxy surveys: H₀ = 67.9 ± 4.4
- Strong gravitational lensing (TDCOSMO): H₀ = 71.6 ± 3.6

Combined: H₀ = 68.8 ± 2.3 — consistent with our 68.3.

**The bottom line:** every new dataset, every new test, every independent method converges on H₀ ≈ 68. The only way to get 73 is to force the Cepheid calibration. The data has spoken, and it hasn't changed its mind.

---

Future measurements — from JWST's higher-resolution Cepheid observations, from the Euclid and Roman space telescopes, and from the next generation of cosmic chronometer surveys — will settle the question definitively.

**But for now, the data has spoken. And it says: H₀ ≈ 68.**

---

*The full paper with all technical details is available on Zenodo: [10.5281/zenodo.20778036](https://zenodo.org/records/20778036)*

*All analysis code is publicly available on GitHub: [github.com/ivan-hernandez/h0-symbolic-regression](https://github.com/ivan-hernandez/h0-symbolic-regression)*
