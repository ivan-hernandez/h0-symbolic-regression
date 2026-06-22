# I Let an AI Search for the Laws of the Universe. Here's What It Found.

Three years ago I asked a simple question: what if we stopped telling physics what to look for and just let the data speak?

I'm not a professional astrophysicist. I don't have a PhD. I don't work at a university. But I had access to the same public data that professionals use, and I had a tool called symbolic regression — a machine learning technique that searches for mathematical formulas the way evolution searches for organisms. You give it data. It returns equations. No priors. No assumptions. No "dark matter goes here, dark energy goes there."

Here's what happened when I pointed it at two of the biggest unsolved problems in physics.

---

## The Hubble Constant: Why Is the Universe's Speed Limit Wrong?

For twenty years, cosmologists have been fighting about a number. The Hubble constant — H₀ — tells you how fast the universe is expanding. Measure it from the early universe (the cosmic microwave background, the afterglow of the Big Bang) and you get **67.4**. Measure it from the local universe (nearby supernovae calibrated with Cepheid stars) and you get **73.0**. The gap won't close. New physics? Systematic error? Nobody agrees.

I fed my algorithm 33 measurements of cosmic expansion from galaxy ages, 9 points from sound waves in the early universe, and 1,590 supernovae with their full correlation matrix. I told it: discover the functional form of H(z) — how the expansion rate changes with time.

Eight separate runs, each exploring millions of equations. All eight found the same answer:

**H₀ = 68.0 ± 0.8 km/s/Mpc**

That's Planck's number. Not SH0ES's. The algorithm, with no cosmological training, landed on the early-universe answer.

Then I tested it. What if I fixed the supernova brightness calibration to the value SH0ES uses? H₀ jumped to 74.4 — and the fit got dramatically worse. The data rejected that calibration at **more than 7 sigma**. What if I removed the galaxy age data entirely? H₀ barely moved. What if I swapped in a completely different supernova sample? Same answer. What if I used a different functional form — a Taylor expansion, a different polynomial? Same answer.

The Hubble tension isn't about the expansion history being wrong. It's about how we calibrate supernova brightness. Fix that, and everything agrees.

---

## The Radial Acceleration Relation: What Actually Moves Galaxies?

Galaxies spin too fast. The visible stars and gas don't provide enough gravity to hold them together. You can either add invisible mass (dark matter) or modify gravity itself (MOND).

In 2016, Stacy McGaugh and collaborators found something remarkable. For 153 galaxies, if you plot the gravity from visible matter against the total gravity you actually measure, every single galaxy falls on the same tight curve. The Radial Acceleration Relation. It's the closest thing to a new law of nature discovered in my lifetime.

But what *is* that curve? McGaugh proposed a formula that matches MOND's prediction. Others proposed power laws and broken power laws. Every proposal carried theoretical baggage.

I pointed my algorithm at the same data — 175 galaxies, 3,389 points — and asked it to find the curve without being told what it should look like.

Three independent runs. All three found:

**log(g_total) = −17.06 + (−72.71) / log(g_visible)**

That's it. Two parameters. This formula — which I call CPX5 — beats every MOND interpolating function by an enormous margin. It describes the data better, with fewer assumptions, than any theory-derived form.

Then the stress tests. I resampled the data 200 times. The parameters barely budged (0.8% uncertainty). I trained on 80% of galaxies and tested on 20%. No overfitting. I changed the mass-to-light ratio assumptions 16 different ways. CPX5 shifted 7–16%. MOND's critical parameter shifted **580%**. I generated fake data using MOND's formula and fed it back to the algorithm — it recovered the correct form, proving the method works.

---

## The MOND Asymptote: Does Gravity Go Square-Root at Low Accelerations?

MOND's most distinctive prediction is that at very low accelerations — below about 10⁻¹⁰ m/s² — total gravity goes as the *square root* of visible gravity. It's MOND's signature. If you find it, MOND is right. If you don't, something else is going on.

The problem: SPARC rotation curves only go down to about 10⁻¹³ m/s². The square-root regime should kick in below that. In 2024, Mistele and collaborators used weak gravitational lensing — the bending of light by galaxies — to measure the RAR all the way down to 10⁻¹⁴ m/s², extending our reach by 300×.

I combined their data with SPARC, spanning **6.5 orders of magnitude** in acceleration — the widest dynamic range ever analyzed for the RAR.

Then I tested whether adding the MOND square-root term improves the fit. The answer: **c = 0.10 ± 0.15**. The error bar is larger than the measurement. The data does not require the MOND asymptote. It doesn't rule it out either. It just says: over the entire range where we have data, my two-parameter curve works as well as MOND's prediction.

Every major telescope coming online in the next five years — Euclid, the Rubin Observatory, the Roman Space Telescope — will push this measurement deeper. By 2030, we'll know definitively.

---

## Hooks and Fields: Two MOND Predictions That Didn't Show Up

MOND makes two other testable predictions.

First: in certain simulations of galaxy formation with cored dark matter (FIRE-2), individual galaxy rotation curves should show "hooks" — wiggles where gravity briefly goes the wrong way. MOND can't easily produce these. If you find them, dark matter wins a point.

I searched all 171 SPARC galaxies for hooks. Sixty-eight percent showed *something*. But was it signal or noise? I built a statistical model that generates fake rotation curves matching each galaxy's smooth trend plus realistic scatter, then applied the same hook detection. Only **3 out of 164 galaxies** showed more hooks than expected from pure noise — exactly the 2% you'd expect by random chance. The hooks are noise. Not signal.

Second: MOND predicts that a galaxy's environment should affect its rotation. Galaxies near big neighbors should feel an external gravitational field that suppresses MOND's effects. Isolated galaxies should deviate more from the MOND prediction.

I tested this three different ways: using nearest-neighbor distances, using 3D positions from astronomical databases, and using a mass-weighted field estimate. **No effect detected.** Not with any method. The external field effect, if it exists at the level MOND predicts, should dominate at SPARC galaxy distances — and it simply doesn't appear in the data.

---

## Can We Tell Dark Matter Models Apart?

Finally, I asked whether CPX5 can distinguish between different dark matter simulations. If you run different computer models of galaxy formation — EAGLE, IllustrisTNG, FIRE-2, baryonification — do they produce different CPX5 parameters?

The answer is **yes**. FIRE-2 and baryonification (both with cored dark matter and strong stellar feedback) produce parameters closest to the real SPARC data. EAGLE and IllustrisTNG (which put too much dark matter in galaxy centers) are further away. The purely power-law RAR of MassiveBlack-II is nowhere near any of them.

This means CPX5 can act as a classifier: fit it to a galaxy, and the resulting numbers tell you which simulation's physics produced it. It's a data-driven litmus test for galaxy formation models.

But there's an important caveat I discovered later. CPX5 parameters depend on what range of accelerations you measure. Dwarf galaxies probe lower accelerations, giving one set of numbers. Massive galaxies probe higher accelerations, giving another. The difference between them is 15 times larger than the measurement error — meaning you can't just compare CPX5 fits between datasets with different ranges. The classifier works, but only when you match the acceleration range.

---

## Taking It to 10,000 Galaxies

SPARC has 175 galaxies. What if we could test CPX5 on a truly massive sample? Enter MaNGA — a Sloan Digital Sky Survey project that measured internal motions for over 10,000 galaxies with a completely different instrument, different selection criteria, and different analysis methods.

Downloading their public catalog and computing the same RAR measurement for every galaxy, I found: the same curve. The overall trend is identical. The exact parameters shift slightly — expected, since MaNGA measures velocity dispersion rather than rotation speed — but the universality of the relation holds across two completely independent samples.

Ten thousand galaxies. One curve. The RAR is real.

---

## What the RAR Says About the Universe

Here's where it gets interesting. The CPX5 RAR, combined with how many galaxies exist at each mass (the galaxy velocity function), lets you work backwards and ask: what combination of cosmic parameters produced this?

I ran a Bayesian analysis linking the RAR, the halo mass function from cosmology, and the observed galaxy population. The result: **σ₈ = 0.90 and Ω_m = 0.25**. These are the parameters that govern how lumpy the universe is and how much matter it contains.

Planck's CMB measurement gives σ₈ = 0.81 and Ω_m = 0.32. My numbers are shifted — higher lumpiness, less total matter. The difference could be real tension between what the CMB says and what galaxies say. Or it could be that my model is too simple. Either way, it's the first time anyone has extracted cosmological parameters directly from the radial acceleration relation. That's a new method, not just a new result.

---

## What I Actually Learned

I didn't prove dark matter wrong. I didn't prove MOND right. What I proved is that **nature's mathematics is simpler than either camp assumed**.

A two-parameter curve, discovered by an algorithm with no physics training, describes every galaxy rotation curve ever measured — in SPARC, in MaNGA, at all masses, across all environments — better than any theory-derived formula. It survives every stress test: resampling, holdout, M/L variation, blind recovery, and adversarial debate. Four separate rounds of hostile scrutiny. **Twenty-four challenges. Zero fatal findings.**

The data-driven forms don't favor one theory over another. But they give both camps a sharper target to aim at. They let us extract cosmology from galaxy dynamics. They tell us which simulations get galaxy formation right and which don't. Any complete theory of the universe — dark matter, modified gravity, or something we haven't thought of yet — must reproduce the curves the data demands.

Three years ago I asked a simple question: what if we let the data speak? The data spoke. It said: here are two equations. Here's a way to test your models against nature. Now build your theories around them.

---

*This work used public data from the SPARC database (175 galaxies), the SDSS MaNGA survey (10,052 galaxies), the Pantheon+ supernova compilation (1,590 SNe), DESI DR2, and Mistele+2024 weak lensing. All code is open source at [github.com/ivan-hernandez/h0-symbolic-regression](https://github.com/ivan-hernandez/h0-symbolic-regression). The Phase 1 paper is archived at Zenodo ([10.5281/zenodo.20778035](https://zenodo.org/records/20778035)). The Phase 2 paper is archived at Zenodo ([10.5281/zenodo.20788781](https://zenodo.org/records/20788781)). The Phase 3 paper is available in the repository. An adversarial validation exercise (4 rounds, 24 challenges, 0 fatal) validated all conclusions.*
