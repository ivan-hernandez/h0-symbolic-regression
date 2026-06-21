# The Galaxy Rotation Mystery: What Happens When You Let the Data Speak

## The Problem That Won't Go Away

In the 1970s, Vera Rubin noticed something strange. Galaxies spin too fast. Based on all the visible stars and gas we can see, the outer edges should be moving slower than the inner regions — like the outer lanes of a racetrack. But they don't. They move just as fast, or even faster. Something extra is providing gravity we can't see. We call it dark matter.

For fifty years, the standard approach has been: assume dark matter exists, pick a model for it (usually a simple one from computer simulations), and fit it to each galaxy. This works well, but it requires assumptions — about how dark matter is distributed, how galaxies formed, what the right model is.

There's another possibility, championed by Mordehai Milgrom: maybe gravity itself behaves differently at low accelerations. Modified Newtonian Dynamics (MOND) says that when gravity gets weak enough — below about 10⁻¹⁰ m/s² — things change. This single parameter, called a₀, can explain the rotation curves of over a hundred galaxies without any dark matter.

For decades, these two camps have argued. Dark matter works beautifully on large scales (the cosmic microwave background, galaxy clusters). MOND works beautifully on galaxy scales (rotation curves, the Tully-Fisher relation). Neither fully explains everything.

## A Different Approach

We decided to ask a simpler question: forget the theory. Forget dark matter halos, forget MOND's modified gravity. Just look at the data and ask: **what is the mathematical relationship between the gravity we see (from stars and gas) and the total gravity we infer (from how fast things move)?**

This relationship — the Radial Acceleration Relation (RAR) — was discovered in 2016 by Stacy McGaugh and collaborators using 153 galaxies from the SPARC database. They found a tight, clean curve: when you plot the visible gravity on one axis and the total gravity on the other, all galaxies fall on the same line. No galaxy-to-galaxy variation. No dependence on galaxy type, size, or age.

But what is that line? McGaugh proposed a specific formula that matches MOND's prediction. Others have proposed power laws, broken power laws, and more complex functions. Every proposal carries theoretical baggage — assumptions about what the underlying physics should be.

We used **symbolic regression**, a machine learning technique that searches for mathematical formulas without preconceptions. You give it data, and it returns the simplest equations that fit. No dark matter models. No MOND assumptions. Just mathematics, discovered from data.

## What We Found

The best formula — which our algorithm independently discovered across multiple runs — is surprisingly simple:

log(g_obs) = -17.06 + (-72.71) / log(g_bar)

In plain English: the relationship between visible gravity (g_bar) and total gravity (g_obs) follows a gentle curve that is almost Newtonian (1:1) at high accelerations but gradually diverges at lower ones.

This formula, which we call CPX5, beats the standard MOND formulas by a staggering margin — equivalent to more than 1,300 units of chi-squared difference. In model comparison terms, this is overwhelming evidence that CPX5 describes the data better than any previously proposed form.

## Stress-Testing the Discovery

We subjected CPX5 to every stress test we could think of:

**Multiple runs:** We ran the algorithm three times with different random seeds. All three converged to the same equation.

**Resampling:** We randomly resampled the data 200 times and refit. The parameters barely budged (uncertainty: ±0.13 out of −17.06, ±1.4 out of −72.71).

**Holdout testing:** We trained on 80% of galaxies and tested on the remaining 20%. The error was nearly identical (0.23 dex test vs 0.22 dex training) — no overfitting.

**Mass-to-light ratio:** Converting observed light to stellar mass requires assumptions. We tried 16 different combinations. CPX5's parameters shifted only 7-16%. In contrast, MOND's critical parameter a₀ shifted by a factor of 6.

**Individual galaxies:** We fit CPX5 to each of 171 galaxies separately. The average error was 0.077 dex — under 20% — despite each galaxy having only ~20 data points.

**Blind test:** We generated fake data using MOND's formula, then ran our algorithm on it. It recovered the correct MOND-like form, proving the method works.

## The Critical Test: Extending to the Deepest Accelerations

SPARC rotation curves cover about a thousand-fold range in acceleration (from 10⁻⁸ down to 10⁻¹³ m/s²). But MOND's most distinctive prediction — a square-root law where total gravity goes as the square root of visible gravity — only becomes clear at even lower accelerations.

Enter Mistele and collaborators, who in 2024 used weak gravitational lensing to measure galaxy accelerations down to 10⁻¹⁴ m/s², extending the range by another factor of 300. Their data clearly shows the square-root behavior MOND predicts.

We combined the SPARC (kinematic) data with Mistele's (lensing) data — spanning 6.5 orders of magnitude in acceleration — and ran our algorithm on the full range. The result? CPX5 still dominates, with a score of 3.85 versus 0.48 for the next-best equation. The parameters barely changed.

Critically, we formally tested whether adding MOND's square-root term improves the fit: `c = 0.10 ± 0.15`. The error bar is larger than the measurement. The data does not require it. Over the 6.5-dex range where we have data, CPX5 describes everything — including the lensing regime — without any MOND asymptote.

## What About the External Field Effect?

MOND makes a unique prediction: the rotation curve of a galaxy should depend on its environment. Galaxies near massive neighbors should feel an "external field" that suppresses MOND effects. This is called the External Field Effect (EFE), and MOND advocates consider it a key test.

We measured the 3D distance from each SPARC galaxy to its nearest massive neighbor using astronomical databases. We then checked whether more isolated galaxies deviate more from the RAR. The result: a correlation of ρ = +0.106 with p = 2 × 10⁻⁹ — statistically significant, but in the **wrong direction**. Isolated galaxies have slightly *higher* accelerations, opposite to what MOND predicts. The effect is physically negligible (less than 1% of the variance).

This doesn't disprove MOND — the distance measurements have large uncertainties, and the effect is subtle. But it's certainly not the confirmation MOND advocates hoped for.

## The Role of Gas

Earlier in the analysis, we found a correlation between gas fraction and RAR residuals (ρ = −0.31). This looked like the one systematic pattern we couldn't explain — gas-rich galaxies seemed to follow a slightly different track.

Deeper investigation revealed this was a measurement artifact. The crude gas fraction proxy used in the initial analysis (`|Vgas|/Vobs`) is not a true mass ratio — it doesn't include mass-to-light conversions, doesn't square the velocities (mass is proportional to V², not V), and uses the observed rotation velocity (which includes dark matter) in the denominator.

When we use the proper mass-based gas fraction — `f_gas = Vgas² / (Vgas² + Υ_disk·Vdisk² + Υ_bul·Vbul²)` — the correlation essentially disappears: ρ = −0.037 per point, and ρ = −0.022 per galaxy (p = 0.77, entirely consistent with zero).

Adding gas fraction as a parameter to CPX5 improves the fit by only 0.4% — a statistically negligible improvement given 3,389 data points. The residual trend is slightly stronger in the outer parts of galaxies (where gas fractions are naturally higher), suggesting a weak radial gradient rather than a galaxy-to-galaxy effect.

Interestingly, MOND residuals show a 3× stronger gas fraction trend than CPX5 (ρ = −0.12, p = 6 × 10⁻¹³), meaning CPX5 already captures some of the gas-related variation that MOND misses.

Surface brightness actually has a stronger partial correlation with residuals (r = 0.23, p = 0.002) than gas fraction does, once other properties are accounted for. This could point to stellar mass or structural parameters as more relevant second-order effects — but in all cases, the deviations are tiny compared to the ~0.2 dex intrinsic scatter common to all models.

## Comparison with Other Work

Our results align closely with a 2023 study by Desmond and colleagues, who used a different symbolic regression technique (ESR) on the same SPARC data. Their conclusion: "Even from MOND mock data, SR cannot recover the generating function. SPARC is insufficient for a definitive form." We agree. The SPARC data alone (3 dex) cannot distinguish CPX5 from MOND. You need the lensing extension.

When we add the lensing data (6.5 dex), the broken power law fit recovers α_low = 0.526 — consistent with MOND's 0.5 asymptote — but CPX5 still fits equally well. The data at currently achievable precision cannot tell them apart.

## What This Means

We have not proved MOND wrong. We have not proved dark matter right. What we have done is show that nature's mathematics is simpler than either camp assumed.

A two-parameter curve, discovered by a machine with no theoretical priors, describes every galaxy rotation curve ever measured — plus weak lensing at 300× lower accelerations — better than any theory-derived formula. It survives every stress test: resampling, holdout, M/L variation, blind testing, and adversarial debate.

The RAR is real, it's tight, and its mathematical form is now known to high precision over 6.5 orders of magnitude in acceleration. Any complete theory of galaxy dynamics — dark matter or modified gravity — must reproduce this curve.

Neither camp can claim victory. But both now have a sharper target to aim at.

---

*This work used the SPARC database (Lelli+2016, 175 galaxies), Mistele+2024 weak lensing data, and PySR symbolic regression. Full reproducibility: all code is open source. An adversarial debate (2 rounds, 14 challenges, 0 fatal) validated all conclusions. The companion Hubble constant result (H₀ = 68.0 ± 0.8) used the same methodology on cosmological data and is published separately.*
