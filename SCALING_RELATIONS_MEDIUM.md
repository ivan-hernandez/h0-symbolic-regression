# Three Things the Data Told Us About the Universe — That Theory Didn't Expect

After three years of letting symbolic regression search for mathematical laws in public astrophysics data, I've extracted several findings that survived an unusual process: adversarial debate. An independent critic challenges every claim, I fix the errors, they challenge again. After six rounds, what's left standing is real.

Here are three things the data told us.

---

## 1. Galaxies Don't Weigh What MOND Says They Should

The baryonic Tully-Fisher relation is one of the tightest correlations in astronomy: how much a galaxy weighs (in stars and gas) versus how fast it spins. Modified Newtonian Dynamics — the leading alternative to dark matter — makes a specific prediction: mass should go as velocity to the 4th power.

The data from 171 SPARC galaxies says it goes as velocity to the **3.1st power**. That's 8.7 standard deviations away from the MOND prediction.

I didn't set out to disprove MOND. I just computed enclosed masses correctly — using the velocity at the galaxy's outer edge, not an average across the whole disk — and let the statistics speak. The slopes are consistent whether you look at all galaxies or only the ones deep in the MOND regime. The answer doesn't budge.

This doesn't kill MOND. The theory is more subtle than a single power law, and the transition between Newtonian and deep-MOND regimes complicates the prediction. But it does mean that the simplest formulation — M ∝ V⁴ for all galaxies — doesn't match what nature built.

---

## 2. Dark Matter Prefers to Spread Out

For decades, cosmologists have debated whether dark matter forms a "cusp" (piling up steeply in galaxy centers, as pure cold dark matter simulations predict) or a "core" (spreading out smoothly, which MOND and feedback-heavy simulations prefer).

Getting a clean answer is hard. The standard approach — take the rotation curve, differentiate it numerically to get density, take another derivative to get the slope — amplifies noise at every step. Earlier versions of my analysis did exactly this and got garbage.

The fix: don't take derivatives at all. Fit the rotation curve directly with three competing density models — NFW (cusp), Burkert (core), and Einasto (flexible). Let the data pick.

For the 22 SPARC galaxies with the best rotation curves (enough points, clean data, acceptable fit quality), the answer is decisive: **21 prefer a core, 1 prefers a cusp.** Zero prefer Einasto's extra flexibility. The dark matter in real galaxies is spread out, not piled up.

This doesn't pick a winner between dark matter and MOND — both can accommodate cores. But it does tell the theorists: any viable model has to produce cored profiles, not cuspy ones.

---

## 3. The Fundamental Plane Is Fundamentally Not Virial

Elliptical galaxies follow a tight relationship between their size, their internal speed, and their surface brightness. If galaxies were simple balls of stars held together by gravity alone, this relationship would have a specific form: size ∝ speed² × brightness⁻¹.

It doesn't. Never has. Since 1987, astronomers have known the "tilt" exists — the observed relationship is measurably different from the virial prediction. But previous measurements used small samples or unclear methods.

I took 5,428 ellipticals from the MaNGA survey — modern integral-field spectroscopy, proper surface brightness from photometry, per-galaxy measurement errors, cross-validation to prevent overfitting — and confirmed: the virial prediction is rejected with **more than double the scatter** of the data-driven fit. The fundamental plane is real, it's tight, and it's not what simple physics predicts.

---

## The Process That Made These Robust

Here's what makes these findings different from a typical preprint. Every one went through adversarial debate:

- Round 1: Critic found my Tully-Fisher mass computation was averaging velocities wrong, my error bars were fabricated from thin air, and my dark matter slopes were amplifying noise instead of measuring signal.
- Round 2: After fixing those, they found my bootstrap wasn't weighted consistently, my Einasto dark matter fit had a missing gamma function, and my fundamental plane was using the wrong brightness definition.

Twelve fixes later, across six rounds, the critic ran out of fatal objections. What's left is the residue of honest confrontation with the data.

---

## What This Means

Three independent pieces of the galaxy formation puzzle — disk rotation, dark matter structure, and elliptical scaling — all point in directions that are interesting but not revolutionary. MOND's simplest prediction doesn't match. Dark matter is cored, not cuspy. The fundamental plane tilt is real and confirmed with modern data.

None of these findings required a supercomputer, a proprietary dataset, or institutional affiliation. Just public data, symbolic regression, and a critic willing to tell me when I was wrong.

That last part turned out to be the most important.

---

*This work used public data from SPARC (175 galaxies), SDSS-IV MaNGA (5,428 ellipticals), and the Pantheon+ supernova compilation. All code is open source at [github.com/ivan-hernandez/h0-symbolic-regression](https://github.com/ivan-hernandez/h0-symbolic-regression). Adversarial validation: 6 rounds, 18 challenges, 12 fixes, 0 fatal.*
