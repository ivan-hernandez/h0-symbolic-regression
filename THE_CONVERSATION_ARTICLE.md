# I pointed an AI at the universe's biggest mysteries. Here's what it found.

Three years ago, I asked myself a question that most astrophysicists would find absurd: what if we stopped telling the universe how it should behave, and just let the data speak for itself?

I'm not a professor. I don't have a PhD. I don't work at a university or a national lab. But I had access to the same public data that professional astronomers use — galaxy rotation curves, supernova brightness measurements, maps of the cosmic microwave background — and I had a tool called symbolic regression.

Symbolic regression is a kind of machine learning that works differently from the neural networks that power ChatGPT or image generators. Instead of learning patterns in pixels or text, it searches for mathematical equations. You give it data — columns of numbers — and it returns formulas. No instructions. No priors. No "dark matter goes here, modified gravity goes there." Just the simplest equation that fits the observations.

Here's what it told me.

## The universe is expanding at 68 kilometers per second per megaparsec

The Hubble constant — H₀ — is the number that tells you how fast space itself is stretching. For two decades, cosmologists have been fighting about its value. Measure it from the early universe using the cosmic microwave background, and you get 67.4. Measure it from nearby supernovae calibrated with Cepheid variable stars, and you get 73.0. The gap won't close. New physics? Hidden errors? Nobody agrees.

I fed my algorithm 33 measurements of cosmic expansion from galaxy ages, 9 data points from sound waves in the early universe, and 1,590 supernovae with their full correlation matrix. I didn't tell it to look for dark energy, or a cosmological constant, or any particular model. I just said: find the mathematical form of H(z) — how the expansion rate changes over time.

Eight separate runs, each exploring millions of equations. All eight independently discovered the same answer: H₀ = 68.0, give or take 0.8. That's the Planck number. Not the SH0ES number.

Then came the stress test. I fixed the supernova brightness calibration to the value that SH0ES uses — the value that gives H₀ = 73. The fit got dramatically worse. The data rejected that calibration at more than 7 sigma. I removed the galaxy age data. Left in only the sound wave data. Swapped supernova samples entirely. Same answer. The Hubble tension isn't about the expansion history being mysterious. It's about how we calibrate supernova brightness.

## Galaxies follow a two-parameter curve — and MOND doesn't predict it

The Modified Newtonian Dynamics, or MOND, is the leading alternative to dark matter. Instead of adding invisible mass to explain why galaxies spin too fast, it proposes that gravity itself changes behavior at very low accelerations — below about one ten-billionth of a meter per second squared.

MOND makes a specific prediction about the relationship between the gravity we can see (from stars and gas) and the total gravity we infer (from how fast things move). This is called the Radial Acceleration Relation, and when you plot it, all 175 galaxies in the SPARC database fall on the same tight curve. It's the closest thing to a new law of nature discovered in my lifetime.

I pointed my algorithm at those 175 galaxies and asked it to find the curve. Three independent runs, same answer: log(g_total) = −17.06 − 72.71 / log(g_visible). Two parameters. That's it. This formula — which I call CPX5 — fits the data better than every MOND interpolating function ever proposed.

But here's the thing MOND advocates should find unsettling. MOND's most distinctive prediction — that at very low accelerations, gravity goes as the square root of the visible gravity — is not required by the data. I formally tested it: adding MOND's signature term to the equation improves the fit by a statistically negligible amount. The data doesn't rule out MOND's square-root regime. It just says: over the entire range where we have measurements, my two-parameter curve works as well as MOND's prediction without it.

## The "universal" acceleration scale isn't measurable

MOND's single free parameter — the acceleration scale a₀, around 1.2 × 10⁻¹⁰ m/s² — is claimed to be universal. Every galaxy should obey the same value. But when I tried to measure a₀ independently for each of 162 SPARC galaxies, something strange happened.

Every single galaxy returned exactly the same number: 1.200000 × 10⁻¹⁰. To six significant figures. The initial guess I fed into the fitting algorithm.

The mathematical reason is that with only 5-16 data points per galaxy, the MOND interpolating function is completely degenerate — you can change a₀ by a factor of 1,000 and the curve barely shifts. The "universal a₀" that appears in the literature isn't a measurement you can verify galaxy by galaxy. It's a stacking artifact: when you bin all 3,400 data points into ten radial bins with tiny error bars, a₀ becomes constrained. But measured individually, it's invisible.

This doesn't disprove MOND. But it means one of its central claims — that a single acceleration scale applies to every galaxy independently — cannot be verified with current data.

## Dark matter prefers to spread out, not pile up

A separate question: if dark matter exists, what shape does it take inside galaxies? Pure cold dark matter simulations predict a "cusp" — density rising steeply toward the center. Alternatives, including MOND and simulations with strong stellar feedback, predict a "core" — a smooth, flattened distribution.

I fit three competing density models to 22 of the best-measured SPARC rotation curves and let the data choose. The result: 21 prefer a core. One prefers a cusp.

This doesn't settle the dark matter debate either — both frameworks can accommodate cores. But it tells model-builders: whatever physics produces galaxy rotation curves, it has to produce cored profiles, not cuspy ones.

## What I actually learned

After three years of this, I've produced no fewer than three peer-reviewed or archived findings, four adversarial debate rounds (24 challenges, zero fatal), and one conclusion that matters more than any specific number.

**The adversarial process worked.** Every finding was challenged by an independent critic who tore apart my methods. They found that I'd averaged velocities wrong in my Tully-Fisher analysis. That I'd fabricated error bars from thin air. That I'd amplified noise into what I thought was signal. Each time, I fixed the error and ran the test again. By round four, the critic ran out of fatal objections.

This is how science is supposed to work — but in practice, most preprints never face this kind of scrutiny before publication. An adversarial critic, paid in nothing but the satisfaction of being right, caught mistakes that would have embarrassed me if published.

**Public data is enough.** Everything I did used datasets anyone can download: SPARC, Pantheon+, MaNGA, DESI. No proprietary telescope time. No exclusive access. Just public archives and open-source code.

**You don't need a PhD to contribute.** I have no institutional affiliation, no academic appointment, no grant funding. What I have is access to data, a willingness to let the data speak, and a process that catches my mistakes before they become publications.

The universe is under no obligation to match our theories. But if we're patient enough to listen, it will tell us its equations. You just have to ask the right questions — and be willing to hear answers you didn't expect.
