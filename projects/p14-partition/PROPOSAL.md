# P14: Partition Function Asymptotics via Symbolic Regression

## Hypothesis
The Hardy-Ramanujan asymptotic formula p(n) ~ exp(pi*sqrt(2n/3))/(4n*sqrt(3)) can be either:
- **(A) Confirmed** — SR recovers the canonical asymptotic form → practice
- **(B) Improved** — SR discovers a better approximation with correction terms → novel

## Background
The partition function p(n) counts the number of ways to write n as a sum of positive integers. Hardy & Ramanujan (1918) derived the asymptotic:
p(n) ~ 1/(4n*sqrt(3)) * exp(pi*sqrt(2n/3))

Rademacher (1937) gave an exact convergent series. Modern researchers still study correction terms and remainders (Bruinier & Ono 2010, Folsom & Ono 2016). No one has applied SR to discover the asymptotic form from data.

## Crap-or-Worthwhile Test
| Finding | Verdict | Propaganda |
|---------|---------|------------|
| SR recovers pi*sqrt(2n/3) exactly | Practice — confirms Hardy-Ramanujan | KILLED |
| SR discovers rational approx to pi*sqrt(2/3) | Mildly interesting, probably known | Borderline |
| SR finds novel functional form beating HR | Novel | Proceed |
| SR discovers correction terms (Dean & Majumdar 2000 style) | Novel | Proceed |

## Data
p(n) computed via Euler's pentagonal recurrence:
p(0) = 1
p(n) = sum_{k != 0} (-1)^(k-1) * p(n - g_k)
where g_k = k(3k-1)/2 (generalized pentagonal numbers)

Compute n up to 10^5 (arbitrary precision Python ints).

## Key References
- Hardy & Ramanujan 1918. Asymptotic formulae for the partition function. Proc. LMS
- Rademacher 1937. On the partition function p(n). Proc. LMS
- Bruinier & Ono 2010. Algebraic formulas for the coefficients of half-integral weight harmonic weak Maass forms. Adv. Math.
- Dean & Majumdar 2000. Large deviations of extreme eigenvalues of random matrices. Phys. Rev. E (correction term for p(n))
- OEIS A000041

## Propaganda Clause
If SR recovers Hardy-Ramanujan exactly — KILL.
If SR discovers a novel form or correction — proceed.
