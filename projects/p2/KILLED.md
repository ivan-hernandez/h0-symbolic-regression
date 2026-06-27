# P2: GW Mass Distribution — KILLED

## Reason: Practice (data insufficient)

All 5 SR seeds finished (42, 43, 44, 123, 456). The common discovered form
was `x³ · (const − x + α/(x₀ − x))` — a cubic zero-crossing with rational
pole near the peak — but this is unreliable.

## Fatal flaw: overfitting

- Only 10 non-empty bins from 184 GWTC-3 events (5 bins/dex, 1–200 M☉)
- Best SR models had 14–15 complexity (parameters)
- Data:parameter ratio = 0.67
- 2 of 3 local seeds had poles; the surviving exp-form (seed 456) is
  overfit, not physically meaningful
- PL+Peak baseline used literature (population-level) parameters, not fit
  to the same 184 events — comparison was unfair

## Verdict

The result cannot be trusted. 184 events is fundamentally insufficient for
binned SR. To do this properly would require:
- GWTC-4 (~1000+ events)
- Or an individual-event approach (fit CDF with monotonic SR)
Neither is worth pursuing as a standalone project at this point.

## Crap-or-Worthwhile

Practice — does not pass the test.
