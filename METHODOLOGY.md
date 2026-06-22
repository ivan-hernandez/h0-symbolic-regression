# Symbolic Regression in Astrophysics — Reusable Pipeline

A battle-tested methodology for applying symbolic regression (PySR) to
astrophysical problems, validated across 3 published phases, 4 adversarial
debate rounds, and 24 challenges (0 fatal).

---

## Phase Outline

Every project follows this structure:

### Discovery
1. Parse public data into `(X, y)` with proper error bars
2. Run PySR with `model_selection="accuracy"`, ≥3 random seeds
3. Identify Pareto-optimal forms (CPX3, CPX5, CPX7)
4. Select the best 2-parameter form by stability across seeds

### Validation
1. **Multi-seed** — ≥3 independent SR runs, verify equation convergence
2. **Bootstrap** — Galaxy/point-wise resampling (≥200), get parameter distributions
3. **Holdout** — K-fold (≥10), compare train vs test RMS
4. **M/L sweep** — Grid of mass-to-light assumptions (≥16 combinations)
5. **Blind test** — Generate mock data from known models, verify recovery
6. **Per-unit** — Fit the form to each individual data unit (galaxy, etc.)
7. **Literature** — Direct comparison with published results

### Extension
1. **Additional data** — Combine with independent samples (lensing, other surveys)
2. **Cross-sample** — Test form on completely different instruments/telescopes
3. **Theory tests** — Specific falsifiable predictions (MOND asymptote, EFE, hooks)
4. **Parameter dependence** — Correlate parameters with physical properties

### Synthesis (optional)
1. **Model discrimination** — Use form as classifier between competing theories
2. **Joint constraints** — Combine with other forms for multi-dimensional constraints
3. **Forecast** — Predict when future surveys will distinguish between models

### Adversarial Validation (mandatory)
1. Round 1: Adversary challenges every claim → Defender responds
2. Round 2: Adversary escalates deeper counter-attacks
3. Fixes: Implement adversary's valid critiques
4. Final round: Re-evaluate fixed results
5. Document all challenges and resolutions in debate log

### Publication
1. **Paper** — LaTeX in `aastex631.cls` two-column format
2. **LaTeX compile** — `tectonic paper.tex` (portable, no texlive needed)
3. **GitHub** — All code, data, figures, and paper source
4. **Zenodo** — Zip paper, code, CSVs → upload → get DOI
5. **OSF** — Create project hub linking GitHub + Zenodo
6. **Medium** — General public article (if desired)
7. **RNAAS** — Short notes for novel singular findings

### Documentation
1. **AGENTS.md** — Project state, constraints, progress
2. **PROPOSAL.md** — Original plan (static, written before work)
3. **PROGRESS.md** — Living document tracking actual accomplishments
4. **README.md** — Quick start, file listing, validation summary

---

## File Structure

```
project/
├── AGENTS.md                  # Project state + constraints
├── PROPOSAL.md                # Original plan (static)
├── PROGRESS.md                # Accomplishments (living)
├── README.md                  # Quick start
├── paper/                     # LaTeX paper
│   ├── paper.tex
│   ├── paper.pdf
│   ├── aastex631.cls
│   └── fig1_*.pdf
├── analysis/                  # Generated outputs
│   ├── model_comparison.csv
│   ├── bootstrap_results.csv
│   ├── holdout_results.csv
│   └── *.png
├── data/                      # Input data (if redistributable)
├── parse_data.py              # Data loader
├── sr_discovery.py            # PySR search
├── analysis.py                # Full analysis pipeline
└── validate.py                # Validation suite
```

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| `model_selection="accuracy"` | Uses lowest-loss model directly |
| Weak prior at boundaries | Suppresses pathological solutions without biasing result |
| Log-log space for RAR | Avoids high-value domination of MSE |
| `max(e, 0.1*y)` error floor | Captures intrinsic scatter uniformly across models |
| AIC = χ² + 2k | Correct for known errors (not n·ln(χ²/n)) |
| M/L range [0.3, 1.0] | Plausible stellar population range |
| ≥3 SR seeds | Verifies equation convergence (not a lucky draw) |
| ≥200 bootstrap resamples | Stable parameter uncertainty estimates |
| ≥10 holdout splits | Reliable train/test comparison |

---

## Adversarial Debate Protocol

1. Adversary reads ALL project files, challenges every claim
2. Defender responds point by point, conceding honestly where needed
3. Challenges categorized: conceded, partially sustained, rejected
4. Fixes implemented for sustained challenges
5. Re-evaluated in subsequent round
6. Target: 0 fatal findings across all rounds

---

## Tool Dependency Notes

- **PySR** — Heavy (Julia runtime). Run on remote machine.
- **emcee** — Lightweight MCMC. Can run locally for 2-3D.
- **scikit-learn** — Gaussian Processes for null tests.
- **astropy** — FITS file handling.
- **tectonic** — Portable LaTeX. Download binary, no texlive needed.
- **WebPlotDigitizer** — Manual digitization of published figures (when digital tables unavailable).

---

## Publishing Checklist

- [ ] Paper compiled (0 errors, 0 overfull boxes)
- [ ] All numbers verified against CSV outputs
- [ ] GitHub pushed with full history
- [ ] Zenodo: zip source files + paper PDF → upload → publish → get DOI
- [ ] OSF: create project hub, link GitHub + Zenodo
- [ ] Medium: general public article (if desired)
- [ ] RNAAS: short note for novel singular findings (if applicable)
- [ ] Debate log updated with all rounds
- [ ] PROGRESS.md updated with final results
- [ ] README.md updated with quick start and validation summary

---

## Zenodo Fields Template

| Field | Value |
|-------|-------|
| **Title** | [Descriptive title with key result] |
| **Description** | [Abstract + methods summary + validation summary] |
| **Keywords** | [5-8 relevant terms] |
| **License** | MIT |
| **Resource type** | Preprint |
| **Related identifiers** | GitHub URL (is supplemented by) |

---

## Medium Article Structure

1. Hook (Vera Rubin moment, big question)
2. Background (what's the problem, why it matters)
3. Method (symbolic regression in simple terms)
4. Discovery (the form found, with numbers)
5. Stress tests (all the validation, in plain English)
6. Critical tests (specific predictions tested)
7. What it means (neither camp wins, both have sharper target)
8. Footer (data sources, GitHub, Zenodo DOIs, debate count)
