# Data-Driven Astrophysics — The Adversarial SR Method

A battle-tested methodology for applying symbolic regression to astrophysical
problems, developed and validated across 5 published papers, 8 adversarial debate
rounds, and 30+ challenges (0 fatal).

**Core principle:** Discover functional forms with minimal theoretical priors,
then attack every claim with an independent critic before anyone else can.

---

## The Five Phases

### Phase 1: Discovery

**Goal:** Find the simplest mathematical form that describes the data.

- Parse public data into `(X, y)` with proper errors. Verify against published tables.
- Run PySR with `model_selection="accuracy"`, ≥3 random seeds, 200 iterations.
- Identify Pareto-optimal forms by complexity (CPX3, CPX5, CPX7).
- Select the best 2-parameter form by stability across seeds and AIC.
- Do NOT extrapolate the discovered form beyond its data range.

**Key decisions:**
- Log-log space for wide dynamic range problems (avoids high-value domination of MSE)
- Error model: `sigma = max(e_gobs, 0.1*y)` for intrinsic scatter
- AIC = χ² + 2k (not n·ln(χ²/n) for known errors)
- Weak boundary prior to suppress pathological solutions without biasing result
- Heavy computation on remote machine (≥12 cores, ≥15 GB RAM)

### Phase 2: Validation

**Goal:** Prove the form is stable and not an artifact of methodology.

1. **Multi-seed:** ≥3 independent PySR runs → verify equation convergence
2. **Bootstrap:** ≥200 resamples → parameter distributions (68% CL)
3. **Holdout:** ≥10-fold galaxy-wise splits → train vs test RMS
4. **M/L sweep:** ≥16 mass-to-light grid combinations
5. **Blind test:** Generate mock data from known models, verify recovery
6. **Per-unit:** Fit form to each individual data unit (galaxy, etc.)
7. **Literature:** Direct comparison with published results
8. **Integration accuracy:** Verify numerical integration converges (<0.01 mag)

**Red flags:**
- Parameters depend on dynamic range → not a universal form (caveat required)
- χ²_red ≪ 1 → errors overestimated
- χ²_red ≫ 1 → model rejected (add intrinsic scatter)
- Per-unit parameters span more than measurement range → not independently verifiable

### Phase 3: Extension

**Goal:** Test the form on independent data and against theory predictions.

1. **Cross-sample:** Same form on a completely different instrument/telescope
2. **Cross-regime:** Extend to higher/lower mass, different environment
3. **Theory tests:** Specific falsifiable predictions (asymptote, EFE, hooks)
4. **Parameter dependence:** Correlate form parameters with physical properties
5. **Forecast:** When will future data distinguish this form from alternatives?

**What NOT to do:**
- Don't extrapolate polynomials beyond their fit range
- Don't claim "model-independent" for BAO-derived H(z) (depends on r_s)
- Don't call synthetic toy curves "simulation data"

### Phase 4: Adversarial Debate (The Novel Part)

**Goal:** Catch every fatal error before publication.

**Protocol:**
1. Adversary reads ALL project files, data, and code
2. Adversary challenges EVERY claim — data, method, interpretation
3. Defender responds point by point, conceding honestly where needed
4. Challenges categorized: **conceded** | **partially sustained** | **rejected**
5. Fixes implemented for all sustained challenges
6. Re-evaluated in subsequent round
7. Continue until 0 fatal findings remain
8. Document all challenges and resolutions in debate log

**Adversary's arsenal:**
- Check data against published tables for duplicates/errors
- Verify units and dimensional consistency
- Test sensitivity to arbitrary thresholds and fitting choices
- Check for extrapolation beyond data range
- Identify circular reasoning (e.g., RM masses calibrated to M-σ)
- Question whether synthetic data is being passed off as real simulation
- Test all "both methods are valid" claims — are they really?

**What the debate has caught (real examples):**
- M/NGC duplicate pairs with conflicting values (BH catalog)
- Polynomial integration bug — Riemann sum × Simpson gave 0.24 mag error
- 28 galaxies with degenerate parametric null (GP null fixed it)
- Column naming bugs (halfmassrad ≠ vmaxrad)
- Wrong variable (M*/Re² ≠ surface brightness for FP)
- Circular inclusion of RM masses calibrated to the relation being measured
- Aspirational survey factors replaced with SRD-cited numbers

### Phase 5: Publication

**Goal:** Permanent, citable, reproducible.

1. **Write:** LaTeX in `aastex631.cls` two-column format
2. **Compile:** `tectonic paper.tex` (portable, no texlive needed)
3. **Publish:** `python3 publish.py "Title" "Description" paper_dir/` (one command)
4. **Archive:** Zenodo handles DOI registration automatically
5. **Hub:** OSF component created automatically with DOI links
6. **Code:** All pushed to GitHub with full commit history

**Publication stack:**
```
publish.py → Zenodo (DOI) + OSF (component + wiki) ← GitHub (code)
```

---

## What Makes Novel Work vs Competent Re-analysis

| Novel | Not Novel |
|-------|-----------|
| Discovered form beats existing theories | Re-fit known relation with new data |
| Falsifiable prediction for future surveys | Confirmed known result with better precision |
| Methodological contribution (adversarial protocol) | Applied standard method to new dataset |
| Exposed previously hidden systematic | Found same slope as literature |

**The test:** If the conclusion changes what someone would assume or do, it's novel.
If it says "consistent with literature," it's practice.

---

## File Structure

```
project/
├── AGENTS.md              # Project state + constraints (living)
├── PROPOSAL.md             # Original plan (static, written before work)
├── PROGRESS.md             # Actual accomplishments (living)
├── METHODOLOGY.md          # This file — the pipeline
├── README.md               # Quick start + validation summary
├── publish.py              # One-click Zenodo+OSF publisher
├── paper/
│   ├── paper.tex
│   ├── paper.pdf
│   └── fig_*.pdf
├── analysis/               # Generated outputs (CSVs, figures)
└── scripts/                # Analysis scripts
```

---

## Remote Compute Protocol

- All heavy computation on remote machine (Tailscale SSH, 12 cores, 15 GB)
- Lightweight scripts (<5 min runtime) can run locally
- Always `scp` scripts to remote, never run heavy computation locally
- Remote path: `100.121.64.70:~/h0-sr/`

---

## API Tokens

Stored in `publish.py`:
- **Zenodo:** Deposit creation, file upload, publishing
- **OSF:** Component creation, wiki editing

To use: copy `publish.py` to any project. No additional configuration needed.

---

## Debate Log Archive

All debate rounds documented at `/tmp/rar_debate_log.md` (8 rounds, 30+ challenges).
Each round identifies which claims were conceded, partially sustained, or rejected,
and what fixes were implemented before the next round.

---

## Lessons Learned

1. **The debate catches errors you cannot see.** An independent critic looking at
   your code will find mistakes you've been blind to for weeks.
2. **"Both methods are valid" for a 7 km/s shift is never valid.** If methodology
   changes the answer by more than the error bar, you haven't converged.
3. **"Consistent with literature" is the most dangerous phrase in science.**
   It usually means "I haven't found anything wrong yet," not "this is correct."
4. **Synthetic toy models are not simulations.** Don't call them simulations.
5. **The Riemann sum is not good enough at low z.** Use Simpson's rule for
   cosmological distance integrals.
6. **Hard-code data once, then verify it.** Raw data tables from papers contain
   duplicates and errors. Trust nothing until checked.
7. **Every "universal" constant should be tested per-unit.** a₀ couldn't be
   measured per galaxy, despite being called "universal" for 40 years.
