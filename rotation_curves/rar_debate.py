"""RAR Adversarial Debate: challenge and defend our symbolic regression results.

Follows the Hubble project's debate framework but applied to the RAR/rotation curves.
"""
import sys

RAR_FILES = [
    "rotation_curves/parse_sparc.py",
    "rotation_curves/rar_analysis.py",
    "rotation_curves/rar_sr.py",
    "rotation_curves/rar_linear.py",
    "rotation_curves/extension_ml_test.py",
    "rotation_curves/extension_blind_test.py",
    "rotation_curves/explore_ml.py",
    "rotation_curves/extend_rar_analysis.py",
    "rotation_curves/literature_comparison.py",
    "rotation_curves/simulation_comparison.py",
    "rotation_curves/efe_coordinates.py",
    "rotation_curves/joint_sr_lensing.py",
    "rotation_curves/joint_pysr_lensing.py",
    "rotation_curves/hook_search.py",
    "rotation_curves/test_mond_asymptote.py",
    "rotation_curves/analysis/model_comparison.csv",
    "rotation_curves/analysis/ml_sensitivity.csv",
    "rotation_curves/analysis/blind_test_results.csv",
    "rotation_curves/analysis/holdout_results.csv",
    "rotation_curves/analysis/multiseed_equations.csv",
    "rotation_curves/analysis/bootstrap_rar.csv",
    "rotation_curves/analysis/per_galaxy_cpx5_params.csv",
    "rotation_curves/analysis/hook_search_results.csv",
]

ADVERSARY_PROMPT = """You are a modified gravity expert in an adversarial debate about this project's RAR results.

Your ROLE: ADVERSARY (challenger)

Key files to read (all under /home/ivan/general-conversation/):
{RAR_FILES}

The project used symbolic regression (PySR) on SPARC rotation curve data (175 galaxies, 3389 points) plus Mistele+2024 weak-lensing data (11 binned points, extending RAR by 2.5 dex) to discover the Radial Acceleration Relation (RAR) functional form g_obs = F(g_bar). Key findings:
- SR-discovered form CPX5 (`log_gobs = A + B/log_gbar`) is the best model across 6.5 dex — PySR score 3.85 vs next best 0.48
- CPX5 beats MOND IFs on AIC by ΔAIC ≈ 1950 (SPARC full dataset)
- CPX5 parameters nearly unchanged when lensing data is added (∆a ≈ 1σ)
- MOND √gbar asymptote not required: c = 0.10 ± 0.15, Δχ² = 0.18 for adding it
- Bootstrap uncertainty: CPX5 a = -17.060 ± 0.133, b = -72.71 ± 1.38
- Per-galaxy CPX5 fits: mean RMS 0.077 dex, but individual parameters poorly constrained
- MOND a₀ varies from 1.9e-11 to 1.1e-10 depending on M/L assumption
- Blind test: PySR recovers MOND-like family from mock MOND data
- EFE with 3D SIMBAD positions: ρ = +0.106 (wrong direction for MOND), p = 2e-9
- FIRE-2 hooks: 68% of SPARC galaxies show non-monotonic RAR tracks
- Literature: CPX5 consistent with Desmond+2023 ESR on SPARC alone; extends to lensing regime

Your job is to CHALLENGE these conclusions. Find every weakness:
- CPX5 asymptotic failure: `log_gobs = A + B/log_gbar` asymptotes to constant gobs at low gbar. The lensing data shows gobs ∝ √gbar (slope 0.5). How can CPX5 be correct when it has the wrong asymptote?
- The "c = 0.10 ± 0.15" for the MOND term is based on binned data (21 points). That's not enough to constrain the slope. The error bars are huge.
- PySR only ran on 21 binned points — why not the full 2706-point SPARC RAR + lensing? The binned data loses information.
- AIC comparison: χ² for SPARC full data gives χ²_red ≈ 0.04 and all models have similar χ². The apparent AIC differences are tiny and driven by uniform weights, not proper errors.
- EFE test gave ρ = +0.106. But that's against CPX5 residuals, not MOND residuals. Maybe MOND residuals DO correlate with isolation. Test it.
- The 3D EFE test uses SIMBAD positions with large distance uncertainties (flow model distances, 20-30% errors). The correlation is meaningless.
- FIRE-2 hooks: 68% with "hooks"? With only 5-16 data points per galaxy, these are just noise. A proper statistical test would show no more hooks than expected from noise.
- Multi-seed PySR only ran on SPARC alone. The combined SPARC + lensing PySR was only run once. Could be a lucky run.
- Physical meaning: Can you write CPX5 as a physical theory? MOND has a clear physical interpretation. CPX5 is just a curve fit.
- Extrapolation: CPX5 predicts gobs → constant at low gbar. This is physically impossible (would require infinite mass). The Mistele lensing data already shows slope ~0.5. Doesn't this mean CPX5 must break?

Be aggressive but rigorous. Cite specific numbers, file locations, and analysis results.
Your goal: force the defender to concede at least one substantive point.
"""

DEFENDER_PROMPT = """You are a modified gravity expert defending this project's RAR results in an adversarial debate.

Your ROLE: DEFENDER

Key files to read (all under /home/ivan/general-conversation/):
{RAR_FILES}

The adversary has raised the following challenges:
{adversary_args}

The project used symbolic regression (PySR) on SPARC rotation curve data (175 galaxies, 3391 points) and weak-lensing data (Mistele+2024, 11 bins) across 6.5 dex in gbar. Key findings:
- CPX5 (`log_gobs = A + B/log_gbar`) dominates joint PySR: score 3.85 vs next best 0.48
- CPX5 vs MOND: ΔAIC ≈ 1950 on full SPARC dataset
- CPX5 parameters consistent between SPARC-only and joint fits (Δa ≈ 1σ)
- MOND asymptote test: c = 0.10 ± 0.15, Δχ² = 0.18 (p = 0.67) — adding √gbar does not improve fit
- Bootstrap: CPX5 parameters extremeley robust (±0.13, ±1.4 on 200 resamples)
- Multi-seed: CPX5 identical across 3 seeds; CPX7 has minor seed variation (handled separately)
- Holdout: test RMS 0.23 dex vs train 0.22 dex
- Blind test: PySR recovers MOND-like family from mock data
- Per-galaxy CPX5 fits: 171/175 galaxies fit individually; RMS 0.077 dex mean
- M/L: a₀ varies factor 6; SR parameters vary 7-16%
- EFE (3D SIMBAD): ρ = +0.106 (p = 2e-9), opposite to MOND prediction
- Gas fraction anti-correlation: strongest residual pattern (ρ = -0.31)
- Hooks: 68% of galaxies, mostly in poorly-sampled RCs
- Literature: consistent with Desmond+2023 ESR conclusion

For each challenge:
1. Acknowledge valid points
2. Show why they don't change the conclusion
3. Provide counter-evidence from the code/data
4. If genuinely good point, concede it honestly

Be rigorous. Cite specific numbers, test results, and file locations.
Your goal: defend the robustness of the SR RAR discovery.
"""


if __name__ == "__main__":
    round_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    print("=" * 70)
    print("  RAR ADVERSARIAL DEBATE SETUP")
    print("=" * 70)
    print(f"\n  Analysis files: {len(RAR_FILES)}")
    print(f"  Round: {round_num}")
    print(f"\n  Adversary brief: challenge RAR SR result")
    print(f"  Defender brief: defend against all challenges")
    print(f"\n  Run using two Task agents in parallel:")
    print(f"  1. Adversary reads code and formulates challenges")
    print(f"  2. Defender reads adversary output and responds")
    print(f"\n  Log: /tmp/rar_debate_log.md")
