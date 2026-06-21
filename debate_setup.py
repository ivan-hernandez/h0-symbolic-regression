"""Adversarial debate: challenge and defend our H0 symbolic regression results.

Usage:
    python3 debate_setup.py [--rounds N]

The debate proceeds as rounds between an adversary (challenger) and
defender, each reviewing the analysis files and formulating arguments.
"""
import sys, os, json

# Files the agents should read
ANALYSIS_FILES = [
    "AGENTS.md",
    "data.py",
    "m_z_evolution.py",
    "all_extensions.py",
    "extension_summary.py",
    "lcdm_fit.py",
    "reject_all.py",
    "pantheon_cov.py",
    "desi_dr2_data.py",
]

def write_debate_prompt(role, opponent_args=None, round_num=1):
    """Generate prompt for a debate agent."""
    base = f"""You are a cosmology expert in an adversarial debate about this paper's result:
H0 = 68.0 ± 0.8 km/s/Mpc from symbolic regression of CC+BAO+DESI+SNe data.

Your ROLE: {role}

The analysis files are in /home/ivan/general-conversation/. Read the key files first:
{ANALYSIS_FILES}

"""
    if role == "adversary":
        base += """Your job is to CHALLENGE the result. Find every weakness:
- Methodological issues (symbolic regression overfitting, selection effects)
- Data issues (CC systematics, BAO r_d dependence, SN covariance treatment)
- Statistical issues (profile likelihood coverage, parameter degeneracies)
- Logical issues (does free M really test the calibration bias?)
- External contradictions (JWST Cepheid validation, Riess+2025)
- Hidden assumptions

Be aggressive but rigorous. Cite specific numbers, chi2 values, and file locations.
Your goal: force the defender to concede at least one point, or show the result is flawed.
"""
    else:
        base += """Your job is to DEFEND the result against the adversary's challenges.

Opponent's arguments:
{opponent_args}

For each argument:
1. Acknowledge valid points
2. Show why they don't change the conclusion
3. Provide counter-evidence from the code/data
4. If the adversary makes a genuinely good point, concede it and explain why it doesn't matter

Be rigorous. Cite specific chi2 values, test results, and file line numbers.
"""
    return base

ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--rounds" else 3

print("=" * 70)
print("  ADVERSARIAL DEBATE SETUP")
print("=" * 70)
print(f"\n  Analysis files: {len(ANALYSIS_FILES)}")
print(f"  Max rounds: {ROUNDS}")
print(f"\n  Adversary brief: challenge H0=68 result")
print(f"  Defender brief: defend against all challenges")
print(f"\n  To run: use two Task agents in parallel")
print(f"  1st: adversary reads code and formulates challenges")
print(f"  2nd: defender reads adversary output and responds")
print(f"  Continue for {ROUNDS} rounds or until stalemate")
print(f"\n  Debate log: /tmp/debate_log.md")
