"""
Simulation for "The Wrong Pairameter"

Demonstrates the gap between naive own-treatment estimators and
dyadic cell means in a trust game with two-sided treatment.
"""

import numpy as np
import pandas as pd

# ── Parameters ──────────────────────────────────────────────────
np.random.seed(42)

N_STUDIES = 1000
N_PER_ROLE = 100          # 100 first movers, 100 second movers per study
TREAT_PROB = 0.5
NOISE_SD = 0.3

ENDOWMENT = 6
MULTIPLIER = 3

# Treatment effects on decisions
SEND_CONTROL = 3.6
SEND_TREATED = 3.9
RETURN_CONTROL = 0.33
RETURN_TREATED = 0.40


# ── Payoff functions ────────────────────────────────────────────
def fm_payoff(s, rho):
    return ENDOWMENT - s + MULTIPLIER * s * rho

def sm_payoff(s, rho):
    return MULTIPLIER * s * (1 - rho)

def total_surplus(s):
    return ENDOWMENT + (MULTIPLIER - 1) * s


# ── Theoretical cell means ─────────────────────────────────────
print("Theoretical cell means")
print("=" * 70)
header = f"{'Pair':<8} {'Send':>6} {'Return':>8} {'FM pay':>8} {'SM pay':>8} {'Surplus':>8}"
print(header)
print("-" * 70)

for label, s, rho in [
    ("C-C", SEND_CONTROL, RETURN_CONTROL),
    ("C-T", SEND_CONTROL, RETURN_TREATED),
    ("T-C", SEND_TREATED, RETURN_CONTROL),
    ("T-T", SEND_TREATED, RETURN_TREATED),
]:
    print(f"{label:<8} {s:>6.1f} {rho:>8.2f} {fm_payoff(s, rho):>8.3f} "
          f"{sm_payoff(s, rho):>8.3f} {total_surplus(s):>8.1f}")

print()

# ── Monte Carlo ─────────────────────────────────────────────────
naive_diffs = []
cell_means = {key: [] for key in ["CC", "CT", "TC", "TT"]}

for _ in range(N_STUDIES):
    # Draw treatment assignments
    z1 = np.random.binomial(1, TREAT_PROB, N_PER_ROLE)
    z2 = np.random.binomial(1, TREAT_PROB, N_PER_ROLE)

    # Generate decisions with noise
    send = np.where(z1, SEND_TREATED, SEND_CONTROL) + np.random.normal(0, NOISE_SD, N_PER_ROLE)
    send = np.clip(send, 0, ENDOWMENT)

    ret = np.where(z2, RETURN_TREATED, RETURN_CONTROL) + np.random.normal(0, NOISE_SD / 10, N_PER_ROLE)
    ret = np.clip(ret, 0, 1)

    # ── Naive estimator: compare FM payoff by own treatment ─────
    # Random pairing for realized outcomes
    perm = np.random.permutation(N_PER_ROLE)
    ret_matched = ret[perm]
    fm_pay = fm_payoff(send, ret_matched)

    naive_t = fm_pay[z1 == 1].mean()
    naive_c = fm_pay[z1 == 0].mean()
    naive_diffs.append(naive_t - naive_c)

    # ── Synthetic pairing: all FM x SM within each cell ─────────
    for z, w, key in [(0, 0, "CC"), (0, 1, "CT"), (1, 0, "TC"), (1, 1, "TT")]:
        s_cell = send[z1 == z]
        r_cell = ret[z2 == w]
        if len(s_cell) == 0 or len(r_cell) == 0:
            continue
        # Mean over all synthetic pairs in this cell
        mean_s = s_cell.mean()
        mean_r = r_cell.mean()
        cell_means[key].append(fm_payoff(mean_s, mean_r))

# ── Results ─────────────────────────────────────────────────────
print("Monte Carlo results (1,000 studies)")
print("=" * 70)

mc_naive = np.mean(naive_diffs)
mc_TT = np.mean(cell_means["TT"])
mc_CC = np.mean(cell_means["CC"])
mc_TC = np.mean(cell_means["TC"])
mc_CT = np.mean(cell_means["CT"])

results = [
    ("Naive own-treatment diff (FM payoff)", mc_naive),
    ("Synthetic: T-T minus C-C", mc_TT - mc_CC),
    ("Synthetic: T-C minus C-C", mc_TC - mc_CC),
    ("Synthetic: C-T minus C-C", mc_CT - mc_CC),
]

for label, val in results:
    print(f"  {label:<45} {val:>8.3f}")
