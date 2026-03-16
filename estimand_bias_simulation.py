import numpy as np
import pandas as pd


# -----------------------------
# Trust-game simulation script
# -----------------------------
# Purpose:
#   Show why a role-level estimator is the wrong tool when the estimand of
#   interest is a dyadic pair-type contrast such as TT - CC.
#
# Design mimicked from the paper:
#   - treatment assigned at the individual level within session
#   - all choices made before matching
#   - matching random and implemented after choices are recorded
#   - no one observes partner treatment before deciding
#
# Estimand used here:
#   theta = E[payoff_to_trustor | T-T] - E[payoff_to_trustor | C-C]
#
# "Naive" estimator shown here:
#   average trustor payoff among treated trustors minus average trustor payoff
#   among control trustors, collapsing over trustee treatment.
#   This is not theta; it is a weighted average over mixed pair types.
#
# "Correct" estimator shown here (when individual decisions are retained):
#   reconstruct expected pair-type means by synthetic random rematching within
#   session and treatment cells.


def simulate_study(
    seed: int,
    session_sizes=None,
    mu_send_control: float = 3.6,
    tau_send: float = 0.30,
    sd_send: float = 0.80,
    mu_return_control: float = 0.33,
    tau_return: float = 0.07,
    sd_return: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one study.

    Each participant provides both a trustor decision (how much to send)
    and a trustee decision rule summarized here as a return share.
    Matching happens only after all decisions are made.
    """
    if session_sizes is None:
        # 302 participants total, close to Study 1 in the paper
        session_sizes = [22] * 13 + [16]

    rng = np.random.default_rng(seed)
    participant_rows = []
    realized_match_rows = []
    pid = 0

    for session_id, n in enumerate(session_sizes):
        z = rng.integers(0, 2, size=n)  # individual treatment assignment

        send = np.clip(
            rng.normal(mu_send_control + tau_send * z, sd_send),
            0.0,
            6.0,
        )
        return_share = np.clip(
            rng.normal(mu_return_control + tau_return * z, sd_return),
            0.0,
            1.0,
        )

        # Random post-decision matching. We create a near-derangement so no one
        # is matched to themselves.
        perm = rng.permutation(n)
        for i in range(n):
            if perm[i] == i:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]

        for i in range(n):
            participant_rows.append(
                {
                    "id": pid + i,
                    "session": session_id,
                    "Z": int(z[i]),
                    "send": float(send[i]),
                    "return_share": float(return_share[i]),
                }
            )

        for i in range(n):
            j = perm[i]
            payoff_trustor = 6.0 - send[i] + 3.0 * send[i] * return_share[j]
            payoff_trustee = 3.0 * send[i] * (1.0 - return_share[j])
            realized_match_rows.append(
                {
                    "session": session_id,
                    "trustor_id": pid + i,
                    "trustee_id": pid + j,
                    "Z1": int(z[i]),
                    "Z2": int(z[j]),
                    "send": float(send[i]),
                    "return_share_partner": float(return_share[j]),
                    "payoff_trustor": float(payoff_trustor),
                    "payoff_trustee": float(payoff_trustee),
                }
            )

        pid += n

    participants = pd.DataFrame(participant_rows)
    realized_matches = pd.DataFrame(realized_match_rows)
    return participants, realized_matches


def theoretical_cell_means(
    mu_send_control: float = 3.6,
    tau_send: float = 0.30,
    mu_return_control: float = 0.33,
    tau_return: float = 0.07,
) -> pd.DataFrame:
    """Population cell means with no sampling noise."""
    rows = []
    for z1 in (0, 1):
        for z2 in (0, 1):
            send = mu_send_control + tau_send * z1
            rho = mu_return_control + tau_return * z2
            payoff_trustor = 6.0 - send + 3.0 * send * rho
            payoff_trustee = 3.0 * send * (1.0 - rho)
            rows.append(
                {
                    "Z1": z1,
                    "Z2": z2,
                    "send": send,
                    "return_share": rho,
                    "payoff_trustor": payoff_trustor,
                    "payoff_trustee": payoff_trustee,
                    "total_surplus": payoff_trustor + payoff_trustee,
                }
            )
    return pd.DataFrame(rows)


def synthetic_pair_means(participants: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct expected pair-type means by synthetic rematching.

    This does not require the realized match IDs. It only requires participant-level
    decisions, treatment status, and session membership.
    """
    rows = []

    for session_id, g in participants.groupby("session"):
        for z1 in (0, 1):
            trustors = g.loc[g["Z"] == z1, "send"].to_numpy()
            if trustors.size == 0:
                continue
            for z2 in (0, 1):
                trustees = g.loc[g["Z"] == z2, "return_share"].to_numpy()
                if trustees.size == 0:
                    continue

                send_matrix = trustors[:, None]
                rho_matrix = trustees[None, :]
                payoff_trustor = 6.0 - send_matrix + 3.0 * send_matrix * rho_matrix
                payoff_trustee = 3.0 * send_matrix * (1.0 - rho_matrix)

                rows.append(
                    {
                        "session": session_id,
                        "Z1": z1,
                        "Z2": z2,
                        "n_pairs": payoff_trustor.size,
                        "sum_payoff_trustor": float(payoff_trustor.sum()),
                        "sum_payoff_trustee": float(payoff_trustee.sum()),
                        "mean_send": float(trustors.mean()),
                        "mean_return_share": float(trustees.mean()),
                    }
                )

    tmp = pd.DataFrame(rows)
    out = []
    for (z1, z2), g in tmp.groupby(["Z1", "Z2"]):
        w = g["n_pairs"].to_numpy()
        out.append(
            {
                "Z1": z1,
                "Z2": z2,
                "send": np.average(g["mean_send"], weights=w),
                "return_share": np.average(g["mean_return_share"], weights=w),
                "payoff_trustor": g["sum_payoff_trustor"].sum() / w.sum(),
                "payoff_trustee": g["sum_payoff_trustee"].sum() / w.sum(),
                "total_surplus": (
                    g["sum_payoff_trustor"].sum() + g["sum_payoff_trustee"].sum()
                )
                / w.sum(),
            }
        )
    return pd.DataFrame(out).sort_values(["Z1", "Z2"]).reset_index(drop=True)


def naive_estimator(realized_matches: pd.DataFrame) -> float:
    """Difference in trustor payoff by own treatment, collapsing over partner treatment."""
    treated = realized_matches.loc[realized_matches["Z1"] == 1, "payoff_trustor"].mean()
    control = realized_matches.loc[realized_matches["Z1"] == 0, "payoff_trustor"].mean()
    return float(treated - control)


def cell_contrast(cell_means: pd.DataFrame, outcome: str = "payoff_trustor") -> dict:
    pivot = cell_means.pivot(index="Z1", columns="Z2", values=outcome)
    mu00 = float(pivot.loc[0, 0])
    mu01 = float(pivot.loc[0, 1])
    mu10 = float(pivot.loc[1, 0])
    mu11 = float(pivot.loc[1, 1])
    return {
        "mu00": mu00,
        "mu01": mu01,
        "mu10": mu10,
        "mu11": mu11,
        "TT_minus_CC": mu11 - mu00,
        "TC_minus_CC": mu10 - mu00,
        "CT_minus_CC": mu01 - mu00,
        "own_treatment_mixture_if_p_half": 0.5 * (mu11 - mu01) + 0.5 * (mu10 - mu00),
    }


def monte_carlo(n_rep: int = 1000) -> pd.DataFrame:
    rows = []
    for seed in range(n_rep):
        participants, realized_matches = simulate_study(seed=seed)
        synthetic = synthetic_pair_means(participants)
        theoretical = theoretical_cell_means()
        truth = cell_contrast(theoretical)
        synthetic_contrasts = cell_contrast(synthetic)

        rows.append(
            {
                "naive_own_treatment_diff": naive_estimator(realized_matches),
                "synthetic_TT_minus_CC": synthetic_contrasts["TT_minus_CC"],
                "synthetic_TC_minus_CC": synthetic_contrasts["TC_minus_CC"],
                "synthetic_CT_minus_CC": synthetic_contrasts["CT_minus_CC"],
                "truth_TT_minus_CC": truth["TT_minus_CC"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)

    theoretical = theoretical_cell_means()
    print("\nTheoretical pair-type means (ordered pairs: first mover, second mover):")
    print(theoretical)

    print("\nTheoretical contrasts for trustor payoff:")
    print(cell_contrast(theoretical, outcome="payoff_trustor"))

    participants, realized_matches = simulate_study(seed=123)
    synthetic = synthetic_pair_means(participants)

    print("\nSynthetic expected pair-type means from one simulated study:")
    print(synthetic)

    mc = monte_carlo(n_rep=1000)
    summary = pd.DataFrame(
        {
            "mean": mc.mean(),
            "sd": mc.std(),
        }
    )
    print("\nMonte Carlo summary over 1000 simulated studies:")
    print(summary)
