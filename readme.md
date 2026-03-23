# The Wrong Pairameter

Experiments on strategic interactions often report individual-level treatment effects on pair-level outcomes. In a trust game, for instance, the standard analysis compares treated first movers to control first movers, conditioning only on the first mover's own treatment status. But first-mover payoffs depend on both sides. The naive estimator averages over the partner's treatment status, producing a weighted mixture of regime-level contrasts that does not correspond to any deployable policy.

This repository provides a simulation demonstrating the gap between the naive own-treatment estimator and regime-level dyadic cell means in a trust game with two-sided treatment. The simulation shows that the naive estimator can report near-zero effects even when the fully treated pair is substantially better off than the fully control pair.

## Companion essay

[The Wrong Pairameter](https://www.gojiberries.io/the-wrong-pairameter/) on Gojiberries.

## Running the simulation

```bash
python simulate_trust_game.py
```

Requires Python 3.8+ and NumPy. No other dependencies.

## What the simulation does

A trust game with endowment 6 and a tripling multiplier. Treatment increases sending (3.6 → 3.9) and returning (33% → 40%). The exclusion restriction holds exactly: treatment affects only each player's own decision rule.

For each of 1,000 simulated studies (100 participants per role, independent treatment at probability 0.5):

1. The **naive estimator** compares first-mover payoffs by own treatment status, ignoring partner treatment.
2. **Synthetic pairing** recovers expected payoffs for each of the four ordered pair types (C–C, C–T, T–C, T–T).

## Key result

| Quantity | MC mean |
|---|---|
| Naive own-treatment difference in FM payoff | 0.027 |
| Synthetic pair: T–T minus C–C | 0.814 |
| Synthetic pair: T–C minus C–C | −0.003 |
| Synthetic pair: C–T minus C–C | 0.755 |

The naive estimator reports ≈0.03. The regime where both sides are treated shows a gain of ≈0.81. The discrepancy arises because surplus creation (driven by sending) and surplus distribution (driven by returning) flow through different roles. The naive estimator mixes these channels together, and they cancel.

## License

MIT