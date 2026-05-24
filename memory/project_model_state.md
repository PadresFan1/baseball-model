---
name: project-model-state
description: Current live model state, backtest round history, and key findings
metadata:
  type: project
---

Live model (model.py) is on v2 since 2026-05-21. O/U betting disabled (OU_BETTING_ENABLED = False) — backtest showed anti-predictive signal at all thresholds.

**Backtest history:**
- Rounds 1-5: team-level rolling stats, max +16.3% ROI (overfitted)
- Round 6: O/U tracking added
- Round 7: park factors added to lambda calculation
- Round 8: holdout split discovered overfitting — 2025 holdout: -1.9% ROI, p=1.0 (not significant)
- Meta model (logistic regression over 4 validated metrics): Brier 0.2449 on 2025 holdout (better than 0.25 baseline but ~49.7% win rate, below 52.4% break-even)
- Round 9 (active): player-level rebuild — see [[project-round9-player-level]]

**4 validated metrics with genuine out-of-sample signal:**
woba, xwoba_bat, xfip, xwoba_pit

**Key decision: why team-level failed:**
Team-level season-to-date Statcast metrics are too coarse and too public. The market has priced them. Win rate stuck at ~49.7% even with market logit anchor.

**How to apply:** Use this to orient any discussion about model direction or data strategy.
