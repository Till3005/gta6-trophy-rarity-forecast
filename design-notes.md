# Design notes — trophy rarity forecast

## Problem

Forecast each GTA VI trophy's Exophase earn rate (% of tracked players) at a
`target` timestamp, as five quantiles. Scored by mean pinball loss on the logit
scale (rates clamped to 0.01–99.99), median weight dominates. Two regimes:

- **Practice (p00/p01):** 90-minute horizons on launch night; the answer is a
  post-refresh page, so the last read is stale. These mainly test format.
- **Final (f00):** ~36h of reads, target ~6.5 days out. This is the real game.

## Why the naive approaches lose

- **Flat (last read):** ignores that tracked-player count `N` floods from 638
  grinders to tens of thousands of mixed players; a rate that is 5% on launch
  night will *not* still be ~5% a week later — casuals dilute hard trophies
  down, trivial ones stay ~100% (new players earn them too).
- **Pure trend extrapolation:** fits the launch-night surge and overshoots;
  log-time curves saturate and nobody's week-1 rate is 80%.

## Model

Work on `z = logit(r/100)`; clamp rates to [0.01, 99.99] first.

1. **Dedup.** Reads arrive between refreshes; consecutive identical
   `(players_tracked, rates)` rows are one snapshot. Each distinct
   `(time, rate, N)` is an observation.
2. **Trend estimator.** Recency-weighted OLS (12h half-life) of `z` on
   `x = ln(1 + hours since first read)`, windowed to the last 24h (falls back
   to 48h/7d if too few refreshes), slope clamped to [−0.5, 1.0]. Extrapolate
   by anchoring at the last observed `z` and stepping `slope · Δx`.
   `Δx` over a 6.5-day horizon is small — log-time progress saturates — which
   is intentional: the fit captures *near-term* momentum, not week-scale level.
3. **Category prior (the long-horizon backbone).** Parse name/description/tier
   into a kind and a week-1 endpoint percent, calibrated for a completionist-
   skewed tracker population absorbing a casual flood:
   - chapter trophies: 40% → 13% linearly by chapter number (early chapters
     tens of percent, final chapter single-digit-plus);
   - "Complete the final mission" 10%, character-questline finales 16%,
     generic story missions ("Complete <Mission>") 20%;
   - collect/"all/every/discover" trophies 6.5/4.5/3% by bronze/silver/gold;
   - numeric grinds ("Win 20 races", "250 stunts") 5/3.5/2.5%;
   - misc bronze/silver/gold 12/7/4.5%; "try each … once" easy 32%;
   - 100%-completion 1.8%, platinum 1.6% (platinum ≤ 100% is enforced);
   - secrets share their inferred kind (descriptions usually reveal them) with
     slightly wider bands.
   **Observed-aware guard:** the prior may pull a median down at most 0.9 logit
   below the last observed rate — trophies already above their kind's prior
   (e.g. a 100%-at-launch intro mission) keep their level instead of being
   dragged to a generic 20%.
4. **Blend.** `z_med = w·z_trend + (1−w)·z_prior`, `w = 1/(1 + horizon_days/2)`:
   ≈0.97 at 90 minutes (data rules), ≈0.24 at 6.5 days (prior rules, trend
   contributes momentum). This reproduces the intuition that a story trophy at
   5% after 36h lands around 20–30% in week one while platinum lands ~1%.
5. **Structure passes.** Chapter medians forced non-increasing by chapter
   number; the story-final trophy capped at the last chapter's median.
6. **Spread.** Half-width
   `s = 0.14 + 0.24·√horizon_days + 0.10·min(|z_trend−z_prior|, 2)·(1−w)`,
   multiplied by `1 + 0.35·exp(−m/2)` for edge rates (m = distance to 0/100)
   and ×1.08 for secrets, floored at 0.15, capped at 2.2. Trend–prior
   disagreement widens the band in proportion to the prior's actual influence
   `(1−w)`, so short horizons stay tight-ish (≈0.2–0.35, deliberately wider
   than the house baseline's 0.125 because the answer hides a refresh) and the
   6.5-day 90% band lands ≈0.75–1.1 logit — the sane 0.4–0.9 zone, tilted wide.
7. **Numerics.** Quantiles via `NormalDist().inv_cdf` of the case's `levels`,
   sorted, clamped to [0.01, 99.99], rounded to 4 decimals.

## What we'd do with more time

Model `earned_i/N` directly (dilution as an explicit count process with a
casual-completion curve per kind), estimate refresh-burst size from past jumps
to anticipate post-refresh medians at short horizons, and fit prior constants
from a historical game's actual week-one trajectory — none possible offline here.
