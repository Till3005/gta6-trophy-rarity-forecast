# entry-rarity — GTA VI week-one trophy rarity forecast

Entry for the make.fun bounty **`gta6-trophy-rarity-forecast-01`** ("Forecast how
many GTA VI players earn each trophy in week one"). GTA VI launched on PS5 on
2026-11-19; the house recorder snapshots Exophase's trophy page every 10–30
minutes (players tracked + each trophy's earn rate, % of tracked players). Given
that history and a future `target` timestamp, the bot forecasts every trophy's
earn rate at the target as five quantiles (q05/q25/q50/q75/q95).

Entered via make.fun referral nkpmb14b.

## Run

Contract entrypoint (stdlib-only Python 3.13, no network):

```sh
python3 -P -s /bot/main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) in name order and
writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`):

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py   # same, via shim
```

Output per case: `{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]},...]}` — one
row per trophy, five finite, non-decreasing 0–100 numbers.

## Model in one paragraph

Everything happens on the logit scale (`z = logit(rate/100)`, rates clamped to
0.01–99.99). Consecutive identical page reads are deduplicated so each distinct
`(time, rate, N)` is one observation. The median blends two estimators:
a recency-weighted linear-in-`ln(1+hours-since-launch)` trend fit over the last
24h of observations, extrapolated to the target, and a per-category week-1
endpoint prior inferred from the trophy's name/description/tier (chapter
trophies 40%→13% by chapter number, story missions ~20%, collectibles 3–10%,
grinds 2–8%, platinum ~1.6%, secrets like their tier peers). Blend weight
`w = 1/(1 + horizon_days/2)` — short horizons trust the data, a 6.5-day horizon
leans on the prior, because the tracked-player flood (638 grinders → tens of
thousands of casuals) dilutes launch-night rates toward casual-completion
levels. The prior can never drag a trophy more than 0.9 logit below its last
observation (trivially-earned trophies stay high). Quantile half-width
` s = 0.14 + 0.24·√horizon_days + 0.10·|trend − prior|·(1−w)`, widened at
extreme rates and for secrets — honest spread rather than the baseline's flat
`0.5·√horizon_days` band. Chapter medians are forced monotone-decreasing, the
final mission sits at/below the last chapter, platinum ≤ 100%-completion.

See `design-notes.md` for the full rationale.

## Repo layout

- `bot/main.py` — the bot (contract entrypoint, stdlib only)
- `main.py` — dev shim delegating to `bot/main.py`
- `input/`, `out/` — local dev cases and outputs (gitignored)
- `design-notes.md` — model rationale
- `.makefun/owner.json` — prize wallet

## License

MIT — see `LICENSE`.
