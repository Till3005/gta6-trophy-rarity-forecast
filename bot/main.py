#!/usr/bin/env python3
"""Forecast how many GTA VI players earn each trophy (make.fun bounty gta6-trophy-rarity-forecast-01).

Reads every *.in case from $MAKEFUN_INPUT_DIR (default /input) in name order and
writes <id>.out to $MAKEFUN_OUTPUT_DIR (default /output). Pure Python 3.13
standard library only.

Model (logit scale):
  z(target) = w * z_trend + (1-w) * z_prior,  w = 1 / (1 + horizon_days / 2)
  z_trend   = z_last + slope * (ln(1+h_target) - ln(1+h_last)), slope fit over last 24h
  z_prior   = per-category week-1 endpoint prior (chapter/story/collect/grind/...),
              lifted so it never drags a trophy more than 0.9 logit below its last read
  spread    = (base + a*sqrt(horizon_days) + b*|z_trend - z_prior|) * widening
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from statistics import NormalDist

NORM = NormalDist()
FALLBACK_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]

RE_CHAPTER = re.compile(r"chapter\s*(\d+)", re.IGNORECASE)
RE_STORY_MISSION = re.compile(r"complete\s+[A-Z*\"'\u2019]")
RE_NUMBER = re.compile(r"\d+")
RE_GRIND = re.compile(r"\b(win|spend|earn|rob|reach|register)\b", re.IGNORECASE)
RE_COLLECT = re.compile(r"\b(all|every|each|discover|explore|collect)\b", re.IGNORECASE)
RE_EASY_ONCE = re.compile(r"\beach\b.*\bonce\b|\bonce\b.*\beach\b", re.IGNORECASE)

PCT = {
    "platinum": 1.6,
    "hundred": 1.8,
    "easy": 32.0,
    "story": 20.0,
    "story_char_final": 16.0,
    "story_final": 10.0,
    "collect": {"bronze": 6.5, "silver": 4.5, "gold": 3.0},
    "grind": {"bronze": 5.0, "silver": 3.5, "gold": 2.5},
    "misc": {"bronze": 12.0, "silver": 7.0, "gold": 4.5},
}
CHAPTER_EARLY = 40.0   # week-1 % for chapter 1 of the story
CHAPTER_LATE = 13.0    # week-1 % for the final chapter

SLOPE_MIN, SLOPE_MAX = -0.5, 1.0
PRIOR_DROP_CAP = 0.9   # prior may drag a trophy at most this far below its last read (logit)
W_BASE, W_HORIZON, W_DISAGREE = 0.14, 0.24, 0.10


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def logit(pct):
    p = min(99.99, max(0.01, pct))
    return math.log(p / (100.0 - p))


def sigmoid(z):
    p = 100.0 / (1.0 + math.exp(-z))
    return min(99.99, max(0.01, p))


def classify(trophy, max_chapter):
    """Map a trophy to a kind and a week-1 endpoint prior (percent)."""
    name = trophy.get("name", "")
    desc = trophy.get("description", "")
    text = f"{name} {desc}"
    low = text.lower()
    tier = trophy.get("tier", "bronze")

    if tier == "platinum" or "platinum" in low:
        return "platinum", PCT["platinum"]
    if "100%" in text or "100 percent" in low:
        return "hundred", PCT["hundred"]

    m = RE_CHAPTER.search(text)
    if m and max_chapter is not None:
        ch = int(m.group(1))
        rel = (ch - 1) / max(1, max_chapter - 1)
        return "chapter", CHAPTER_EARLY + (CHAPTER_LATE - CHAPTER_EARLY) * rel

    low_desc = desc.lower()
    if "final mission" in low_desc:
        if " for " in low_desc:
            return "story_char_final", PCT["story_char_final"]
        return "story_final", PCT["story_final"]

    if RE_STORY_MISSION.search(desc) and not RE_NUMBER.search(desc):
        return "story", PCT["story"]

    if RE_EASY_ONCE.search(desc):
        return "easy", PCT["easy"]

    if RE_COLLECT.search(low_desc):
        return "collect", PCT["collect"].get(tier, 6.5)
    if RE_GRIND.search(low_desc) or RE_NUMBER.search(desc):
        return "grind", PCT["grind"].get(tier, 5.0)
    return "misc", PCT["misc"].get(tier, 12.0)


def dedupe(history):
    """Collapse consecutive identical reads (same N, same rates) into one observation."""
    out = []
    prev = None
    for row in history:
        key = (row.get("players_tracked"), tuple(row.get("rates") or []))
        if prev is not None and key == prev:
            continue
        prev = key
        out.append(row)
    return out


def fit_slope(points, t0, t_ref, window_hours=24.0, half_life_hours=12.0):
    """Recency-weighted OLS slope of z on x = ln(1 + hours since t0)."""
    for wh in (window_hours, 7 * 24.0, None):
        pts = points if wh is None else [p for p in points if (t_ref - p[0]).total_seconds() <= wh * 3600.0]
        if len(pts) >= 2:
            break
    if len(pts) < 2:
        return 0.0
    sw = sx = sy = sxx = sxy = 0.0
    for t, z in pts:
        x = math.log1p(max(0.0, (t - t0).total_seconds()) / 3600.0)
        w = 0.5 ** (((t_ref - t).total_seconds()) / 3600.0 / half_life_hours)
        sw += w
        sx += w * x
        sy += w * z
        sxx += w * x * x
        sxy += w * x * z
    den = sw * sxx - sx * sx
    if abs(den) < 1e-12:
        return 0.0
    return min(SLOPE_MAX, max(SLOPE_MIN, (sw * sxy - sx * sy) / den))


def solve(case):
    trophies = case["trophies"]
    levels = case.get("levels") or FALLBACK_LEVELS
    zscores = [NORM.inv_cdf(q) for q in levels]

    max_ch = max((int(m.group(1)) for m in [RE_CHAPTER.search(f'{t["name"]} {t["description"]}') for t in trophies] if m), default=None)
    meta = [classify(t, max_ch) for t in trophies]

    history = dedupe(case.get("history") or [])
    if not history:
        history = [{"at": case.get("at"), "players_tracked": 0, "rates": [None] * len(trophies)}]
    t0 = parse_ts(history[0]["at"])
    t_last = parse_ts(history[-1]["at"])
    t_target = parse_ts(case["target"])
    horizon_days = max(1e-6, (t_target - t_last).total_seconds() / 86400.0)
    w_blend = 1.0 / (1.0 + horizon_days / 2.0)
    x_step = math.log1p((t_target - t0).total_seconds() / 3600.0) - math.log1p((t_last - t0).total_seconds() / 3600.0)

    rows = []
    for i, tr in enumerate(trophies):
        kind, pct_prior = meta[i]
        series = []
        for row in history:
            rates = row.get("rates") or []
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            series.append((parse_ts(row["at"]), float(r)))
        z_last = logit(series[-1][1]) if series else None
        slope = fit_slope([(t, logit(r)) for t, r in series], t0, t_last) if len(series) >= 2 else 0.0
        z_trend = z_last + slope * x_step if z_last is not None else None

        z_prior_kind = logit(pct_prior)
        if z_last is not None:
            z_prior = max(z_prior_kind, z_last - PRIOR_DROP_CAP)
        else:
            z_prior = z_prior_kind
            z_trend = z_prior_kind
        z_med = w_blend * z_trend + (1.0 - w_blend) * z_prior
        rows.append({
            "id": tr["id"], "kind": kind, "chapter": None, "secret": bool(tr.get("secret")),
            "z_last": z_last, "z_trend": z_trend, "z_prior": z_prior, "z_med": z_med,
        })

    # Monotone story structure: chapter medians must decrease with chapter number,
    # and the true final mission sits at or below the final chapter.
    chap_pairs = []
    for idx, r in enumerate(rows):
        m = RE_CHAPTER.search(f'{trophies[idx]["name"]} {trophies[idx]["description"]}')
        if r["kind"] == "chapter" and m:
            chap_pairs.append((int(m.group(1)), idx))
    chap_pairs.sort()
    prev_z = None
    last_chapter_z = None
    for _, idx in chap_pairs:
        if prev_z is not None:
            rows[idx]["z_med"] = min(rows[idx]["z_med"], prev_z)
        prev_z = rows[idx]["z_med"]
        last_chapter_z = prev_z
    for idx, r in enumerate(rows):
        if r["kind"] == "story_final" and last_chapter_z is not None:
            r["z_med"] = min(r["z_med"], last_chapter_z)
    # Platinum implies 100% completion, so platinum rate <= 100%-completion rate.
    plat = next((r for r in rows if r["kind"] == "platinum"), None)
    hund = next((r for r in rows if r["kind"] == "hundred"), None)
    if plat and hund:
        plat["z_med"] = min(plat["z_med"], hund["z_med"])

    out_rows = []
    for idx, r in enumerate(rows):
        med_pct = sigmoid(r["z_med"])
        disagree = abs(r["z_trend"] - r["z_prior"]) if r["z_trend"] is not None else 1.5
        # Widen only in proportion to how much the prior actually influences the median,
        # so short-horizon bands stay tight while long-horizon disagreement widens them.
        s = W_BASE + W_HORIZON * math.sqrt(horizon_days) + W_DISAGREE * min(disagree, 2.0) * (1.0 - w_blend)
        m_edge = min(med_pct, 100.0 - med_pct)
        s *= 1.0 + 0.35 * math.exp(-m_edge / 2.0)
        if r["secret"]:
            s *= 1.08
        s = max(0.15, min(s, 2.2))
        q = sorted(min(99.99, max(0.01, sigmoid(r["z_med"] + z * s))) for z in zscores)
        out_rows.append({"id": r["id"], "q": [round(v, 4) for v in q]})
    return {"trophies": out_rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = solve(case)
        case_id = case.get("id") or os.path.splitext(name)[0]
        payload = json.dumps(result, separators=(",", ":"))
        with open(os.path.join(out_dir, f"{case_id}.out"), "w", encoding="utf-8") as f:
            f.write(payload)


if __name__ == "__main__":
    main()
