"""
Download NBA line scores from ESPN and build data/nba_games.csv (one row per game).

Source: ESPN's public scoreboard API (no key needed). One request per date returns every
game played that day with its quarter-by-quarter line score and team shooting totals.
Responses are cached as JSON in data/raw_espn/, so the script can be stopped and re-run
without downloading the same dates again. A full download (about 8,800 dates) takes a few hours.

Seasons 1993/94 to 2024/25. ESPN labels a season by the year it ends (2015 = 2014/15).

Dropped here: games not marked as completed, All-Star and exhibition games, malformed
entries, and games whose period scores do not add up to the final score. The restriction
to regular-season, non-neutral-site games happens in analysis.py.
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw_espn"
SEASONS = range(1994, 2026)

SEASON_TYPES = {1: "preseason", 2: "regular", 3: "playoffs", 5: "play-in"}
SHOT_STATS = {"fga": "fieldGoalsAttempted", "tpa": "threePointFieldGoalsAttempted"}
# team abbreviations used for All-Star, Rising Stars and international exhibition games
EXHIBITION_TEAMS = {"EAST", "WEST", "USA", "WORLD", "LEB", "GIA", "STE", "KEN", "CHK", "SHQ"}


def fetch_date(d: date) -> dict:
    """Scoreboard for one date, from the cache if available."""
    cache = RAW_DIR / f"{d:%Y%m%d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(8):
        try:
            r = requests.get(URL, params={"dates": f"{d:%Y%m%d}", "limit": 100}, timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(min(5 * 2 ** attempt, 120))
    cache.write_text(r.text)
    time.sleep(0.3)
    return r.json()


def season_dates(season: int):
    """Every date from 1 October to 30 June (later for the two COVID-shifted seasons)."""
    start, end = date(season - 1, 10, 1), date(season, 6, 30)
    if season == 2020:  # bubble ended in October 2020
        end = date(2020, 10, 15)
    if season == 2021:  # season ran December 2020 to July 2021
        start, end = date(2020, 12, 1), date(2021, 7, 31)
    while start <= end:
        yield start
        start += timedelta(days=1)


def parse_event(ev: dict, d: date) -> dict | None:
    """One ESPN event as a game row seen from the home team, or None if the game is skipped."""
    comp = ev["competitions"][0]
    if not comp["status"]["type"]["completed"]:
        return None
    teams = {t["homeAway"]: t for t in comp["competitors"]}
    if set(teams) != {"home", "away"}:
        return None
    if any("abbreviation" not in t["team"] or not t.get("linescores") for t in teams.values()):
        return None  # placeholder entries without a line score
    if (comp.get("type", {}).get("abbreviation") == "ALLSTAR"
            or {t["team"]["abbreviation"] for t in teams.values()} & EXHIBITION_TEAMS):
        return None

    row = {
        "game_id": ev["id"],
        "date": d.isoformat(),  # US date of the game; ev["date"] is UTC
        "season": ev["season"]["year"],
        "season_type": SEASON_TYPES.get(ev["season"]["type"], ev["season"]["type"]),
        "neutral_site": comp.get("neutralSite", False),
    }
    for side, t in teams.items():
        periods = [float(ls.get("value", ls.get("displayValue", "nan"))) for ls in t["linescores"]]
        row[f"{side}_team"] = t["team"]["abbreviation"]
        row[f"{side}_pts"] = int(t["score"])
        for i in range(4):
            row[f"{side}_q{i + 1}"] = periods[i] if len(periods) > i else float("nan")
        row[f"{side}_ot"] = sum(periods[4:])
        stats = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
        for col, name in SHOT_STATS.items():  # missing for some games
            row[f"{side}_{col}"] = pd.to_numeric(stats.get(name), errors="coerce")
    row["n_periods"] = len(periods)
    return row


def parse_date(data: dict, d: date) -> list[dict]:
    rows = []
    for ev in data.get("events", []):
        try:
            row = parse_event(ev, d)
        except (KeyError, IndexError, TypeError, ValueError):
            continue  # malformed entries, e.g. an empty event on 2007-10-19
        if row is not None:
            rows.append(row)
    return rows


def build_games() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for season in SEASONS:
        n = len(rows)
        for d in season_dates(season):
            rows.extend(parse_date(fetch_date(d), d))
        print(f"{season - 1}/{str(season)[-2:]}: {len(rows) - n} games")
    g = pd.DataFrame(rows).drop_duplicates("game_id")

    # period scores (including overtime) must add up to the final score
    ok = True
    for side in ("home", "away"):
        periods = g[[f"{side}_q1", f"{side}_q2", f"{side}_q3", f"{side}_q4", f"{side}_ot"]]
        ok &= periods.sum(axis=1) == g[f"{side}_pts"]
    g = g[ok].copy()

    g["half_margin_home"] = g["home_q1"] + g["home_q2"] - g["away_q1"] - g["away_q2"]
    g["final_margin_home"] = g["home_pts"] - g["away_pts"]
    g["home_win"] = (g["final_margin_home"] > 0).astype(int)
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


if __name__ == "__main__":
    games = build_games()
    games.to_csv(DATA_DIR / "nba_games.csv", index=False)
    print(f"{len(games)} games saved to {DATA_DIR / 'nba_games.csv'}")
