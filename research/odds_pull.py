#!/usr/bin/env python3
"""
odds_pull.py - Timestamped NFL totals/spreads pull from The Odds API (historical).

WHY THIS EXISTS
  The current dataset has no real opening line (nflverse total_line/spread_line are
  CLOSING lines from PFR), so `total_movement` is cross-source noise and
  `spread_movement` is sign-corrupted (~ -2x the spread). This script replaces that
  with a clean, single-source, single-sign-convention, game_id-keyed line table
  carrying OPEN / PICK-TIME / CLOSE totals & spreads, which also enables the
  train/serve-skew and real-CLV work.

DESIGN (two phases, separated so credits are spent exactly once)
  1) FETCH  - download raw historical snapshots to a disk cache. Spends credits.
              Re-runs SKIP already-cached snapshots, so a crash / 429 / resume
              never re-spends. One JSON file per (timestamp) snapshot.
  2) BUILD  - assemble the game-keyed table from cached JSON. FREE + re-runnable,
              so any parsing bug is fixed without touching the API again.

SNAPSHOT TIMING is schedule-driven: each game's true kickoff (from the nflverse
schedule) gives its CLOSE snapshot; weekly anchors give OPEN (Tue) and PICK
(Sat for the Sunday slate, Tue for TNF). The API serves 5-minute snapshots, so
all timestamps are floored to the 5-minute grid.

CREDIT MODEL (historical endpoint): 10 credits per region per market per snapshot.
  markets=totals,spreads with regions=us  ->  20 credits / snapshot.

USAGE
  export ODDS_API_KEY=...
  python odds_pull.py --dry-run                     # estimate credits, fetch nothing
  python odds_pull.py --pilot 2023 --only-week 1    # fetch ONE week, sanity-check
  python odds_pull.py --seasons 2020-2025           # full fetch
  python odds_pull.py --build-only                  # (re)build table from cache, free

Sanity-check the pilot output (open/pick/close should be within ~1-3 pts of each
other; |open-close| > 5 means a bad capture) BEFORE the full burn.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

# ===========================================================================
# CONFIG  -- the genuine decisions live here; change and re-run --dry-run
# ===========================================================================
SPORT          = "americanfootball_nfl"
HOST           = "https://api.the-odds-api.com"
REGIONS        = "us"                      # bills as 1 region, returns ALL us books
MARKETS        = ["totals", "spreads"]      # drop "spreads" to ~halve credits
ODDS_FORMAT    = "american"
SEASONS        = list(range(2020, 2026))    # Odds API historical totals: mid-2020+

ANCHOR_BOOK    = "draftkings"               # the "line a subscriber bets"
USE_CONSENSUS  = True                       # also store median across all us books

# Snapshot anchors (UTC hour on the 5-min grid). 12:00Z ~= 7-8am ET depending on DST.
OPEN_DOW, OPEN_HOUR   = 1, 12               # Tue 12:00Z  (Mon=0 .. Sun=6)
SAT_PICK_DOW, SAT_HOUR = 5, 12              # Sat 12:00Z  -> Sunday-slate pick-time
TNF_PICK_HOUR          = 14                 # Tue 14:00Z  -> TNF pick-time (~10am ET)
CLOSE_LEAD_MIN         = 10                 # close = snapshot floor(kickoff - 10min)

# Paths (override with CLI)
SCHEDULE_PATH  = "data/schedules/games_2016_2024.parquet"   # nflverse schedule you have
CACHE_DIR      = "odds_cache"               # raw snapshot JSON cache (one file / ts)
OUT_PATH       = "data/odds/lines_timestamped_2020_2025.parquet"

CREDITS_PER_SNAPSHOT = 10 * len(MARKETS) * len(REGIONS.split(","))
ET = ZoneInfo("America/New_York")

# Odds API full team name -> nflverse abbreviation (2020+ era; no SD/STL/OAK issues
# except the 2020 Raiders, already "Las Vegas Raiders").
NAME2ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL",
    "Buffalo Bills":"BUF","Carolina Panthers":"CAR","Chicago Bears":"CHI",
    "Cincinnati Bengals":"CIN","Cleveland Browns":"CLE","Dallas Cowboys":"DAL",
    "Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX",
    "Kansas City Chiefs":"KC","Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC",
    "Los Angeles Rams":"LAR","Miami Dolphins":"MIA","Minnesota Vikings":"MIN",
    "New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT",
    "San Francisco 49ers":"SF","Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB",
    "Tennessee Titans":"TEN","Washington Commanders":"WAS",
    "Washington Football Team":"WAS","Washington Redskins":"WAS",
}

# ===========================================================================
# Schedule -> kickoff times -> needed snapshot timestamps
# ===========================================================================
def floor_5min(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_schedule(path: str) -> pd.DataFrame:
    g = pd.read_parquet(path)
    g = g[g["season"].isin(SEASONS)].copy()
    # kickoff: gameday (date) + gametime (ET HH:MM) localized ET -> UTC
    def kickoff(row):
        try:
            d = pd.to_datetime(row["gameday"]).strftime("%Y-%m-%d")
            t = str(row.get("gametime") or "13:00")
            if t in ("nan", "None", ""): t = "13:00"
            naive = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M")
            return naive.replace(tzinfo=ET).astimezone(timezone.utc)
        except Exception:
            return pd.NaT
    g["kickoff_utc"] = g.apply(kickoff, axis=1)
    g = g.dropna(subset=["kickoff_utc"])
    return g[["game_id","season","week","gameday","gametime",
              "home_team","away_team","kickoff_utc","total","away_score","home_score"]]

def week_anchor(kickoff_utc: datetime, target_dow: int, hour: int) -> datetime:
    """The target_dow (Mon=0..Sun=6) at `hour`:00Z in the SAME game-week as kickoff,
    i.e. the most recent such weekday at or before kickoff."""
    k = kickoff_utc.astimezone(timezone.utc)
    back = (k.weekday() - target_dow) % 7
    anchor = (k - timedelta(days=back)).replace(hour=hour, minute=0, second=0, microsecond=0)
    if anchor > k:                      # never anchor after kickoff
        anchor -= timedelta(days=7)
    return floor_5min(anchor)

def plan_snapshots(sched: pd.DataFrame):
    """Return (per_game dict, set of unique snapshot ISO timestamps to fetch)."""
    per_game, needed = {}, set()
    for _, r in sched.iterrows():
        k = r["kickoff_utc"]
        is_thu = k.astimezone(ET).weekday() == 3      # Thursday game -> TNF pick path
        open_ts  = iso_z(week_anchor(k, OPEN_DOW, OPEN_HOUR))
        if is_thu:
            pick_ts = iso_z(week_anchor(k, OPEN_DOW, TNF_PICK_HOUR))
        else:
            pick_ts = iso_z(week_anchor(k, SAT_PICK_DOW, SAT_HOUR))
        close_ts = iso_z(floor_5min(k - timedelta(minutes=CLOSE_LEAD_MIN)))
        per_game[r["game_id"]] = {"open": open_ts, "pick": pick_ts, "close": close_ts}
        needed.update([open_ts, pick_ts, close_ts])
    return per_game, sorted(needed)

# ===========================================================================
# FETCH (spends credits; caches raw JSON; resumable)
# ===========================================================================
def cache_path(ts: str) -> str:
    return os.path.join(CACHE_DIR, ts.replace(":", "").replace("-", "") + ".json")

def fetch_snapshot(ts: str, api_key: str, max_retries: int = 5) -> dict | None:
    """Fetch one historical snapshot; cache to disk. Returns parsed JSON or None."""
    cp = cache_path(ts)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    import urllib.request, urllib.error, urllib.parse
    qs = urllib.parse.urlencode({
        "apiKey": api_key, "regions": REGIONS, "markets": ",".join(MARKETS),
        "oddsFormat": ODDS_FORMAT, "date": ts,
    })
    url = f"{HOST}/v4/historical/sports/{SPORT}/odds?{qs}"
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                used = resp.headers.get("x-requests-used")
                remaining = resp.headers.get("x-requests-remaining")
                payload = json.loads(resp.read().decode())
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cp, "w") as f:
                json.dump(payload, f)
            print(f"  [fetch] {ts}  games={len(payload.get('data', []))}  "
                  f"used={used} remaining={remaining}")
            return payload
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt
                print(f"  [429] rate-limited, backing off {wait}s")
                time.sleep(wait); continue
            print(f"  [HTTP {e.code}] {ts}: {e.read().decode()[:200]}")
            return None
        except Exception as e:
            print(f"  [err] {ts}: {e}")
            time.sleep(2 ** attempt)
    return None

# ===========================================================================
# BUILD (free; from cache)
# ===========================================================================
def _lines_from_event(ev: dict):
    """Return (total_by_book, home_spread_by_book) for one event."""
    totals, spreads = {}, {}
    home = ev.get("home_team")
    for bm in ev.get("bookmakers", []):
        bk = bm["key"]
        for mk in bm.get("markets", []):
            if mk["key"] == "totals":
                pts = [o.get("point") for o in mk["outcomes"] if o.get("point") is not None]
                if pts: totals[bk] = pts[0]                 # over/under share one line
            elif mk["key"] == "spreads":
                for o in mk["outcomes"]:
                    if o.get("name") == home and o.get("point") is not None:
                        spreads[bk] = o["point"]            # home point: neg = home fav
    return totals, spreads

def _index_snapshot(payload: dict) -> dict:
    """Map (home_abbr, away_abbr) -> event dict for one snapshot."""
    idx = {}
    for ev in (payload or {}).get("data", []):
        h = NAME2ABBR.get(ev.get("home_team")); a = NAME2ABBR.get(ev.get("away_team"))
        if h and a:
            idx[(h, a)] = ev
    return idx

def build_table(sched: pd.DataFrame, per_game: dict) -> pd.DataFrame:
    snap_cache = {}
    def get_snap(ts):
        if ts not in snap_cache:
            cp = cache_path(ts)
            snap_cache[ts] = json.load(open(cp)) if os.path.exists(cp) else None
        return _index_snapshot(snap_cache[ts])

    def pick(book_map, key):
        if not book_map: return (None, None, 0)
        anchor = book_map.get(ANCHOR_BOOK)
        consensus = float(pd.Series(list(book_map.values())).median()) if USE_CONSENSUS else None
        return (anchor, consensus, len(book_map))

    rows = []
    for _, r in sched.iterrows():
        gid = r["game_id"]; ts = per_game.get(gid, {})
        rec = {"game_id": gid, "season": r["season"], "week": r["week"],
               "home_team": r["home_team"], "away_team": r["away_team"],
               "kickoff_utc": iso_z(r["kickoff_utc"]),
               "actual_total": r["total"]}
        for phase in ("open", "pick", "close"):
            ev = get_snap(ts.get(phase, "")).get((r["home_team"], r["away_team"]))
            tot, hsp = (_lines_from_event(ev) if ev else ({}, {}))
            a_t, c_t, n = pick(tot, "tot")
            a_s, c_s, _ = pick(hsp, "spr")
            rec[f"{phase}_total"]        = a_t
            rec[f"{phase}_total_consensus"] = c_t
            rec[f"{phase}_spread_home"]  = a_s          # neg = home favored (canonical)
            rec[f"{phase}_nbooks"]       = n
            rec[f"{phase}_ts"]           = ts.get(phase)
        # real movement (open -> close), single source/convention
        if rec["open_total"] is not None and rec["close_total"] is not None:
            rec["total_movement_real"]  = rec["close_total"] - rec["open_total"]
        if rec["open_spread_home"] is not None and rec["close_spread_home"] is not None:
            rec["spread_movement_real"] = rec["close_spread_home"] - rec["open_spread_home"]
        # convenience three-way label vs the TRUE close (push-aware)
        if rec["close_total"] is not None and pd.notna(rec["actual_total"]):
            d = rec["actual_total"] - rec["close_total"]
            rec["result"] = "over" if d > 0 else ("under" if d < 0 else "push")
        # data-quality flags
        rec["flag_no_close"] = rec["close_total"] is None
        rec["flag_big_move"] = (rec.get("total_movement_real") is not None
                                and abs(rec["total_movement_real"]) >= 5)
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df

def report(df: pd.DataFrame):
    n = len(df)
    print(f"\n{'='*64}\nBUILD REPORT  ({n} games)\n{'='*64}")
    print(f"close coverage : {df['close_total'].notna().sum()}/{n} "
          f"({df['close_total'].notna().mean()*100:.1f}%)")
    print(f"pick  coverage : {df['pick_total'].notna().sum()}/{n}")
    print(f"open  coverage : {df['open_total'].notna().sum()}/{n}")
    if "result" in df:
        print(f"label split    : {df['result'].value_counts().to_dict()}")
    mv = df["total_movement_real"].dropna()
    if len(mv):
        print(f"real movement  : mean={mv.mean():+.3f} std={mv.std():.3f} "
              f"|move|>=5 (suspect)={int((mv.abs()>=5).sum())}")
    print(f"flagged no-close: {int(df['flag_no_close'].sum())}  "
          f"big-move: {int(df['flag_big_move'].sum())}")
    print(f"avg books/close: {df['close_nbooks'].replace(0, pd.NA).dropna().mean():.1f}")

# ===========================================================================
# main
# ===========================================================================
def parse_seasons(s: str):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]

def main():
    global SEASONS, SCHEDULE_PATH, CACHE_DIR, OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--seasons", type=str)
    ap.add_argument("--pilot", type=int, help="single season")
    ap.add_argument("--only-week", type=int)
    ap.add_argument("--schedule", type=str, default=SCHEDULE_PATH)
    ap.add_argument("--cache", type=str, default=CACHE_DIR)
    ap.add_argument("--out", type=str, default=OUT_PATH)
    args = ap.parse_args()

    if args.seasons: SEASONS = parse_seasons(args.seasons)
    if args.pilot:   SEASONS = [args.pilot]
    SCHEDULE_PATH, CACHE_DIR, OUT_PATH = args.schedule, args.cache, args.out

    sched = load_schedule(SCHEDULE_PATH)
    if args.only_week is not None:
        sched = sched[sched["week"] == args.only_week]
    print(f"Seasons={SEASONS}  games={len(sched)}")

    per_game, needed = plan_snapshots(sched)
    print(f"Unique snapshots required: {len(needed)}")
    cached = sum(os.path.exists(cache_path(ts)) for ts in needed)
    to_fetch = len(needed) - cached
    print(f"  already cached: {cached}   to fetch: {to_fetch}")
    print(f"  credits/snapshot: {CREDITS_PER_SNAPSHOT}   "
          f"EST CREDIT COST (uncached): {to_fetch * CREDITS_PER_SNAPSHOT}")

    if args.dry_run:
        print("\n[dry-run] no fetch performed. Review the estimate above.")
        return

    if not args.build_only:
        api_key = os.environ.get("ODDS_API_KEY", "")
        if not api_key:
            sys.exit("ODDS_API_KEY not set.")
        for i, ts in enumerate(needed, 1):
            if os.path.exists(cache_path(ts)):
                continue
            print(f"[{i}/{len(needed)}]", end=" ")
            fetch_snapshot(ts, api_key)

    df = build_table(sched, per_game)
    report(df)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}  ({len(df)} rows)")

if __name__ == "__main__":
    main()