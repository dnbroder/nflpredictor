"""
NFL Data Ingestion
Mirrors GolfPredictor historical_data.py

Hierarchy:
  Play-level  → one row per play   (~45k plays/season)
  Drive-level → one row per drive  (~3,000 drives/season)
  Game-level  → one row per game   (~285 games/season regular + playoffs)
  Player-game → player × game      (~22k records/season)
  Contract    → player × contract  (all-time, ~3,000+ records)

Primary source: nfl_data_py (Python wrapper for nflfastR data)
  https://github.com/nflverse/nfl_data_py
  No API key required — data hosted on GitHub releases.
"""

import io
from nfl_predictor.logger import get_logger
import os
import time
from typing import List, Optional

import boto3
import numpy as np
import pandas as pd

logger = get_logger(__name__)

S3_BUCKET = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
CURRENT_SEASON = int(os.environ.get("NFL_SEASON", 2024))

# nfl_data_py column sets — keep only what we need to manage memory
SCHEDULE_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "away_team", "home_team", "away_score", "home_score", "location",
    "result",         # home_score - away_score
    "total",          # home_score + away_score
    "overtime",
    "old_game_id",
    "spread_line",    # opening/consensus spread (home perspective, negative = home favored)
    "total_line",     # opening total
    "closing_spread_line",  # closing spread (Pro Football Reference)
    "closing_total_line",   # closing total (Pro Football Reference) — PRIMARY TARGET SOURCE
    "div_game",
    "roof",           # dome / open / closed / retractable
    "surface",
    "temp",
    "wind",
    "away_qb_id", "home_qb_id",
    "away_qb_name", "home_qb_name",
    "away_coach", "home_coach",
    "stadium_id", "stadium",
]

PBP_COLS = [
    "game_id", "play_id", "season", "week", "game_type",
    "home_team", "away_team", "posteam", "defteam",
    "game_date",
    "qtr", "time", "yrdln", "yardline_100",
    "down", "ydstogo", "yards_gained",
    "play_type",      # run, pass, punt, field_goal, kickoff, no_play, qb_kneel, qb_spike
    "touchdown", "interception", "fumble_lost", "sack",
    "complete_pass",
    "ep",             # expected points before play
    "epa",            # expected points added
    "wp",             # win probability (home team)
    "wpa",            # win probability added
    "vegas_wp",       # market-implied win probability
    "vegas_home_wp",
    "posteam_score", "defteam_score",
    "score_differential",
    "total_home_score", "total_away_score",  # Q1/H1 pace signal (PIM Layer 4)
    "half_seconds_remaining", "game_seconds_remaining",
    "drive",          # drive number
    "fixed_drive",    # corrected drive number
    "fixed_drive_result",  # TD, FG, Opp TD, Punt, Turnover, Turnover on Downs, End of Half, End of Game
    "drive_first_downs", "drive_inside20", "drive_ended_with_score",
    "drive_quarter_start", "drive_quarter_end",
    "drive_yards_penalized",
    "drive_play_count",
    "drive_time_of_possession",
    "drive_first_down_percentage",
    "drive_yards_gained",
    "drive_start_yard_line",
    "drive_end_yard_line",
    "passer_player_id", "passer_player_name",
    "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name",
    "penalty", "penalty_team",
    "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed",
    "field_goal_result",   # made / missed / blocked
    "kick_distance",
    "two_point_attempt", "two_point_conv_result",
    "closing_spread_line",   # closing spread (Pro Football Reference)
    "closing_total_line",    # closing total (Pro Football Reference) — used as target
    "offense_players",       # semicolon-delimited gsis_ids — Layer 3 PIC personnel
    "defense_players",       # defensive personnel — Layer 3 opponent signal
]


# FIX: The Odds API returns full team names; nfl_data_py uses abbreviations.
# This map handles historical name changes (Raiders move, Rams move, etc.)
ODDS_API_TEAM_MAP = {
    "Arizona Cardinals":      "ARI",
    "Atlanta Falcons":        "ATL",
    "Baltimore Ravens":       "BAL",
    "Buffalo Bills":          "BUF",
    "Carolina Panthers":      "CAR",
    "Chicago Bears":          "CHI",
    "Cincinnati Bengals":     "CIN",
    "Cleveland Browns":       "CLE",
    "Dallas Cowboys":         "DAL",
    "Denver Broncos":         "DEN",
    "Detroit Lions":          "DET",
    "Green Bay Packers":      "GB",
    "Houston Texans":         "HOU",
    "Indianapolis Colts":     "IND",
    "Jacksonville Jaguars":   "JAX",
    "Kansas City Chiefs":     "KC",
    "Las Vegas Raiders":      "LV",
    "Oakland Raiders":        "OAK",
    "Los Angeles Chargers":   "LAC",
    "San Diego Chargers":     "SD",
    "Los Angeles Rams":       "LA",
    "St. Louis Rams":         "STL",
    "Miami Dolphins":         "MIA",
    "Minnesota Vikings":      "MIN",
    "New England Patriots":   "NE",
    "New Orleans Saints":     "NO",
    "New York Giants":        "NYG",
    "New York Jets":          "NYJ",
    "Philadelphia Eagles":    "PHI",
    "Pittsburgh Steelers":    "PIT",
    "San Francisco 49ers":    "SF",
    "Seattle Seahawks":       "SEA",
    "Tampa Bay Buccaneers":   "TB",
    "Tennessee Titans":       "TEN",
    "Washington Commanders":  "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins":    "WAS",
}


# ---------------------------------------------------------------------------
# Public API — mirrors GolfPredictor historical_data.py
# ---------------------------------------------------------------------------

def load_games(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    seasons = seasons or list(range(2016, CURRENT_SEASON + 1))
    cache_key = f"data/schedules/games_{min(seasons)}_{max(seasons)}.parquet"
    t0 = time.time()

    logger.info(f"[DATA] load_games: seasons={seasons}, cache={cache}")

    if cache:
        logger.info(f"[DATA] Checking S3 cache: {cache_key}")
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] S3 cache HIT: {len(cached)} games in {time.time()-t0:.1f}s")
            return cached
        logger.info(f"[DATA] S3 cache MISS — fetching from nfl-data-py")

    logger.info(f"[DATA] Loading games for seasons {seasons} via nfl-data-py...")
    try:
        import nfl_data_py as nfl
        t1 = time.time()
        df = nfl.import_schedules(seasons)
        logger.info(f"[DATA] nfl-data-py: {len(df)} schedule rows in {time.time()-t1:.1f}s")

        # Audit which SCHEDULE_COLS are missing — surfaces API changes immediately
        missing_cols = [c for c in SCHEDULE_COLS if c not in df.columns]
        if missing_cols:
            logger.warning(f"[DATA] SCHEDULE_COLS missing from nfl-data-py: {missing_cols}")
        keep = [c for c in SCHEDULE_COLS if c in df.columns]
        df = df[keep].copy()
        df = _clean_games(df)

        # Key column coverage audit — catches NaN issues before they hit feature engineering
        for col in ["total_line", "spread_line", "closing_total", "closing_spread", "home_score", "away_score"]:
            if col in df.columns:
                n_valid = df[col].notna().sum()
                logger.info(f"[DATA]   {col}: {n_valid}/{len(df)} valid")
            else:
                logger.warning(f"[DATA]   {col}: NOT IN DATAFRAME")

        logger.info(f"[DATA] games: {len(df)} rows loaded ({len(seasons)} seasons) in {time.time()-t0:.1f}s total")
        if cache:
            _save_to_s3(df, cache_key)
        return df
    except ImportError:
        logger.error("[DATA] nfl_data_py not installed. Run: pip install nfl_data_py")
        raise
    except Exception as e:
        logger.error(f"[DATA] Failed to load games: {e}", exc_info=True)
        raise


def load_drives(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load drive-level data — PRIMARY INPUT for L1 (P2S) and L2 (P2SZ) cycle computation.
    Aggregated from play-by-play. Each row = one offensive drive.
    """
    seasons = seasons or list(range(2016, CURRENT_SEASON + 1))
    cache_key = f"data/drives/drives_{min(seasons)}_{max(seasons)}.parquet"
    t0 = time.time()

    logger.info(f"[DATA] load_drives: seasons={seasons}, cache={cache}")

    if cache:
        logger.info(f"[DATA] Checking S3 cache: {cache_key}")
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] S3 cache HIT: {len(cached)} drives in {time.time()-t0:.1f}s")
            return cached
        logger.info(f"[DATA] S3 cache MISS — building drives from PBP")

    logger.info(f"[DATA] Building drives from PBP for seasons {seasons}...")
    pbp = load_plays(seasons, cache=True)
    logger.info(f"[DATA] PBP loaded: {len(pbp)} plays — aggregating to drives...")
    t_agg = time.time()
    drives = _aggregate_to_drives(pbp)
    logger.info(f"[DATA] Drive aggregation done in {time.time()-t_agg:.1f}s — {len(drives)} drives")
    logger.info(f"[DATA] Drives complete: {len(drives)} rows in {time.time()-t0:.1f}s total")

    if cache:
        _save_to_s3(drives, cache_key)
    return drives


def load_plays(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load play-by-play data. Large (~1M rows for full history).
    Used as raw input for drive aggregation and L3 PIC computation.
    """
    seasons = seasons or list(range(2016, CURRENT_SEASON + 1))
    # v2: added total_home_score, total_away_score for Q1 pace signal
    cache_key = f"data/pbp/plays_{min(seasons)}_{max(seasons)}_v2.parquet"
    t0 = time.time()

    logger.info(f"[DATA] load_plays: seasons={seasons}, cache={cache}")

    if cache:
        logger.info(f"[DATA] Checking S3 cache: {cache_key}")
        cached = _load_from_s3(cache_key)
        if cached is not None:
            mb = cached.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"[DATA] PBP S3 cache HIT: {len(cached)} plays ({mb:.0f}MB) in {time.time()-t0:.1f}s")
            return cached
        logger.info(f"[DATA] S3 cache MISS — downloading PBP from nfl-data-py")

    logger.info(f"[DATA] Loading PBP for {len(seasons)} seasons: {seasons}")
    logger.info(f"[DATA] Caching per-season to S3 — OOM kills won't lose completed seasons")
    try:
        import nfl_data_py as nfl
        dfs = []
        for i, season in enumerate(seasons, 1):
            # Check per-season S3 cache first — survives OOM kills mid-run
            season_cache_key = f"data/pbp/plays_{season}.parquet"
            season_cached = _load_from_s3(season_cache_key) if cache else None
            if season_cached is not None:
                # Validate cached parquet has required columns — bust cache if stale
                missing_cols = [c for c in ["total_home_score", "total_away_score"]
                                if c not in season_cached.columns]
                if missing_cols:
                    logger.warning(f"[DATA] Season {season} PBP cache STALE — missing {missing_cols}, re-downloading")
                    season_cached = None
                else:
                    mb = season_cached.memory_usage(deep=True).sum() / 1024 / 1024
                    logger.info(f"[DATA] Season {season} PBP cache HIT: {len(season_cached)} plays ({mb:.0f}MB)")
                    dfs.append(season_cached)
                    continue

            t1 = time.time()
            logger.info(f"[DATA] Downloading PBP season {season} ({i}/{len(seasons)})...")
            df = nfl.import_pbp_data([season], downcast=True)
            logger.info(f"[DATA] Season {season} raw: {len(df)} plays, {len(df.columns)} cols — trimming...")
            keep = [c for c in PBP_COLS if c in df.columns]
            df = df[keep].copy()
            df = _clean_plays(df)
            elapsed = time.time() - t1
            mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"[DATA] Season {season} done: {len(df)} plays, {mb:.1f}MB in {elapsed:.1f}s")

            # Cache immediately — completed seasons survive if container OOMs
            if cache:
                _save_to_s3(df, season_cache_key)
                logger.info(f"[DATA] Season {season} cached: {season_cache_key}")

            dfs.append(df)

        logger.info(f"[DATA] All {len(seasons)} seasons ready — concatenating...")
        result = pd.concat(dfs, ignore_index=True)
        total_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"[DATA] Total plays: {len(result)} rows, {total_mb:.1f}MB in {time.time()-t0:.1f}s total")

        if cache:
            logger.info(f"[DATA] Caching combined PBP to S3 ({total_mb:.0f}MB)...")
            _save_to_s3(result, cache_key)
            logger.info(f"[DATA] PBP cached to S3: {cache_key}")

        return result

    except ImportError:
        logger.error("[DATA] nfl_data_py not installed")
        raise
    except Exception as e:
        logger.error(f"[DATA] Failed to load plays: {e}", exc_info=True)
        raise


def load_players(
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load player identity table — gsis_id, position, pfr_id, display_name.
    Used as the Rosetta Stone for joining PBP offense_players to snap counts.
    Static data — rarely changes, aggressive caching appropriate.
    """
    cache_key = "data/players/players_identity.parquet"
    t0 = time.time()

    if cache:
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] Players S3 cache HIT: {len(cached)} players in {time.time()-t0:.1f}s")
            return cached

    try:
        import nfl_data_py as nfl
        logger.info("[DATA] Fetching player identity table...")
        df = nfl.import_players()
        logger.info(f"[DATA] Players: {len(df)} rows, {df['position'].nunique()} positions")

        # Keep only columns needed for L3 join
        keep = [c for c in [
            "gsis_id", "display_name", "position", "pfr_id",
            "first_name", "last_name", "status",
            "draft_year", "draft_round", "draft_pick",
            "years_of_experience", "latest_team",
        ] if c in df.columns]
        df = df[keep].copy()
        df = df[df["gsis_id"].notna()].copy()

        if cache:
            _save_to_s3(df, cache_key)
        return df

    except Exception as e:
        logger.error(f"[DATA] load_players failed: {e}", exc_info=True)
        return pd.DataFrame()


def load_snap_counts(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load snap count data per player per game.
    Used in Layer 3 PIC to weight player cycle contributions by usage.

    A player at 5% snaps is essentially absent from the cycle.
    A player at 100% snaps is the dominant signal for that position group.

    Returns: game_id, pfr_player_id, player, position, team,
             offense_snaps, offense_pct, defense_snaps, defense_pct
    """
    seasons = seasons or list(range(2020, CURRENT_SEASON + 1))
    cache_key = f"data/snaps/snap_counts_{min(seasons)}_{max(seasons)}.parquet"
    t0 = time.time()

    logger.info(f"[DATA] load_snap_counts: seasons={seasons}")

    if cache:
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] Snap counts S3 cache HIT: {len(cached)} rows in {time.time()-t0:.1f}s")
            return cached

    try:
        import nfl_data_py as nfl
        logger.info(f"[DATA] Fetching snap counts for {seasons}...")
        df = nfl.import_snap_counts(seasons)
        logger.info(f"[DATA] Snap counts: {len(df)} player-game rows")

        # Audit coverage
        pct_valid = df["offense_pct"].notna().sum() / len(df) * 100
        logger.info(f"[DATA]   offense_pct valid: {pct_valid:.0f}%")

        if cache:
            _save_to_s3(df, cache_key)
        return df

    except Exception as e:
        logger.error(f"[DATA] load_snap_counts failed: {e}", exc_info=True)
        return pd.DataFrame()


def load_injuries(
    seasons: List[int] = None,
    week: int = None,
) -> pd.DataFrame:
    """
    Load injury report data for lineup adjustment in Layer 3.
    Used to quantify PIC delta before games (players marked Out/Doubtful).

    Returns:
        player_id, player_name, team, season, week,
        report_status (Out/Doubtful/Questionable/Probable),
        practice_status (DNP/Limited/Full)
    """
    seasons = seasons or [CURRENT_SEASON]
    try:
        import nfl_data_py as nfl
        df = nfl.import_injuries(seasons)
        if week is not None:
            df = df[df["week"] == week]
        logger.info(f"[DATA] Loaded {len(df)} injury records")
        return df
    except Exception as e:
        logger.warning(f"[DATA] Injury load failed: {e}")
        return pd.DataFrame()


def load_contracts(source: str = "local") -> pd.DataFrame:
    """
    Load player contract data for Layer 4 PIM salary arbitrage.

    Expected columns:
        player_id, player_name, position, position_group, team,
        contract_year, contract_length, total_value, aav,
        guaranteed, guaranteed_pct, signing_age, inflated_aav

    Sources:
        'local'  — /data/contracts/nfl_contracts.csv (pre-downloaded)
        's3'     — S3 bucket
        'otc'    — OverTheCap API (future: requires scraper)

    Download contracts from:
        https://overthecap.com/contracts
        https://www.spotrac.com/nfl/contracts/
    """
    if source == "s3":
        df = _load_from_s3("data/contracts/nfl_contracts.parquet")
        if df is not None:
            return df
    
    # Try local CSV
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "contracts", "nfl_contracts.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df = _clean_contracts(df)
        logger.info(f"[DATA] Loaded {len(df)} contracts from local file")
        return df
    
    logger.warning("[DATA] No contract data found. Download from Spotrac/OTC and place at data/contracts/nfl_contracts.csv")
    return pd.DataFrame()


def load_closing_lines(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load closing total and spread lines from The Odds API (Historical tier).
    Cached aggressively to S3 — credits only consumed on first fetch.

    FIX: Fetches THREE timestamps per gameday (18:00Z, 21:00Z, 23:59Z) to
    capture all game windows. Single 23:59Z fetch only returned ~340/838 games
    because early Sunday 1pm ET games were already dropped from the API snapshot
    by midnight UTC. Deduplication by event_id prevents double-counting.

    Returns DataFrame with columns:
        game_id, season, week, home_team, away_team,
        closing_total, closing_spread
    """
    seasons = seasons or list(range(2020, CURRENT_SEASON + 1))
    cache_key = f"data/odds/closing_lines_{min(seasons)}_{max(seasons)}.parquet"
    t0 = time.time()

    api_key = os.environ.get("ODDS_API_KEY", "")

    logger.info(f"[DATA] load_closing_lines: seasons={seasons}, cache={cache}")

    if cache:
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] Closing lines S3 cache HIT: {len(cached)} games in {time.time()-t0:.1f}s")
            if "season" in cached.columns:
                for s, grp in cached.groupby("season"):
                    logger.info(f"[DATA]   {s}: {len(grp)} games cached")
            # Supplement with SBR data for seasons not covered by Odds API (≤2019)
            sbr_seasons_needed = [s for s in seasons if s <= 2019]
            if sbr_seasons_needed:
                cached_game_ids = set(cached["game_id"].tolist())
                sbr_df = load_sbr_closing_lines(sbr_seasons_needed)
                if len(sbr_df) > 0:
                    # Only add games not already in the cached Odds API data
                    sbr_new = sbr_df[~sbr_df["game_id"].isin(cached_game_ids)]
                    if len(sbr_new) > 0:
                        cached = pd.concat([cached, sbr_new], ignore_index=True)
                        logger.info(f"[DATA] SBR supplement: added {len(sbr_new)} games for seasons {sbr_seasons_needed}")
            return cached

    if not api_key:
        logger.warning("[DATA] ODDS_API_KEY not set — closing lines unavailable")
        return pd.DataFrame()

    try:
        import requests
        import nfl_data_py as nfl

        # Load schedule to get game_id → date/team mapping
        schedule = nfl.import_schedules(seasons)
        n_total = len(schedule)
        schedule = schedule[schedule["home_score"].notna()].copy()  # completed games only
        logger.info(f"[DATA] Schedule loaded: {len(schedule)} completed games of {n_total} total")

        all_lines = []
        seen_game_ids = set()  # deduplicate across timestamp fetches
        sport = "americanfootball_nfl"
        regions = "us"
        markets = "totals,spreads"
        # DraftKings only — single sharp book cuts credit consumption ~80% vs all-us-books.
        # DK closing lines are liquid and sharp enough for feature construction and target.
        bookmaker = "draftkings"

        # Fetch three timestamps per gameday to capture all game windows:
        #   18:00Z = 1pm ET (early Sunday window)
        #   21:00Z = 4pm ET (late Sunday window)
        #   23:59Z = midnight UTC (primetime/MNF)
        FETCH_HOURS = ["18:00:00Z", "21:00:00Z", "23:59:00Z"]

        for season in seasons:
            season_games = schedule[schedule["season"] == season].copy()
            season_games["gameday"] = pd.to_datetime(season_games["gameday"], errors="coerce")
            dates = season_games["gameday"].dropna().unique()
            season_target = len(season_games)
            season_matched = 0
            season_api_events = 0
            season_no_abbr = 0
            season_no_schedule = 0

            logger.info(f"[DATA] ── Season {season}: {season_target} games across {len(dates)} dates ──")

            for gameday in sorted(dates):
                day_events = {}  # event_id → event dict, deduplicated across hours
                day_str = pd.Timestamp(gameday).strftime("%Y-%m-%d")

                # Fetch all timestamps for this gameday
                for hour in FETCH_HOURS:
                    date_str = pd.Timestamp(gameday).strftime(f"%Y-%m-%dT{hour}")
                    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds-history/"
                    params = {
                        "apiKey": api_key,
                        "regions": regions,
                        "markets": markets,
                        "bookmakers": "draftkings",  # single book = ~80% fewer credits consumed
                        "date": date_str,
                        "oddsFormat": "american",
                    }
                    try:
                        resp = requests.get(url, params=params, timeout=15)
                        if resp.status_code == 200:
                            data = resp.json()
                            events = data.get("data", [])
                            new_events = 0
                            for event in events:
                                event_id = event.get("id", "")
                                if event_id and event_id not in day_events:
                                    day_events[event_id] = event
                                    new_events += 1
                            # Track credits on every successful call — log at INFO
                            # so we always know where we stand without digging
                            credits_remaining = resp.headers.get("x-requests-remaining")
                            credits_used = resp.headers.get("x-requests-used")
                            credits_last = resp.headers.get("x-requests-last")
                            logger.info(
                                f"[DATA]   {date_str}: {len(events)} events, {new_events} new | "
                                f"credits used={credits_used} remaining={credits_remaining} "
                                f"(this call={credits_last})"
                            )
                            # Warn loudly at credit thresholds
                            if credits_remaining is not None:
                                rem = int(credits_remaining)
                                if rem == 0:
                                    logger.error(
                                        f"[DATA] CREDITS EXHAUSTED — 0 remaining. "
                                        f"Upgrade at https://the-odds-api.com"
                                    )
                                    return pd.DataFrame()
                                elif rem < 500:
                                    logger.warning(
                                        f"[DATA] ⚠️  CRITICAL: Only {rem} credits remaining — "
                                        f"fetch may not complete. Consider upgrading plan."
                                    )
                                elif rem < 2000:
                                    logger.warning(
                                        f"[DATA] ⚠️  LOW CREDITS: {rem} remaining — monitor closely"
                                    )
                        elif resp.status_code == 422:
                            logger.debug(f"[DATA]   {date_str}: no odds data (422)")
                        elif resp.status_code == 401:
                            logger.error(f"[DATA] ODDS_API_KEY invalid or expired — aborting")
                            return pd.DataFrame()
                        elif resp.status_code == 429:
                            logger.error(f"[DATA] Odds API rate limit hit — aborting. Credits exhausted.")
                            return pd.DataFrame()
                        else:
                            # Check for out-of-credits in response body
                            try:
                                err = resp.json()
                                if err.get("error_code") == "OUT_OF_USAGE_CREDITS":
                                    logger.error(
                                        f"[DATA] Odds API credits exhausted — aborting. "
                                        f"Upgrade plan at https://the-odds-api.com"
                                    )
                                    return pd.DataFrame()
                            except Exception:
                                pass
                            logger.warning(f"[DATA]   {date_str}: HTTP {resp.status_code} — {resp.text[:200]}")
                    except Exception as e:
                        logger.warning(f"[DATA]   {date_str}: request failed — {e}")

                season_api_events += len(day_events)

                # Log per-day event count — use same ±3 day window for expected count
                gameday_ts = pd.Timestamp(gameday)
                window_games = season_games[
                    (season_games["gameday"] >= gameday_ts - pd.Timedelta(days=3)) &
                    (season_games["gameday"] <= gameday_ts + pd.Timedelta(days=3))
                ]
                expected_games = len(window_games)
                if len(day_events) < expected_games:
                    logger.warning(
                        f"[DATA]   {day_str}: API returned {len(day_events)} events, "
                        f"expected ~{expected_games} — possible coverage gap"
                    )
                else:
                    logger.debug(f"[DATA]   {day_str}: {len(day_events)} events for {expected_games} scheduled games")

                # Process deduplicated events for this gameday
                for event_id, event in day_events.items():
                    home = event.get("home_team", "")
                    away = event.get("away_team", "")
                    closing_total = None
                    closing_spread = None

                    for bm in event.get("bookmakers", []):
                        for market in bm.get("markets", []):
                            if market["key"] == "totals" and closing_total is None:
                                for outcome in market.get("outcomes", []):
                                    if outcome["name"] == "Over":
                                        closing_total = outcome.get("point")
                                        break
                            if market["key"] == "spreads" and closing_spread is None:
                                for outcome in market.get("outcomes", []):
                                    if outcome["name"] == home:
                                        closing_spread = outcome.get("point")
                                        break
                        if closing_total and closing_spread:
                            break

                    if closing_total is None:
                        logger.debug(f"[DATA]   {day_str}: no totals market for '{home}' vs '{away}'")
                        continue

                    # Map full team names → nfl_data_py abbreviations
                    home_abbr = ODDS_API_TEAM_MAP.get(home)
                    away_abbr = ODDS_API_TEAM_MAP.get(away)
                    if not home_abbr or not away_abbr:
                        # WARNING not DEBUG — unmapped names must be visible to fix ODDS_API_TEAM_MAP
                        logger.warning(
                            f"[DATA]   {day_str}: UNMAPPED team name — "
                            f"home='{home}' (mapped={home_abbr}) "
                            f"away='{away}' (mapped={away_abbr}) — add to ODDS_API_TEAM_MAP"
                        )
                        season_no_abbr += 1
                        continue

                    # Match to schedule by team abbreviations within ±3 day window.
                    # The Odds API event dates don't always align exactly with
                    # nfl-data-py schedule dates — the API may return events a few
                    # days before or after the actual gameday. Match on teams only
                    # within a tight window to avoid cross-week false matches.
                    gameday_ts = pd.Timestamp(gameday)
                    date_window = (
                        (season_games["gameday"] >= gameday_ts - pd.Timedelta(days=3)) &
                        (season_games["gameday"] <= gameday_ts + pd.Timedelta(days=3))
                    )
                    mask = (
                        date_window &
                        (season_games["home_team"] == home_abbr) &
                        (season_games["away_team"] == away_abbr)
                    )
                    matched = season_games[mask]

                    if len(matched) == 0:
                        # Try flipped — neutral site or data inconsistency
                        mask_flipped = (
                            date_window &
                            (season_games["home_team"] == away_abbr) &
                            (season_games["away_team"] == home_abbr)
                        )
                        matched = season_games[mask_flipped]
                        if len(matched) > 0:
                            logger.debug(f"[DATA]   {day_str}: matched {home_abbr} vs {away_abbr} via flip (neutral site?)")

                    if len(matched) > 0:
                        row = matched.iloc[0]
                        gid = row["game_id"]
                        if gid not in seen_game_ids:
                            seen_game_ids.add(gid)
                            season_matched += 1
                            all_lines.append({
                                "game_id": gid,
                                "season": season,
                                "week": row.get("week"),
                                "home_team": row["home_team"],
                                "away_team": row["away_team"],
                                "closing_total": closing_total,
                                "closing_spread": closing_spread,
                            })
                    else:
                        # WARNING — schedule miss means a game has odds but no game_id match
                        logger.warning(
                            f"[DATA]   {day_str}: no schedule match for "
                            f"{home_abbr} vs {away_abbr} — check gameday alignment"
                        )
                        season_no_schedule += 1

            # Per-season summary — the key diagnostic that was missing before
            coverage_pct = season_matched / season_target * 100 if season_target > 0 else 0
            logger.info(f"[DATA] ── Season {season} summary ──")
            logger.info(f"[DATA]   Target games:      {season_target}")
            logger.info(f"[DATA]   API events seen:   {season_api_events}")
            logger.info(f"[DATA]   Matched to sched:  {season_matched} ({coverage_pct:.1f}%)")
            logger.info(f"[DATA]   Unmapped teams:    {season_no_abbr}")
            logger.info(f"[DATA]   Schedule misses:   {season_no_schedule}")
            if coverage_pct < 80:
                logger.warning(
                    f"[DATA]   Season {season} coverage {coverage_pct:.1f}% < 80% — "
                    f"check FETCH_HOURS or API subscription tier"
                )

        if not all_lines:
            logger.warning("[DATA] No closing lines fetched from Odds API")
            return pd.DataFrame()

        df = pd.DataFrame(all_lines)
        df["closing_total"] = pd.to_numeric(df["closing_total"], errors="coerce")
        df["closing_spread"] = pd.to_numeric(df["closing_spread"], errors="coerce")
        n_valid = df["closing_total"].notna().sum()
        n_spread_valid = df["closing_spread"].notna().sum()
        total_target = len(schedule)

        logger.info(f"[DATA] ══ CLOSING LINES COMPLETE ══")
        logger.info(f"[DATA]   Total fetched:     {len(df)}/{total_target} games ({len(df)/total_target*100:.1f}%)")
        logger.info(f"[DATA]   Valid totals:      {n_valid}/{len(df)}")
        logger.info(f"[DATA]   Valid spreads:     {n_spread_valid}/{len(df)}")
        logger.info(f"[DATA]   Elapsed:           {time.time()-t0:.1f}s")
        if len(df) < total_target * 0.8:
            logger.warning(
                f"[DATA]   WARNING: Only {len(df)/total_target*100:.1f}% coverage — "
                f"consider adding more FETCH_HOURS or checking API tier"
            )

        if cache and len(df) > 0:
            _save_to_s3(df, cache_key)

        return df

    except ImportError as e:
        logger.error(f"[DATA] Missing dependency for closing lines: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"[DATA] load_closing_lines failed: {e}", exc_info=True)
        return pd.DataFrame()




def load_sbr_closing_lines(seasons, s3_client=None, bucket=None):
    """
    Load SBR historical closing totals for seasons 2016-2019 from S3.
    Returns DataFrame with same schema as load_closing_lines():
        game_id, season, week, home_team, away_team, closing_total, closing_spread
    Only returns rows for games NOT already covered by the Odds API.
    """
    import io as _io
    s3_key = "data/odds/sbr_closing_lines_2016_2019.parquet"
    sbr_seasons = [s for s in seasons if s <= 2019]
    if not sbr_seasons:
        return pd.DataFrame()

    try:
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3", region_name="us-east-1")
        if bucket is None:
            bucket = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")

        obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_parquet(_io.BytesIO(obj["Body"].read()))
        df = df[df["season"].isin(sbr_seasons)].copy()

        if len(df) == 0:
            return pd.DataFrame()

        # Match SBR records to nfl_data_py game_ids via schedule
        import nfl_data_py as nfl
        schedule = nfl.import_schedules(sbr_seasons)
        schedule = schedule[schedule["home_score"].notna()].copy()

        # Build lookup: (season, away_team, home_team) → game_id + week
        sched_lookup = {}
        for _, row in schedule.iterrows():
            key = (int(row["season"]), str(row["away_team"]), str(row["home_team"]))
            sched_lookup[key] = (row["game_id"], row.get("week"))
            # Also try flipped for neutral sites
            key_flip = (int(row["season"]), str(row["home_team"]), str(row["away_team"]))
            if key_flip not in sched_lookup:
                sched_lookup[key_flip] = (row["game_id"], row.get("week"))

        rows = []
        for _, sbr_row in df.iterrows():
            key = (int(sbr_row["season"]), str(sbr_row["away_team"]), str(sbr_row["home_team"]))
            match = sched_lookup.get(key)
            if match:
                game_id, week = match
                rows.append({
                    "game_id":        game_id,
                    "season":         int(sbr_row["season"]),
                    "week":           week,
                    "home_team":      sbr_row["home_team"],
                    "away_team":      sbr_row["away_team"],
                    "closing_total":  float(sbr_row["closing_total"]),
                    "closing_spread": None,
                })

        result = pd.DataFrame(rows)
        logger.info(
            f"[DATA] SBR closing lines: {len(result)}/{len(df)} games matched "
            f"for seasons {sbr_seasons}"
        )
        return result

    except Exception as e:
        logger.warning(f"[DATA] SBR closing lines load failed: {e}")
        return pd.DataFrame()

def load_epa_by_game(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Aggregate EPA from PBP to team-game level.
    Returns rolling offensive and defensive EPA/play per team per game.

    Columns: game_id, season, week, team,
             epa_off_per_play,   # offensive EPA/play this game
             epa_def_per_play,   # defensive EPA/play allowed this game
             epa_off_5g,         # rolling 5-game offensive EPA/play
             epa_def_5g,         # rolling 5-game defensive EPA/play
    """
    seasons = seasons or list(range(2016, CURRENT_SEASON + 1))
    cache_key = f"data/epa/epa_by_game_{min(seasons)}_{max(seasons)}.parquet"
    t0 = time.time()

    logger.info(f"[DATA] load_epa_by_game: seasons={seasons}")

    if cache:
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] EPA S3 cache HIT: {len(cached)} team-game rows in {time.time()-t0:.1f}s")
            return cached

    try:
        pbp = load_plays(seasons, cache=True)

        # Keep only real plays with valid EPA
        plays = pbp[
            pbp["play_type"].isin(["pass", "run"]) &
            pbp["epa"].notna() &
            pbp["posteam"].notna()
        ].copy()

        logger.info(f"[DATA] EPA: {len(plays)} pass/run plays from {len(pbp)} total")

        # Offensive EPA per play per team per game
        off_epa = (
            plays.groupby(["game_id", "season", "week", "posteam"])
            .agg(
                epa_off_per_play=("epa", "mean"),
                epa_off_sum=("epa", "sum"),
                off_plays=("epa", "count"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )

        # Defensive EPA per play allowed per team per game
        def_epa = (
            plays.groupby(["game_id", "season", "week", "defteam"])
            .agg(
                epa_def_per_play=("epa", "mean"),
            )
            .reset_index()
            .rename(columns={"defteam": "team"})
        )

        # Merge off and def
        epa = off_epa.merge(def_epa, on=["game_id", "season", "week", "team"], how="outer")

        # Sort for rolling calcs
        epa = epa.sort_values(["team", "season", "week"]).reset_index(drop=True)

        # Rolling 5-game EPA (lagged — pre-game state)
        epa["epa_off_5g"] = epa.groupby("team")["epa_off_per_play"].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).mean()
        )
        epa["epa_def_5g"] = epa.groupby("team")["epa_def_per_play"].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).mean()
        )
        epa["epa_off_season"] = epa.groupby(["team", "season"])["epa_off_per_play"].transform(
            lambda x: x.shift(1).expanding(min_periods=3).mean()
        )
        epa["epa_def_season"] = epa.groupby(["team", "season"])["epa_def_per_play"].transform(
            lambda x: x.shift(1).expanding(min_periods=3).mean()
        )

        n_valid = epa["epa_off_5g"].notna().sum()
        logger.info(f"[DATA] EPA by game: {len(epa)} team-game rows, {n_valid} with rolling EPA in {time.time()-t0:.1f}s")

        if cache:
            _save_to_s3(epa, cache_key)

        return epa

    except Exception as e:
        logger.error(f"[DATA] load_epa_by_game failed: {e}", exc_info=True)
        return pd.DataFrame()


def load_qb_stats(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load QB efficiency stats per game with rolling averages.
    Used to detect backup starters and QB efficiency signals.

    Columns: game_id, season, week, team,
             qb_epa_per_play,        # QB EPA/dropback this game
             qb_completion_pct,      # completion % this game
             qb_air_yards_per_att,   # air yards per attempt
             qb_epa_5g,              # rolling 5-game QB EPA/play
             is_backup_qb,           # 1 if starter EPA significantly below team avg
             qb_id,                  # starter player ID
    """
    seasons = seasons or list(range(2016, CURRENT_SEASON + 1))
    cache_key = f"data/qb/qb_stats_{min(seasons)}_{max(seasons)}.parquet"
    t0 = time.time()

    logger.info(f"[DATA] load_qb_stats: seasons={seasons}")

    if cache:
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] QB stats S3 cache HIT: {len(cached)} rows in {time.time()-t0:.1f}s")
            return cached

    try:
        import nfl_data_py as nfl

        # Weekly player stats — includes QB passing stats
        logger.info(f"[DATA] Fetching weekly QB stats for {seasons}...")
        weekly = nfl.import_weekly_data(seasons)
        logger.info(f"[DATA] Weekly data: {len(weekly)} rows, {weekly['position'].nunique() if 'position' in weekly.columns else '?'} positions")
        # Filter to QBs only
        if "position" in weekly.columns:
            weekly = weekly[weekly["position"] == "QB"].copy()
            logger.info(f"[DATA] QB rows: {len(weekly)}")

        # Filter to QBs with meaningful dropbacks
        qb_cols = [c for c in [
            "player_id", "player_name", "recent_team", "season", "week",
            "completions", "attempts", "passing_yards", "passing_tds",
            "interceptions", "sacks", "passing_epa", "dakota",
            "air_yards_per_attempt",
        ] if c in weekly.columns]

        qbs = weekly[weekly["attempts"] >= 5][qb_cols].copy() if "attempts" in weekly.columns else weekly[qb_cols].copy()

        # Rename for clarity
        rename = {
            "recent_team": "team",
            "passing_epa": "qb_epa_sum",
            "air_yards_per_attempt": "qb_air_yards",
        }
        qbs = qbs.rename(columns={k: v for k, v in rename.items() if k in qbs.columns})

        # EPA per attempt
        if "qb_epa_sum" in qbs.columns and "attempts" in qbs.columns:
            qbs["qb_epa_per_play"] = qbs["qb_epa_sum"] / qbs["attempts"].replace(0, np.nan)

        # Completion pct
        if "completions" in qbs.columns and "attempts" in qbs.columns:
            qbs["qb_completion_pct"] = qbs["completions"] / qbs["attempts"].replace(0, np.nan)

        # Sort and compute rolling EPA
        qbs = qbs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        if "qb_epa_per_play" in qbs.columns:
            qbs["qb_epa_5g"] = qbs.groupby("player_id")["qb_epa_per_play"].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=2).mean()
            )
            qbs["qb_epa_season"] = qbs.groupby(["player_id", "season"])["qb_epa_per_play"].transform(
                lambda x: x.shift(1).expanding(min_periods=3).mean()
            )

        # Get team starter per game (highest attempts)
        if "attempts" in qbs.columns:
            team_starter = (
                qbs.sort_values("attempts", ascending=False)
                .drop_duplicates(subset=["team", "season", "week"])
            )
        else:
            team_starter = qbs.drop_duplicates(subset=["team", "season", "week"])

        # Flag potential backups: rolling EPA significantly below team season avg
        if "qb_epa_5g" in team_starter.columns and "qb_epa_season" in team_starter.columns:
            team_starter["is_backup_qb"] = (
                (team_starter["qb_epa_5g"] < team_starter["qb_epa_season"] - 0.15)
            ).astype(int)
        else:
            team_starter["is_backup_qb"] = 0

        # Merge with schedule to get game_id
        schedule = nfl.import_schedules(seasons)[["game_id", "season", "week", "home_team", "away_team"]].copy()
        logger.info(f"[DATA] QB schedule merge: {len(team_starter)} starters × {len(schedule)} schedule rows")

        # Join for home team
        home = team_starter.merge(
            schedule.rename(columns={"home_team": "team"})[["game_id", "season", "week", "team"]],
            on=["season", "week", "team"], how="left"
        )
        away = team_starter.merge(
            schedule.rename(columns={"away_team": "team"})[["game_id", "season", "week", "team"]],
            on=["season", "week", "team"], how="left"
        )
        team_starter = pd.concat([home, away]).drop_duplicates(subset=["team", "season", "week"])
        n_before = len(team_starter)
        team_starter = team_starter[team_starter["game_id"].notna()].copy()
        n_dropped = n_before - len(team_starter)
        if n_dropped > 0:
            logger.warning(f"[DATA] QB stats: {n_dropped} rows dropped — no game_id match (team/week not in schedule)")

        keep_cols = [c for c in [
            "game_id", "season", "week", "team", "player_id",
            "qb_epa_per_play", "qb_completion_pct", "qb_air_yards",
            "qb_epa_5g", "qb_epa_season", "is_backup_qb",
        ] if c in team_starter.columns]

        result = team_starter[keep_cols].copy()
        result = result.rename(columns={"player_id": "qb_id"})

        n_valid = result["qb_epa_5g"].notna().sum() if "qb_epa_5g" in result.columns else 0
        n_backup = result["is_backup_qb"].sum() if "is_backup_qb" in result.columns else 0
        logger.info(f"[DATA] QB stats: {len(result)} team-game rows, {n_valid} with rolling EPA, {n_backup} backup flags in {time.time()-t0:.1f}s")

        if cache:
            _save_to_s3(result, cache_key)

        return result

    except Exception as e:
        logger.error(f"[DATA] load_qb_stats failed: {e}", exc_info=True)
        return pd.DataFrame()


def load_lines_historical(source: str = "local") -> pd.DataFrame:
    """
    Load historical closing lines for backtest validation.
    These differ from the opening lines in the schedule data.

    Closing lines are the sharpest signal — they incorporate all sharp money.
    Primary backtest metric: ROI against closing line, not opening line.

    Sources:
        Killer Sports: https://www.killersports.com/  (paid)
        covers.com historical: free but scraping required
        Kaggle dataset: https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data
          → Free, 1966-2022, includes spread and total lines

    Expected columns:
        game_id, season, week, home_team, away_team,
        closing_spread, closing_total,
        actual_result, actual_total
    """
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "lines", "closing_lines.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        logger.info(f"[DATA] Loaded {len(df)} historical closing lines")
        return df
    logger.warning("[DATA] No closing lines file. Download Kaggle dataset to data/lines/closing_lines.csv")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Drive aggregation — core transformation for L1/L2 cycle baseline
# ---------------------------------------------------------------------------

def _aggregate_to_drives(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play-by-play to drive level.

    This is the critical transformation that creates the unit of analysis
    for L1 (P2S scoring) and L2 (P2SZ field position) cycle baseline.

    Drive result categories (from fixed_drive_result):
        'Touchdown'        → scoring (L1 = 1)
        'Field goal'       → scoring (L1 = 1)
        'Punt'             → non-scoring (L1 = 0)
        'Turnover'         → non-scoring (L1 = 0)
        'Turnover on downs'→ non-scoring (L1 = 0)
        'End of half'      → excluded (game-state contaminated)
        'End of game'      → excluded (game-state contaminated)
        'Missed field goal'→ non-scoring (L1 = 0)
        'Blocked field goal' → non-scoring (L1 = 0)
        'Safety'           → non-scoring for offense (L1 = 0)

    Scoring position (L2 targets):
        reached_redzone:           drive ended inside opponent's 20
        reached_scoring_position:  drive ended inside opponent's 35
        reached_opp_territory:     drive crossed midfield
    """
    required = ["game_id", "fixed_drive", "posteam", "defteam", "fixed_drive_result"]
    missing = [c for c in required if c not in pbp.columns]
    if missing:
        raise ValueError(f"PBP missing required columns: {missing}")

    logger.info(f"[DATA] Aggregating {len(pbp)} plays to drives...")
    logger.info(f"[DATA] PBP columns available: {sorted(pbp.columns.tolist())}")

    # Exclude kneeldowns and spikes — not real drives
    pbp_clean = pbp[~pbp["play_type"].isin(["no_play", "qb_kneel", "qb_spike"])].copy()

    # Build agg dict defensively — only include columns that exist
    agg_dict = {
        "season":       ("season", "first"),
        "week":         ("week", "first"),
        "posteam":      ("posteam", "first"),
        "defteam":      ("defteam", "first"),
        "drive_result": ("fixed_drive_result", "last"),
        "drive_play_count": ("play_id", "count"),
        "drive_yards_gained": ("yards_gained", "sum"),
        "epa_sum":      ("epa", "sum"),
        "epa_mean":     ("epa", "mean"),
        "qtr_start":    ("qtr", "first"),
    }

    optional = {
        "game_type":    ("game_type", "first"),
        "game_date":    ("game_date", "first"),
        "drive_inside20": ("drive_inside20", "max"),
        "drive_ended_with_score": ("drive_ended_with_score", "max"),
        "drive_time_of_possession": ("drive_time_of_possession", "first"),
        "drive_first_downs": ("drive_first_downs", "first"),
        "start_yard_line": ("yardline_100", "first"),
        "end_yard_line": ("yardline_100", "last"),
        "third_down_attempts": ("third_down_failed", "sum"),
        "third_down_conversions": ("third_down_converted", "sum"),
        "posteam_score_start": ("posteam_score", "first"),
        "defteam_score_start": ("defteam_score", "first"),
        "game_seconds_remaining_start": ("game_seconds_remaining", "first"),
    }

    for out_col, (src_col, func) in optional.items():
        if src_col in pbp_clean.columns:
            agg_dict[out_col] = (src_col, func)
        else:
            logger.warning(f"[DATA] Optional column '{src_col}' not in PBP — skipping")

    # Per-drive aggregation
    drive_agg = (
        pbp_clean.groupby(["game_id", "fixed_drive"])
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={"fixed_drive": "drive_id"})
    )

    # L1 signal: did this drive result in points?
    scoring_results = {"Touchdown", "Field goal"}
    drive_agg["is_scoring_drive"] = drive_agg["drive_result"].isin(scoring_results).astype(int)

    # L1 miss flag: non-scoring drives that COUNTED (exclude end of half/game)
    excluded_results = {"End of half", "End of game", "Opp touchdown"}
    drive_agg["is_countable_drive"] = ~drive_agg["drive_result"].isin(excluded_results)
    drive_agg["is_miss"] = (
        drive_agg["is_countable_drive"] & ~drive_agg["drive_result"].isin(scoring_results)
    ).astype(int)

    # L2 signals: did the drive reach quality field position?
    # end_yard_line is distance from own goal (100 = opponent's end zone)
    # Inside opponent's 35 = distance from own goal > 65, i.e., yardline_100 < 35
    drive_agg["reached_scoring_position"] = (drive_agg["end_yard_line"] <= 35).astype(int)
    drive_agg["reached_redzone"] = drive_agg["drive_inside20"].fillna(0).astype(int)
    drive_agg["reached_opp_territory"] = (drive_agg["end_yard_line"] <= 50).astype(int)

    # 3rd down conversion rate per drive
    drive_agg["third_down_total"] = (
        drive_agg["third_down_attempts"] + drive_agg["third_down_conversions"]
    )
    drive_agg["third_down_conv_rate"] = np.where(
        drive_agg["third_down_total"] > 0,
        drive_agg["third_down_conversions"] / drive_agg["third_down_total"],
        np.nan,
    )

    # Explosive drive: 20+ yards gained
    drive_agg["is_explosive_drive"] = (drive_agg["drive_yards_gained"] >= 20).astype(int)

    # Score differential at start of drive
    drive_agg["score_diff_start"] = (
        drive_agg["posteam_score_start"] - drive_agg["defteam_score_start"]
    )

    # Remove junk drives
    drive_agg = drive_agg[drive_agg["is_countable_drive"]].copy()
    drive_agg = drive_agg.dropna(subset=["posteam", "drive_result"])

    logger.info(
        f"[DATA] Drive aggregation complete: {len(drive_agg)} drives, "
        f"{drive_agg['is_scoring_drive'].mean():.1%} scoring rate"
    )

    # Log drive result distribution
    dist = drive_agg["drive_result"].value_counts()
    for result, count in dist.items():
        logger.info(f"[DATA]   {result}: {count} ({count/len(drive_agg)*100:.1f}%)")

    return drive_agg


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _clean_games(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize game-level data."""
    # Ensure numeric types
    for col in ["home_score", "away_score", "spread_line", "total_line", "temp", "wind"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute total if missing
    if "total" not in df.columns and "home_score" in df.columns:
        df["total"] = df["home_score"] + df["away_score"]

    # Compute result if missing
    if "result" not in df.columns and "home_score" in df.columns:
        df["result"] = df["home_score"] - df["away_score"]

    # Map closing lines to standard names expected by feature engineering
    if "closing_total_line" in df.columns:
        df["closing_total"] = pd.to_numeric(df["closing_total_line"], errors="coerce")
        n_valid = df["closing_total"].notna().sum()
        logger.info(f"[DATA] closing_total: {n_valid}/{len(df)} valid values")
    else:
        logger.warning("[DATA] closing_total_line not in schedule data — target will be NaN")
        df["closing_total"] = np.nan

    if "closing_spread_line" in df.columns:
        df["closing_spread"] = pd.to_numeric(df["closing_spread_line"], errors="coerce")

    # is_dome
    if "roof" in df.columns:
        df["is_dome"] = df["roof"].isin(["dome", "closed", "retractable"]).astype(int)
    
    # Parse game date
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")

    return df


def _clean_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize play-level data."""
    # Ensure game_date is present
    if "game_date" not in df.columns and "gameday" in df.columns:
        df["game_date"] = df["gameday"]

    for col in ["epa", "ep", "wp", "wpa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure drive indicators are binary
    for col in ["drive_inside20", "drive_ended_with_score", "third_down_converted",
                "third_down_failed", "touchdown", "interception", "fumble_lost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def _clean_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize contract data."""
    for col in ["aav", "total_value", "guaranteed", "contract_length", "signing_age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce"
            )
    if "guaranteed" in df.columns and "total_value" in df.columns:
        df["guaranteed_pct"] = np.where(
            df["total_value"] > 0,
            df["guaranteed"] / df["total_value"],
            np.nan
        )
    return df


def _add_position_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Map position to position group for PIC/PIM analysis."""
    pos_map = {
        # Skill
        "QB": "QB", "WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB",
        # OL
        "LT": "OL", "LG": "OL", "C": "OL", "RG": "OL", "RT": "OL", "OL": "OL",
        # DL
        "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
        # LB
        "ILB": "LB", "OLB": "LB", "MLB": "LB", "LB": "LB",
        # DB
        "CB": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "S": "DB",
        # ST
        "K": "ST", "P": "ST", "LS": "ST",
    }
    if "position" in df.columns:
        df["position_group"] = df["position"].map(pos_map).fillna("OTH")
    return df


# ---------------------------------------------------------------------------
# S3 cache utilities — identical pattern to GolfPredictor
# ---------------------------------------------------------------------------

def _load_from_s3(key: str) -> Optional[pd.DataFrame]:
    """Load parquet from S3 cache. Returns None on miss."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        logger.debug(f"[DATA] S3 cache hit: {key}")
        return df
    except Exception:
        return None


def _save_to_s3(df: pd.DataFrame, key: str):
    """Save DataFrame as parquet to S3 cache."""
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        s3 = boto3.client("s3")
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
        logger.info(f"[DATA] Cached to S3: s3://{S3_BUCKET}/{key} ({len(df)} rows)")
    except Exception as e:
        logger.warning(f"[DATA] S3 cache write failed (non-fatal): {e}")


def _save_local(df: pd.DataFrame, path: str):
    """Save parquet locally."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"[DATA] Saved locally: {path}")