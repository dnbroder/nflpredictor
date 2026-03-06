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
]


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
        keep = [c for c in SCHEDULE_COLS if c in df.columns]
        df = df[keep].copy()
        df = _clean_games(df)
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
    pbp = load_plays(seasons, cache=False)
    logger.info(f"[DATA] PBP loaded: {len(pbp)} plays — aggregating to drives...")
    drives = _aggregate_to_drives(pbp)
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

    Columns: see PBP_COLS above — trimmed from full nflfastR schema
    to manage memory (~300+ columns in raw form).
    """
    seasons = seasons or list(range(2016, CURRENT_SEASON + 1))
    cache_key = f"data/pbp/plays_{min(seasons)}_{max(seasons)}.parquet"

    if cache:
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] Loaded {len(cached)} plays from S3 cache")
            return cached

def load_plays(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load play-by-play data. Large (~1M rows for full history).
    Used as raw input for drive aggregation and L3 PIC computation.
    """
    seasons = seasons or list(range(2016, CURRENT_SEASON + 1))
    cache_key = f"data/pbp/plays_{min(seasons)}_{max(seasons)}.parquet"
    t0 = time.time()

    logger.info(f"[DATA] load_plays: seasons={seasons}, cache={cache}")

    if cache:
        logger.info(f"[DATA] Checking S3 cache: {cache_key}")
        cached = _load_from_s3(cache_key)
        if cached is not None:
            logger.info(f"[DATA] S3 cache HIT: {len(cached)} plays in {time.time()-t0:.1f}s")
            return cached
        logger.info(f"[DATA] S3 cache MISS — downloading PBP from nfl-data-py")

    logger.info(f"[DATA] Loading PBP for {len(seasons)} seasons: {seasons}")
    logger.info(f"[DATA] WARNING: PBP download is large (~500MB/season) — this may take several minutes")
    try:
        import nfl_data_py as nfl
        dfs = []
        for i, season in enumerate(seasons, 1):
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
            dfs.append(df)

        logger.info(f"[DATA] All seasons downloaded — concatenating {len(dfs)} dataframes...")
        result = pd.concat(dfs, ignore_index=True)
        total_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"[DATA] Total plays: {len(result)} rows, {total_mb:.1f}MB in {time.time()-t0:.1f}s total")

        if cache:
            logger.info(f"[DATA] Caching PBP to S3...")
            _save_to_s3(result, cache_key)

        return result

    except ImportError:
        logger.error("[DATA] nfl_data_py not installed")
        raise
    except Exception as e:
        logger.error(f"[DATA] Failed to load plays: {e}", exc_info=True)
        raise


def load_players(
    seasons: List[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load player-level roster + snap count data.
    Used for Layer 3 PIC computation.

    Returns player-week records with:
        player_id, player_name, team, position, position_group,
        season, week, game_id, snap_count, snap_pct,
        age, years_experience
    """
    seasons = seasons or list(range(2016, CURRENT_SEASON + 1))
    cache_key = f"data/players/players_{min(seasons)}_{max(seasons)}.parquet"

    if cache:
        cached = _load_from_s3(cache_key)
        if cached is not None:
            return cached

    logger.info(f"[DATA] Fetching player/roster data for {seasons}...")
    try:
        import nfl_data_py as nfl

        # Rosters
        rosters = nfl.import_weekly_rosters(seasons)

        # Snap counts
        try:
            snaps = nfl.import_snap_counts(seasons)
            players = rosters.merge(snaps, on=["player_id", "game_id", "season", "week"], how="left")
        except Exception:
            logger.warning("[DATA] Snap count import failed — using roster only")
            players = rosters

        # Compute age at time of game
        if "birth_date" in players.columns and "gameday" in players.columns:
            players["birth_date"] = pd.to_datetime(players["birth_date"], errors="coerce")
            players["gameday"] = pd.to_datetime(players["gameday"], errors="coerce")
            players["age"] = (
                (players["gameday"] - players["birth_date"]).dt.days / 365.25
            )

        players = _add_position_groups(players)
        logger.info(f"[DATA] Loaded {len(players)} player-game records")
        if cache:
            _save_to_s3(players, cache_key)
        return players
    except ImportError:
        logger.error("[DATA] nfl_data_py not installed.")
        raise


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
    Load closing total and spread lines from The Odds API.
    Free tier: 500 requests/month — we cache aggressively to S3.

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
            return cached

    if not api_key:
        logger.warning("[DATA] ODDS_API_KEY not set — closing lines unavailable")
        return pd.DataFrame()

    try:
        import requests
        import nfl_data_py as nfl

        # Load schedule to get game_id → date/team mapping
        schedule = nfl.import_schedules(seasons)
        schedule = schedule[schedule["home_score"].notna()].copy()  # completed games only

        all_lines = []
        sport = "americanfootball_nfl"
        regions = "us"
        markets = "totals,spreads"

        for season in seasons:
            season_games = schedule[schedule["season"] == season].copy()
            # Get all historical events for this season
            # The Odds API historical endpoint: /v4/sports/{sport}/odds-history
            # We fetch by date range
            dates = season_games["gameday"].dropna().unique()
            logger.info(f"[DATA] Fetching closing lines for {season}: {len(season_games)} games across {len(dates)} dates")

            for gameday in sorted(dates):
                date_str = pd.Timestamp(gameday).strftime("%Y-%m-%dT23:59:00Z")
                url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds-history/"
                params = {
                    "apiKey": api_key,
                    "regions": regions,
                    "markets": markets,
                    "date": date_str,
                    "oddsFormat": "american",
                }
                try:
                    resp = requests.get(url, params=params, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        for event in data.get("data", []):
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

                            if closing_total is not None:
                                # Match to schedule game_id
                                mask = (
                                    (season_games["gameday"] == gameday) &
                                    (season_games["home_team"].str.contains(home[-3:] if len(home) > 3 else home, case=False, na=False) |
                                     season_games["away_team"].str.contains(away[-3:] if len(away) > 3 else away, case=False, na=False))
                                )
                                matched = season_games[mask]
                                if len(matched) > 0:
                                    row = matched.iloc[0]
                                    all_lines.append({
                                        "game_id": row["game_id"],
                                        "season": season,
                                        "week": row.get("week"),
                                        "home_team": row["home_team"],
                                        "away_team": row["away_team"],
                                        "closing_total": closing_total,
                                        "closing_spread": closing_spread,
                                    })
                    elif resp.status_code == 422:
                        logger.debug(f"[DATA] No odds data for {date_str}")
                    else:
                        logger.warning(f"[DATA] Odds API {resp.status_code} for {date_str}: {resp.text[:100]}")
                except Exception as e:
                    logger.warning(f"[DATA] Odds API request failed for {date_str}: {e}")

        if not all_lines:
            logger.warning("[DATA] No closing lines fetched from Odds API")
            return pd.DataFrame()

        df = pd.DataFrame(all_lines)
        df["closing_total"] = pd.to_numeric(df["closing_total"], errors="coerce")
        df["closing_spread"] = pd.to_numeric(df["closing_spread"], errors="coerce")
        n_valid = df["closing_total"].notna().sum()
        logger.info(f"[DATA] Closing lines fetched: {len(df)} games, {n_valid} with valid totals in {time.time()-t0:.1f}s")

        if cache and len(df) > 0:
            _save_to_s3(df, cache_key)

        return df

    except ImportError as e:
        logger.error(f"[DATA] Missing dependency for closing lines: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"[DATA] load_closing_lines failed: {e}", exc_info=True)
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
        weekly = nfl.import_weekly_data(seasons, ["passing"])
        logger.info(f"[DATA] Weekly data: {len(weekly)} rows")

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
        team_starter = team_starter[team_starter["game_id"].notna()].copy()

        keep_cols = [c for c in [
            "game_id", "season", "week", "team", "player_id",
            "qb_epa_per_play", "qb_completion_pct", "qb_air_yards",
            "qb_epa_5g", "qb_epa_season", "is_backup_qb",
        ] if c in team_starter.columns]

        result = team_starter[keep_cols].copy()
        result = result.rename(columns={"player_id": "qb_id"})

        n_valid = result["qb_epa_5g"].notna().sum() if "qb_epa_5g" in result.columns else 0
        logger.info(f"[DATA] QB stats: {len(result)} team-game rows, {n_valid} with rolling EPA in {time.time()-t0:.1f}s")

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