"""
NFL Data Ingestion
==================
Primary source: nfl-data-py (Python wrapper for nflfastR / nflverse)
Fallback:       Local parquet cache -> S3 parquet cache

Data levels produced:
  games_df   -- one row per game (schedule, scores, lines, weather)
  drives_df  -- one row per drive (team, result, field position, yardage)
  plays_df   -- one row per play  (EPA, WPA, player IDs)
  rosters_df -- one row per player-week (snap counts, status)

All data is stored as parquet in /tmp/nfl_cache/ and optionally mirrored to S3.

COLUMN CONTRACT (drives_df output must match feature_engineering.py expectations):
  - is_miss                  (not is_miss_drive)
  - is_countable_drive       (real offensive possessions only)
  - reached_scoring_position (not reached_scoring_pos)
  - reached_redzone
  - third_down_conv_rate     (not third_down_rate)
  - start_yard_line          (not drive_start_yl)
  - drive_yards_gained       (not drive_yards)
  - is_explosive_drive       (not has_explosive_play)
"""

import logging
import os
import io
import time
import json
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger("nflpredictor")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = Path(os.environ.get("NFL_CACHE_DIR", "/tmp/nfl_cache"))
S3_BUCKET = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
CURRENT_SEASON = int(os.environ.get("NFL_SEASON", 2024))
MIN_TRAINING_SEASON = 2016

DRIVE_SCORING_RESULTS = {"Touchdown", "Field Goal"}
DRIVE_MISS_RESULTS = {"Punt", "Turnover", "Turnover on Downs", "Missed Field Goal",
                      "End of Half", "End of Game", "Fumble", "Interception"}
DRIVE_COUNTABLE_RESULTS = DRIVE_SCORING_RESULTS | DRIVE_MISS_RESULTS


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_games(seasons=None, use_cache=True, use_s3=False):
    seasons = seasons or list(range(MIN_TRAINING_SEASON, CURRENT_SEASON + 1))
    cache_key = f"games_{'_'.join(str(s) for s in sorted(seasons))}.parquet"

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            logger.info(f"[DATA] games: loaded {len(cached)} rows from cache")
            return cached

    if use_s3:
        s3_data = _load_from_s3(f"data/games/{cache_key}")
        if s3_data is not None:
            _save_to_cache(s3_data, cache_key)
            return s3_data

    logger.info(f"[DATA] Loading games for seasons {seasons} via nfl-data-py...")
    df = _load_games_nfl_data_py(seasons)
    df = _enrich_games(df)

    if use_cache:
        _save_to_cache(df, cache_key)
    if use_s3:
        _save_to_s3(df, f"data/games/{cache_key}")

    logger.info(f"[DATA] games: {len(df)} rows loaded ({len(seasons)} seasons)")
    return df


def load_drives(seasons=None, use_cache=True, use_s3=False):
    seasons = seasons or list(range(MIN_TRAINING_SEASON, CURRENT_SEASON + 1))
    cache_key = f"drives_{'_'.join(str(s) for s in sorted(seasons))}.parquet"

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            logger.info(f"[DATA] drives: loaded {len(cached)} rows from cache")
            return cached

    logger.info(f"[DATA] Building drives from PBP for seasons {seasons}...")
    pbp = _load_pbp(seasons)
    df = _build_drives_from_pbp(pbp)
    df = _enrich_drives(df)

    if use_cache:
        _save_to_cache(df, cache_key)
    if use_s3:
        _save_to_s3(df, f"data/drives/{cache_key}")

    logger.info(f"[DATA] drives: {len(df)} rows ({df['is_scoring_drive'].sum()} scoring, "
                f"{df['is_miss'].sum()} misses)")
    return df


def load_plays(seasons=None, cols=None, use_cache=True):
    seasons = seasons or list(range(MIN_TRAINING_SEASON, CURRENT_SEASON + 1))
    default_cols = [
        "game_id", "play_id", "season", "week", "game_date",
        "posteam", "defteam", "side_of_field", "yardline_100",
        "down", "ydstogo", "yards_gained", "play_type",
        "epa", "wpa", "ep", "wp",
        "touchdown", "fumble_lost", "interception",
        "third_down_converted", "third_down_failed",
        "fourth_down_converted", "fourth_down_failed",
        "passer_player_id", "rusher_player_id", "receiver_player_id",
        "passer_player_name", "rusher_player_name", "receiver_player_name",
        "fixed_drive", "fixed_drive_result",
        "home_team", "away_team", "home_score", "away_score",
        "qb_scramble", "pass_attempt", "rush_attempt",
        "complete_pass", "incomplete_pass",
        "penalty", "penalty_yards",
        "field_goal_attempt", "field_goal_result",
        "punt_attempt", "kickoff_attempt",
        "no_play", "special_teams_play",
    ]
    cols = cols or default_cols
    cache_key = f"plays_lean_{'_'.join(str(s) for s in sorted(seasons))}.parquet"

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            logger.info(f"[DATA] plays: loaded {len(cached)} rows from cache")
            return cached

    logger.info(f"[DATA] Loading PBP plays for seasons {seasons}...")
    df = _load_pbp(seasons, cols=cols)

    if use_cache:
        _save_to_cache(df, cache_key)

    logger.info(f"[DATA] plays: {len(df)} rows")
    return df


def load_rosters(seasons=None, use_cache=True):
    seasons = seasons or list(range(MIN_TRAINING_SEASON, CURRENT_SEASON + 1))
    cache_key = f"rosters_{'_'.join(str(s) for s in sorted(seasons))}.parquet"

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            logger.info(f"[DATA] rosters: loaded {len(cached)} rows from cache")
            return cached

    try:
        import nfl_data_py as nfl
        logger.info("[DATA] Loading weekly rosters via nfl-data-py...")
        rosters = nfl.import_weekly_rosters(seasons)
        logger.info("[DATA] Loading snap counts via nfl-data-py...")
        snaps = nfl.import_snap_counts(seasons)
        df = rosters.merge(
            snaps[["player_id", "game_id", "offense_snaps", "offense_pct",
                   "defense_snaps", "defense_pct", "st_snaps", "st_pct"]],
            on=["player_id", "game_id"], how="left"
        )
        if use_cache:
            _save_to_cache(df, cache_key)
        logger.info(f"[DATA] rosters: {len(df)} player-game rows")
        return df
    except ImportError:
        logger.warning("[DATA] nfl-data-py not available for rosters")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"[DATA] roster load failed: {e}")
        return pd.DataFrame()


def load_injuries(season=None, week=None):
    season = season or CURRENT_SEASON
    try:
        import nfl_data_py as nfl
        df = nfl.import_injuries([season])
        if week:
            df = df[df["week"] == week]
        logger.info(f"[DATA] injuries: {len(df)} records for S{season} W{week or 'all'}")
        return df
    except Exception as e:
        logger.warning(f"[DATA] injury load failed: {e}")
        return pd.DataFrame()


def load_contracts(source="local"):
    if source == "s3":
        df = _load_from_s3("data/contracts/nfl_contracts.parquet")
        return df if df is not None else pd.DataFrame()
    local_path = CACHE_DIR / "nfl_contracts.csv"
    if local_path.exists():
        df = pd.read_csv(local_path)
        logger.info(f"[DATA] contracts: {len(df)} rows from {local_path}")
        return df
    logger.warning(f"[DATA] No contract data at {local_path}.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# nfl-data-py wrappers
# ---------------------------------------------------------------------------

def _load_games_nfl_data_py(seasons):
    try:
        import nfl_data_py as nfl
        df = nfl.import_schedules(seasons)
        logger.info(f"[DATA] nfl-data-py: {len(df)} schedule rows")
        return df
    except ImportError:
        raise ImportError("nfl-data-py not installed. Run: pip install nfl-data-py")


def _load_pbp(seasons, cols=None):
    try:
        import nfl_data_py as nfl
        logger.info(f"[DATA] Loading PBP for {len(seasons)} seasons...")
        pbp = nfl.import_pbp_data(seasons, columns=cols, downcast=True)
        logger.info(f"[DATA] PBP loaded: {len(pbp)} rows")
        return pbp
    except ImportError:
        raise ImportError("nfl-data-py required. pip install nfl-data-py")


# ---------------------------------------------------------------------------
# Drive aggregation
# ---------------------------------------------------------------------------

def _build_drives_from_pbp(pbp):
    logger.info("[DATA] Aggregating PBP to drive level...")
    off_plays = pbp[
        (pbp["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike", "no_play"])) &
        (pbp["posteam"].notna()) &
        (pbp["fixed_drive"].notna())
    ].copy()

    if len(off_plays) == 0:
        logger.warning("[DATA] No offensive plays found")
        return pd.DataFrame()

    drive_agg = (
        off_plays.groupby(["game_id", "posteam", "fixed_drive"])
        .agg(
            defteam=("defteam", "first"),
            season=("season", "first"),
            week=("week", "first"),
            game_date=("game_date", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            drive_result=("fixed_drive_result", "last"),
            start_yard_line=("yardline_100", "first"),
            drive_yards_gained=("yards_gained", "sum"),
            drive_plays=("play_id", "count"),
            total_epa=("epa", "sum"),
            mean_epa=("epa", "mean"),
            third_down_attempts=("down", lambda x: (x == 3).sum()),
            third_down_converted=("third_down_converted", "sum"),
            fourth_down_attempts=("down", lambda x: (x == 4).sum()),
            fourth_down_converted=("fourth_down_converted", "sum"),
            had_touchdown=("touchdown", "max"),
            had_fumble_lost=("fumble_lost", "max"),
            had_interception=("interception", "max"),
            min_yardline=("yardline_100", "min"),
        )
        .reset_index()
    )

    explosive = (
        off_plays[off_plays["yards_gained"] >= 20]
        .groupby(["game_id", "posteam", "fixed_drive"])
        .size()
        .reset_index(name="explosive_plays")
    )
    drive_agg = drive_agg.merge(explosive, on=["game_id", "posteam", "fixed_drive"], how="left")
    drive_agg["explosive_plays"] = drive_agg["explosive_plays"].fillna(0)
    drive_agg["is_explosive_drive"] = (drive_agg["explosive_plays"] > 0).astype(int)

    if "time" in off_plays.columns:
        top = (
            off_plays.groupby(["game_id", "posteam", "fixed_drive"])
            .apply(lambda g: _calc_drive_top(g))
            .reset_index(name="drive_top_seconds")
        )
        drive_agg = drive_agg.merge(top, on=["game_id", "posteam", "fixed_drive"], how="left")
    else:
        drive_agg["drive_top_seconds"] = drive_agg["drive_plays"] * 25

    drive_agg = drive_agg.rename(columns={"fixed_drive": "drive_id"})
    drive_agg["drive_id"] = drive_agg["drive_id"].astype(int)
    logger.info(f"[DATA] Drive aggregation complete: {len(drive_agg)} drives from "
                f"{drive_agg['game_id'].nunique()} games")
    return drive_agg


def _calc_drive_top(group):
    try:
        times = pd.to_datetime(group["time"], format="%M:%S", errors="coerce")
        if times.notna().sum() >= 2:
            return abs((times.max() - times.min()).total_seconds())
    except Exception:
        pass
    return group["drive_plays"].iloc[0] * 25


def _enrich_drives(df):
    if len(df) == 0:
        return df

    df["is_scoring_drive"] = df["drive_result"].isin(DRIVE_SCORING_RESULTS).astype(int)
    df["is_miss"] = df["drive_result"].isin(DRIVE_MISS_RESULTS).astype(int)
    df["is_countable_drive"] = df["drive_result"].isin(DRIVE_COUNTABLE_RESULTS).astype(int)
    df["is_turnover"] = df["drive_result"].isin({"Turnover", "Fumble", "Interception"}).astype(int)

    df["drive_end_yl"] = (df["start_yard_line"] - df["drive_yards_gained"]).clip(lower=0)
    df["reached_scoring_position"] = (df["min_yardline"] <= 35).astype(int)
    df["reached_redzone"] = (df["min_yardline"] <= 20).astype(int)
    df["reached_goal_to_go"] = (df["min_yardline"] <= 10).astype(int)

    df["third_down_conv_rate"] = np.where(
        df["third_down_attempts"] > 0,
        df["third_down_converted"] / df["third_down_attempts"],
        np.nan
    )

    df["is_home"] = (df["posteam"] == df["home_team"]).astype(int)
    df = df.sort_values(["game_date", "game_id", "drive_id"]).reset_index(drop=True)
    return df


def _enrich_games(df):
    if len(df) == 0:
        return df

    rename_map = {
        "home_team": "home_team", "away_team": "away_team",
        "home_score": "home_score", "away_score": "away_score",
        "spread_line": "spread_line", "total_line": "total_line",
        "result": "home_margin",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "home_margin" not in df.columns and "home_score" in df.columns:
        df["home_margin"] = df["home_score"] - df["away_score"]
    if "home_score" in df.columns and "away_score" in df.columns:
        df["total_points"] = df["home_score"] + df["away_score"]
    if "wind" in df.columns:
        df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
        df["is_high_wind"] = (df["wind"] >= 20).astype(int)
    if "temp" in df.columns:
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df["is_cold"] = (df["temp"] <= 32).astype(int)
    if "roof" in df.columns:
        df["is_dome"] = df["roof"].isin(["dome", "closed", "retractable"]).astype(int)
    if "div_game" not in df.columns:
        df["div_game"] = 0
    if "gameday" in df.columns:
        df["game_date"] = pd.to_datetime(df["gameday"], errors="coerce")
    elif "game_date" not in df.columns:
        logger.warning("[DATA] No game_date or gameday column found")
    if "game_date" in df.columns:
        df = df.sort_values("game_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _load_from_cache(filename):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"[DATA] Cache read failed for {filename}: {e}")
    return None


def _save_to_cache(df, filename):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    try:
        df.to_parquet(path, index=False)
        logger.debug(f"[DATA] Cached {len(df)} rows to {path}")
    except Exception as e:
        logger.warning(f"[DATA] Cache write failed for {filename}: {e}")


def _load_from_s3(key):
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        logger.debug(f"[DATA] S3 load miss for {key}: {e}")
        return None


def _save_to_s3(df, key):
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        s3 = boto3.client("s3")
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
        logger.info(f"[DATA] S3 saved: s3://{S3_BUCKET}/{key}")
    except Exception as e:
        logger.warning(f"[DATA] S3 save failed for {key}: {e}")


# ---------------------------------------------------------------------------
# CLI bootstrap
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
    parser = argparse.ArgumentParser(description="Pull and cache NFL data")
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=list(range(2018, CURRENT_SEASON + 1)))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--s3", action="store_true", help="Also mirror to S3")
    args = parser.parse_args()
    use_cache = not args.no_cache
    logger.info(f"Bootstrapping data for seasons: {args.seasons}")
    games = load_games(args.seasons, use_cache=use_cache, use_s3=args.s3)
    logger.info(f"Games: {len(games)} rows")
    drives = load_drives(args.seasons, use_cache=use_cache, use_s3=args.s3)
    logger.info(f"Drives: {len(drives)} rows")
    logger.info("Data bootstrap complete.")