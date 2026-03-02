"""
NFL Feature Engineering -- Four-Layer Cyclical Architecture

Layer 1: P2S  -- Possession-to-Score cycle baseline (implemented)
Layer 2: P2SZ -- Possession-to-Scoring Zone cycle baseline (implemented)
Layer 3: PIC  -- Player Impact Curve (Phase 3 stub)
Layer 4: PIM  -- Personal Impact Measurement (Phase 4 stub)

Unit of analysis: game-team row (each game produces two rows).

FIXES vs original:
  - Added TARGET_COLUMNS dict (was missing -- caused ImportError in model.py/backtester.py)
  - Column refs now match nfl_data.py output exactly
  - Added 'target' param to engineer_features()
  - _build_target replaced with _build_targets (builds all target columns)
  - _validate_and_clean now takes target param
"""

import logging
import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nflpredictor")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Targets -- dict mapping market name -> column name in featured DataFrame
# ---------------------------------------------------------------------------
TARGET_COLUMNS = {
    "over_total":       "over_total",
    "spread":           "beat_spread",
    "moneyline":        "win_game",
    "team_total_over":  "team_over",
}

TARGET_COLUMN = "over_total"

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # L1: P2S
    "p2s_baseline", "p2s_ratio", "p2s_miss_streak", "p2s_optimal_zone",
    "p2s_conversion_rate_5g", "p2s_conversion_rate_20g",
    "p2s_cycle_position", "p2s_hot_streak_prob", "p2s_cold_streak_prob",
    # L1 Defense
    "p2s_def_baseline", "p2s_def_ratio", "p2s_def_optimal_zone",
    # L2: P2SZ Red zone
    "p2sz_rz_baseline", "p2sz_rz_ratio", "p2sz_rz_optimal_zone", "p2sz_rz_rate_5g",
    # L2: P2SZ Scoring position
    "p2sz_sp_baseline", "p2sz_sp_ratio", "p2sz_sp_optimal_zone", "p2sz_sp_rate_5g",
    # L2: Drive quality
    "p2sz_yards_per_drive", "p2sz_explosive_rate",
    "p2sz_3rd_conv_baseline", "p2sz_3rd_conv_optimal_zone", "p2sz_avg_start_yard_line",
    # L1 x L2 Intersection
    "p2s_p2sz_dual_optimal", "p2s_p2sz_signal_strength", "matchup_dual_optimal",
    # L3: PIC stubs
    "pic_composite_l1", "pic_composite_l2", "pic_lineup_delta_l1", "pic_lineup_delta_l2",
    "pic_key_player_active", "pic_ol_composite", "pic_pass_rush_opp",
    # L4: PIM stubs
    "pim_trajectory", "pim_salary_signal", "pim_volatility",
    # Contextual
    "is_home", "rest_days", "rest_advantage", "is_dome", "temp", "wind_mph",
    "is_cold_game", "is_windy_game", "is_div_game", "season_week", "is_playoffs",
    "opening_total", "closing_total", "total_movement",
    "opening_spread", "closing_spread", "spread_movement",
    "implied_team_score", "vegas_win_prob",
    "epa_offense_avg", "epa_defense_avg",
    "scoring_avg_5g", "scoring_avg_20g", "scoring_allowed_avg_5g",
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def engineer_features(games_df, drives_df, pbp_df=None, players_df=None,
                      contracts_df=None, closing_lines_df=None, target="over_total"):
    t_start = time.time()
    logger.info("[FEAT] ============================================================")
    logger.info("[FEAT] NFL FEATURE ENGINEERING PIPELINE START")
    logger.info(f"[FEAT]   Games: {len(games_df)}, Drives: {len(drives_df)}, Target: {target}")
    logger.info("[FEAT] ============================================================")

    if games_df.empty or drives_df.empty:
        logger.error("[FEAT] Empty input")
        return pd.DataFrame()

    logger.info("[FEAT] Step 1: Reshaping to game-team rows...")
    df = _reshape_to_game_team(games_df, closing_lines_df)
    logger.info(f"[FEAT]   {len(df)} game-team rows ({df['team'].nunique()} teams)")

    logger.info("[FEAT] Step 2: Building target variables...")
    df = _build_targets(df, target)

    logger.info("[FEAT] Step 3: Building P2S features (Layer 1)...")
    t3 = time.time()
    df = _build_p2s_features(df, drives_df)
    logger.info(f"[FEAT]   Step 3 done in {time.time()-t3:.1f}s")

    logger.info("[FEAT] Step 4: Building P2SZ features (Layer 2)...")
    t4 = time.time()
    df = _build_p2sz_features(df, drives_df)
    logger.info(f"[FEAT]   Step 4 done in {time.time()-t4:.1f}s")

    logger.info("[FEAT] Step 5: Engineering intersection features...")
    df = _build_intersection_features(df)

    for col in [c for c in FEATURE_COLUMNS if c.startswith("pic_")]:
        df[col] = np.nan
    for col in [c for c in FEATURE_COLUMNS if c.startswith("pim_")]:
        df[col] = np.nan

    logger.info("[FEAT] Step 8: Building standard features...")
    df = _build_standard_features(df, games_df)

    logger.info("[FEAT] Step 9: Validation...")
    df = _validate_and_clean(df, target)

    total = time.time() - t_start
    available = [f for f in FEATURE_COLUMNS if f in df.columns and df[f].notna().any()]
    logger.info("[FEAT] ============================================================")
    logger.info(f"[FEAT] PIPELINE COMPLETE: {len(df)} rows, {len(available)}/{len(FEATURE_COLUMNS)} features, {total:.1f}s")
    logger.info("[FEAT] ============================================================")
    return df


# ---------------------------------------------------------------------------
# Step 1: Reshape
# ---------------------------------------------------------------------------
def _reshape_to_game_team(games_df, closing_lines_df=None):
    rows = []
    if closing_lines_df is not None and not closing_lines_df.empty:
        merge_cols = [c for c in ["game_id", "closing_spread", "closing_total"]
                      if c in closing_lines_df.columns]
        if "game_id" in merge_cols:
            games_df = games_df.merge(closing_lines_df[merge_cols], on="game_id", how="left")

    for _, game in games_df.iterrows():
        base = {
            "game_id": game["game_id"],
            "season": game.get("season"),
            "week": game.get("week"),
            "game_type": game.get("game_type", "REG"),
            "gameday": game.get("gameday"),
            "opening_total": pd.to_numeric(game.get("total_line"), errors="coerce"),
            "closing_total": pd.to_numeric(game.get("closing_total"), errors="coerce"),
            "is_dome": int(game.get("is_dome", 0)),
            "temp": float(game.get("temp", 70)) if game.get("is_dome", 0) == 1 else float(game.get("temp", 55)),
            "wind_mph": 0.0 if game.get("is_dome", 0) == 1 else float(game.get("wind", 0)),
            "is_div_game": int(game.get("div_game", 0)),
            "season_week": game.get("week"),
            "is_playoffs": int(game.get("game_type", "REG") != "REG"),
        }
        home_spread = pd.to_numeric(game.get("spread_line"), errors="coerce")
        closing_spread = pd.to_numeric(game.get("closing_spread", game.get("spread_line")), errors="coerce")

        home_row = {**base, "team": game["home_team"], "opponent": game["away_team"],
                    "is_home": 1,
                    "team_score": pd.to_numeric(game.get("home_score"), errors="coerce"),
                    "opponent_score": pd.to_numeric(game.get("away_score"), errors="coerce"),
                    "opening_spread": home_spread, "closing_spread": closing_spread}
        away_row = {**base, "team": game["away_team"], "opponent": game["home_team"],
                    "is_home": 0,
                    "team_score": pd.to_numeric(game.get("away_score"), errors="coerce"),
                    "opponent_score": pd.to_numeric(game.get("home_score"), errors="coerce"),
                    "opening_spread": -home_spread if pd.notna(home_spread) else np.nan,
                    "closing_spread": -closing_spread if pd.notna(closing_spread) else np.nan}
        rows.extend([home_row, away_row])

    df = pd.DataFrame(rows)
    df["spread_movement"] = df["closing_spread"] - df["opening_spread"]
    df["total_movement"] = df["closing_total"] - df["opening_total"]
    df["total_points"] = df["team_score"] + df["opponent_score"]
    df["result"] = df["team_score"] - df["opponent_score"]
    df["implied_team_score"] = (df["closing_total"] - df["closing_spread"]) / 2

    from scipy.stats import norm
    df["vegas_win_prob"] = df["closing_spread"].apply(
        lambda s: norm.cdf(-s / 13.45) if pd.notna(s) else 0.5)
    df["is_cold_game"] = (df["temp"] < 40).astype(int)
    df["is_windy_game"] = (df["wind_mph"] > 15).astype(int)

    if "gameday" in df.columns:
        df = df.sort_values(["gameday", "game_id", "is_home"], ascending=[True, True, False])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Targets
# ---------------------------------------------------------------------------
def _build_targets(df, primary_target="over_total"):
    # over_total
    if "closing_total" in df.columns:
        df["over_total"] = (df["total_points"] > df["closing_total"]).astype(float)
        df.loc[df["closing_total"].isna() | df["total_points"].isna(), "over_total"] = np.nan
    else:
        logger.error("[FEAT] 'closing_total' MISSING -- over_total will be NaN")
        df["over_total"] = np.nan

    # beat_spread
    if "closing_spread" in df.columns:
        df["beat_spread"] = (df["result"] + df["closing_spread"] > 0).astype(float)
        df.loc[df["closing_spread"].isna() | df["result"].isna(), "beat_spread"] = np.nan
    else:
        df["beat_spread"] = np.nan

    # win_game
    df["win_game"] = (df["result"] > 0).astype(float)
    df.loc[df["result"].isna(), "win_game"] = np.nan
    df.loc[df["result"] == 0, "win_game"] = np.nan

    # team_over
    if "implied_team_score" in df.columns:
        df["team_over"] = (df["team_score"] > df["implied_team_score"]).astype(float)
        df.loc[df["implied_team_score"].isna() | df["team_score"].isna(), "team_over"] = np.nan
    else:
        df["team_over"] = np.nan

    target_col = TARGET_COLUMNS.get(primary_target, primary_target)
    if target_col in df.columns:
        n_valid = df[target_col].notna().sum()
        pos_rate = df[target_col].mean()
        logger.info(f"[FEAT]   Primary target '{target_col}': {n_valid}/{len(df)} valid, {pos_rate:.3f} positive rate")
    return df


# ---------------------------------------------------------------------------
# Core cycle baseline engine
# ---------------------------------------------------------------------------
def _compute_cycle_sequence(miss_flags, min_gaps=3):
    n = len(miss_flags)
    baseline_values = np.full(n, np.nan)
    ratio_values = np.full(n, np.nan)
    streak_values = np.full(n, np.nan)
    zone_values = np.full(n, np.nan)
    gaps = []
    current_gap = 0
    current_streak = 0

    for i in range(n):
        v = miss_flags[i]
        if np.isnan(v):
            continue
        if v == 1:
            if current_gap > 0:
                gaps.append(current_gap)
            current_gap = 0
            current_streak += 1
        else:
            current_gap += 1
            current_streak = 0
        if len(gaps) >= min_gaps:
            baseline_values[i] = float(np.mean(gaps))
            if baseline_values[i] > 0:
                ratio_values[i] = current_gap / baseline_values[i]
                zone_values[i] = 1.0 if 0.5 <= ratio_values[i] <= 1.0 else 0.0
        streak_values[i] = float(current_streak)

    return baseline_values, ratio_values, streak_values, zone_values


def _compute_bsr_features(values, min_history=15):
    n = len(values)
    cycle_position = np.full(n, np.nan)
    hot_prob = np.full(n, np.nan)
    cold_prob = np.full(n, np.nan)
    trend = np.full(n, np.nan)

    for i in range(min_history, n):
        history = values[:i]
        valid = history[~np.isnan(history)]
        if len(valid) < min_history:
            continue

        centered = valid - np.mean(valid)
        if np.std(centered) > 0:
            ac = np.correlate(centered, centered, mode="full")
            ac = ac[len(ac) // 2:]
            ac = ac / ac[0] if ac[0] != 0 else ac
            peaks = [j for j in range(2, min(len(ac) - 1, 20))
                     if ac[j] > ac[j-1] and ac[j] > ac[j+1] and ac[j] > 0.15]
            if peaks:
                cycle_len = peaks[0]
                rounds_since = len(valid) % cycle_len
                cycle_position[i] = rounds_since / cycle_len

        recent = valid[-5:] if len(valid) >= 5 else valid
        good = np.sum(recent > np.mean(valid))
        bad = np.sum(recent < np.mean(valid))

        g2g = bad2b = g_total = b_total = 0
        for j in range(1, len(valid)):
            if valid[j-1] > np.mean(valid):
                g_total += 1
                if valid[j] > np.mean(valid):
                    g2g += 1
            elif valid[j-1] < np.mean(valid):
                b_total += 1
                if valid[j] < np.mean(valid):
                    bad2b += 1

        hot_rate = g2g / g_total if g_total > 5 else 0.5
        cold_rate = bad2b / b_total if b_total > 5 else 0.5

        if good >= 3:
            hot_prob[i] = hot_rate
            cold_prob[i] = 1 - hot_rate
        elif bad >= 3:
            hot_prob[i] = 1 - cold_rate
            cold_prob[i] = cold_rate
        else:
            hot_prob[i] = cold_prob[i] = 0.5

        if len(valid) >= 5:
            last5 = valid[-5:]
            slope = np.polyfit(np.arange(5), last5, 1)[0]
            trend[i] = slope

    return {"cycle_position": cycle_position, "hot_prob": hot_prob,
            "cold_prob": cold_prob, "trend": trend}


# ---------------------------------------------------------------------------
# Step 3: Layer 1 -- P2S
# ---------------------------------------------------------------------------
def _build_p2s_features(df, drives_df):
    required = ["game_id", "posteam", "is_miss", "is_scoring_drive", "is_countable_drive"]
    missing = [c for c in required if c not in drives_df.columns]
    if missing:
        logger.warning(f"[FEAT] P2S: drives_df missing {missing} -- filling NaN")
        for col in [c for c in FEATURE_COLUMNS if c.startswith("p2s_")]:
            df[col] = np.nan
        return df

    countable = drives_df[drives_df["is_countable_drive"] == 1].copy()
    sort_cols = [c for c in ["game_date", "game_id", "drive_id"] if c in countable.columns]
    drives_sorted = countable.sort_values(sort_cols).copy()

    team_game_features = {}
    teams_processed = 0

    for team, team_drives in drives_sorted.groupby("posteam"):
        miss_flags = team_drives["is_miss"].values.astype(float)
        score_flags = team_drives["is_scoring_drive"].values.astype(float)
        game_ids = team_drives["game_id"].values

        baseline_vals, ratio_vals, streak_vals, zone_vals = _compute_cycle_sequence(miss_flags)
        bsr = _compute_bsr_features(score_flags)

        scoring_series = pd.Series(score_flags)
        conv_5g_raw = scoring_series.rolling(window=40, min_periods=10).mean()
        conv_20g_raw = scoring_series.rolling(window=160, min_periods=40).mean()

        game_indices = {}
        for idx, gid in enumerate(game_ids):
            if gid not in game_indices:
                game_indices[gid] = []
            game_indices[gid].append(idx)

        for gid, idxs in game_indices.items():
            last_idx = idxs[-1]
            team_game_features[(team, gid)] = {
                "p2s_baseline": baseline_vals[last_idx],
                "p2s_ratio": ratio_vals[last_idx],
                "p2s_miss_streak": streak_vals[last_idx],
                "p2s_optimal_zone": zone_vals[last_idx],
                "p2s_conversion_rate_5g": conv_5g_raw.iloc[last_idx],
                "p2s_conversion_rate_20g": conv_20g_raw.iloc[last_idx],
                "p2s_cycle_position": bsr["cycle_position"][last_idx],
                "p2s_hot_streak_prob": bsr["hot_prob"][last_idx],
                "p2s_cold_streak_prob": bsr["cold_prob"][last_idx],
            }
        teams_processed += 1
        if teams_processed % 8 == 0:
            logger.info(f"[FEAT]   P2S: processed {teams_processed} teams...")

    logger.info(f"[FEAT]   P2S: computed {len(team_game_features)} team-game records")
    df = _merge_lagged_features(df, team_game_features, "p2s_", drives_sorted)

    # Defensive P2S
    game_order = (
        drives_sorted.drop_duplicates(subset=["game_id", "posteam"])
        .sort_values([c for c in ["game_date", "game_id"] if c in drives_sorted.columns])
        [["game_id", "posteam"]].rename(columns={"posteam": "team"})
    )
    game_order["prev_game_id"] = game_order.groupby("team")["game_id"].shift(1)
    prev_game_map = game_order.set_index(["team", "game_id"])["prev_game_id"].to_dict()

    def _get_def_feature(row, feat_key):
        prev_gid = prev_game_map.get((row["opponent"], row["game_id"]))
        if prev_gid is None:
            return np.nan
        return team_game_features.get((row["opponent"], prev_gid), {}).get(feat_key, np.nan)

    logger.info("[FEAT]   Computing defensive P2S features...")
    df["p2s_def_baseline"] = df.apply(lambda r: _get_def_feature(r, "p2s_baseline"), axis=1)
    df["p2s_def_ratio"] = df.apply(lambda r: _get_def_feature(r, "p2s_ratio"), axis=1)
    df["p2s_def_optimal_zone"] = df.apply(lambda r: _get_def_feature(r, "p2s_optimal_zone"), axis=1)

    def_valid = df["p2s_def_baseline"].notna().sum()
    logger.info(f"[FEAT]   Defensive P2S: {def_valid}/{len(df)} rows have data")
    return df


# ---------------------------------------------------------------------------
# Step 4: Layer 2 -- P2SZ
# ---------------------------------------------------------------------------
def _build_p2sz_features(df, drives_df):
    required_cols = ["game_id", "posteam", "reached_redzone", "reached_scoring_position",
                     "is_countable_drive"]
    missing = [c for c in required_cols if c not in drives_df.columns]
    if missing:
        logger.warning(f"[FEAT] P2SZ: drives_df missing {missing} -- filling NaN")
        for col in [c for c in FEATURE_COLUMNS if c.startswith("p2sz_")]:
            df[col] = np.nan
        return df

    countable = drives_df[drives_df["is_countable_drive"] == 1].copy()
    sort_cols = [c for c in ["game_date", "game_id", "drive_id"] if c in countable.columns]
    drives_sorted = countable.sort_values(sort_cols).copy()

    team_game_features = {}
    teams_processed = 0

    for team, team_drives in drives_sorted.groupby("posteam"):
        rz_miss = (1 - team_drives["reached_redzone"].values).astype(float)
        rz_baseline, rz_ratio, _, rz_zone = _compute_cycle_sequence(rz_miss)

        sp_miss = (1 - team_drives["reached_scoring_position"].values).astype(float)
        sp_baseline, sp_ratio, _, sp_zone = _compute_cycle_sequence(sp_miss)

        n_drives = len(team_drives)
        if "third_down_conv_rate" in team_drives.columns:
            td_conv = team_drives["third_down_conv_rate"].values.astype(float)
            td_avg = np.nanmean(td_conv)
            td_miss = np.where(np.isnan(td_conv), np.nan, (td_conv < td_avg).astype(float))
            td_baseline, _, _, td_zone = _compute_cycle_sequence(td_miss)
        else:
            td_baseline = td_zone = np.full(n_drives, np.nan)

        rz_series = pd.Series(team_drives["reached_redzone"].values.astype(float))
        sp_series = pd.Series(team_drives["reached_scoring_position"].values.astype(float))
        rz_5g = rz_series.rolling(window=40, min_periods=10).mean()
        sp_5g = sp_series.rolling(window=40, min_periods=10).mean()

        if "drive_yards_gained" in team_drives.columns:
            yds = pd.Series(team_drives["drive_yards_gained"].values.astype(float))
            yds_per_drive = yds.rolling(window=80, min_periods=16).mean()
        else:
            yds_per_drive = pd.Series(np.nan, index=range(n_drives))

        if "is_explosive_drive" in team_drives.columns:
            exp_rate = pd.Series(team_drives["is_explosive_drive"].values.astype(float))
            explosive_rate = exp_rate.rolling(window=80, min_periods=16).mean()
        else:
            explosive_rate = pd.Series(0.0, index=range(n_drives))

        if "start_yard_line" in team_drives.columns:
            syl = pd.Series(team_drives["start_yard_line"].values.astype(float))
            avg_start = syl.rolling(window=80, min_periods=8).mean()
        else:
            avg_start = pd.Series(np.nan, index=range(n_drives))

        game_ids = team_drives["game_id"].values
        game_indices = {}
        for idx, gid in enumerate(game_ids):
            if gid not in game_indices:
                game_indices[gid] = []
            game_indices[gid].append(idx)

        for gid, idxs in game_indices.items():
            li = idxs[-1]
            team_game_features[(team, gid)] = {
                "p2sz_rz_baseline": rz_baseline[li],
                "p2sz_rz_ratio": rz_ratio[li],
                "p2sz_rz_optimal_zone": rz_zone[li],
                "p2sz_rz_rate_5g": rz_5g.iloc[li],
                "p2sz_sp_baseline": sp_baseline[li],
                "p2sz_sp_ratio": sp_ratio[li],
                "p2sz_sp_optimal_zone": sp_zone[li],
                "p2sz_sp_rate_5g": sp_5g.iloc[li],
                "p2sz_3rd_conv_baseline": td_baseline[li] if not np.isnan(td_baseline[li]) else np.nan,
                "p2sz_3rd_conv_optimal_zone": td_zone[li] if not np.isnan(td_zone[li]) else np.nan,
                "p2sz_yards_per_drive": yds_per_drive.iloc[li],
                "p2sz_explosive_rate": explosive_rate.iloc[li],
                "p2sz_avg_start_yard_line": avg_start.iloc[li],
            }
        teams_processed += 1
        if teams_processed % 8 == 0:
            logger.info(f"[FEAT]   P2SZ: processed {teams_processed} teams...")

    logger.info(f"[FEAT]   P2SZ: computed {len(team_game_features)} team-game records")
    df = _merge_lagged_features(df, team_game_features, "p2sz_", drives_sorted)
    return df


# ---------------------------------------------------------------------------
# Step 5: Intersection features
# ---------------------------------------------------------------------------
def _build_intersection_features(df):
    has_l1 = "p2s_optimal_zone" in df.columns
    has_l2 = "p2sz_sp_optimal_zone" in df.columns

    if has_l1 and has_l2:
        df["p2s_p2sz_dual_optimal"] = (
            (df["p2s_optimal_zone"] == 1.0) & (df["p2sz_sp_optimal_zone"] == 1.0)
        ).astype(float)
        df.loc[df["p2s_optimal_zone"].isna() | df["p2sz_sp_optimal_zone"].isna(),
               "p2s_p2sz_dual_optimal"] = np.nan

        if "p2s_ratio" in df.columns and "p2sz_sp_ratio" in df.columns:
            r1 = df["p2s_ratio"].clip(0, 2).fillna(0)
            r2 = df["p2sz_sp_ratio"].clip(0, 2).fillna(0)
            df["p2s_p2sz_signal_strength"] = (r1 + r2) / 2
        else:
            df["p2s_p2sz_signal_strength"] = np.nan

        if "p2s_def_ratio" in df.columns:
            df["matchup_dual_optimal"] = (
                (df["p2s_p2sz_dual_optimal"] == 1.0) & (df["p2s_def_ratio"] > 1.0)
            ).astype(float)
        else:
            df["matchup_dual_optimal"] = np.nan
    else:
        df["p2s_p2sz_dual_optimal"] = np.nan
        df["p2s_p2sz_signal_strength"] = np.nan
        df["matchup_dual_optimal"] = np.nan

    if "p2s_p2sz_dual_optimal" in df.columns:
        n_dual = (df["p2s_p2sz_dual_optimal"] == 1.0).sum()
        n_valid = df["p2s_p2sz_dual_optimal"].notna().sum()
        if n_valid > 0:
            logger.info(f"[FEAT]   Dual optimal: {n_dual}/{n_valid} ({n_dual/n_valid*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 8: Standard features
# ---------------------------------------------------------------------------
def _build_standard_features(df, games_df):
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"])
        df = df.sort_values(["team", "gameday"])
        df["prev_gameday"] = df.groupby("team")["gameday"].shift(1)
        df["rest_days"] = (df["gameday"] - df["prev_gameday"]).dt.days
        df["rest_days"] = df["rest_days"].fillna(7).clip(4, 21)

        rest_map = df.set_index(["game_id", "team"])["rest_days"].to_dict()
        df["opponent_rest"] = df.apply(
            lambda r: rest_map.get((r["game_id"], r["opponent"]), np.nan), axis=1)
        df["rest_advantage"] = df["rest_days"] - df["opponent_rest"]

    if "epa_offense_avg" not in df.columns:
        df["epa_offense_avg"] = np.nan
        df["epa_defense_avg"] = np.nan

    if "team_score" in df.columns:
        df["scoring_avg_5g"] = df.groupby("team")["team_score"].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).mean())
        df["scoring_avg_20g"] = df.groupby("team")["team_score"].transform(
            lambda x: x.shift(1).rolling(window=20, min_periods=5).mean())
    if "opponent_score" in df.columns:
        df["scoring_allowed_avg_5g"] = df.groupby("team")["opponent_score"].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).mean())
    return df


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------
def _merge_lagged_features(df, team_game_features, prefix, drives_df):
    if not team_game_features:
        return df

    game_order = (
        drives_df.drop_duplicates(subset=["game_id", "posteam"])
        .sort_values([c for c in ["game_date", "game_id"] if c in drives_df.columns])
        [["game_id", "posteam"]].rename(columns={"posteam": "team"})
    )
    game_order["prev_game_id"] = game_order.groupby("team")["game_id"].shift(1)
    prev_game_map = game_order.set_index(["team", "game_id"])["prev_game_id"].to_dict()

    feature_cols = list(next(iter(team_game_features.values())).keys())

    for col in feature_cols:
        if col.startswith(prefix.rstrip("_")) or col.startswith(prefix):
            col_data = []
            for _, row in df.iterrows():
                prev_gid = prev_game_map.get((row["team"], row["game_id"]))
                feats = team_game_features.get((row["team"], prev_gid), {}) if prev_gid else {}
                col_data.append(feats.get(col, np.nan))
            df[col] = col_data

    sample = [c for c in feature_cols if c.startswith(prefix.rstrip("_")) or c.startswith(prefix)]
    if sample and sample[0] in df.columns:
        valid = df[sample[0]].notna().sum()
        logger.info(f"[FEAT]   {prefix}* merged: {valid}/{len(df)} rows have data")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_and_clean(df, target="over_total"):
    initial = len(df)
    target_col = TARGET_COLUMNS.get(target, target)

    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    after_target = len(df)
    logger.info(f"[FEAT]   Dropped {initial - after_target} rows with NaN target ({target_col})")

    available = [f for f in FEATURE_COLUMNS if f in df.columns]
    logger.info(f"[FEAT]   Feature coverage: {len(available)}/{len(FEATURE_COLUMNS)}")
    for col in available:
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100
        if n_valid == 0:
            logger.warning(f"[FEAT]     {col}: 0 valid values")
        elif pct < 50:
            logger.warning(f"[FEAT]     {col}: {pct:.0f}% valid ({n_valid}/{len(df)})")

    if "p2s_p2sz_dual_optimal" in df.columns and target_col in df.columns:
        dual = df[df["p2s_p2sz_dual_optimal"] == 1.0]
        if len(dual) > 10:
            rate_all = df[target_col].mean()
            rate_dual = dual[target_col].mean()
            logger.info(f"[FEAT]   INTERSECTION: overall={rate_all:.3f}, dual={rate_dual:.3f}, "
                        f"lift={rate_dual - rate_all:+.3f}")
    return df