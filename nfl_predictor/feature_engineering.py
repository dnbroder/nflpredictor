"""
NFL Feature Engineering — Four-Layer Cyclical cycle baseline Architecture
Mirrors GolfPredictor feature_engineering.py exactly in structure.

Layer 1: P2S  — Possession-to-Score cycle baseline (L1 implemented)
Layer 2: P2SZ — Possession-to-Scoring Zone cycle baseline (L2 implemented)
Layer 3: PIC  — Player Impact Curve (Phase 3)
Layer 4: PIM  — Personal Impact Measurement (Phase 4)

Unit of analysis: game-team row (each game produces two rows — one per team).
This mirrors GolfPredictor's per-player-round row structure.
"""

from nfl_predictor.logger import get_logger
import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
# ── Live betting features (computed from in-game PBP, NOT used in pre-game model) ──
# These features are valid ONLY after Q1/H1 has been played.
# They encode the Q1/H1 pace signal for live in-game betting.
# Data leakage risk: using these in pre-game training inflates AUC to ~0.80
# because Q1 scoring is partially determined by the same factors as final score.
# Architecture: pre-game model fires first, live model refines after Q1 ends.
LIVE_FEATURE_COLUMNS = [
    "pim_q1_pct_of_total",          # q1_combined / closing_total — primary live signal
    "pim_q1_epa_per_drive",         # avg EPA per Q1 drive (efficiency without scoring)
    "pim_q1_scoring_drives",        # number of scoring drives in Q1
    "pim_q1_cycle_confirmed",       # 1 if pre-game dual_optimal AND q1_pct >= 0.20
    "pim_q1_cycle_contradicted",    # 1 if pre-game dual_optimal AND q1_pct <= 0.10
    "pim_h1_pct_of_total",          # h1_combined / closing_total — halftime live signal
]

# Validated live signal thresholds (887 games, 2016-2019):
#   pim_q1_pct_of_total <= 0.10  → 25% over rate (strong under lean)
#   pim_q1_pct_of_total >= 0.25  → 69-82% over rate (strong over lean)
#   pim_h1_pct_of_total < 0.30   → 11% over rate (near-certain under)
#   pim_h1_pct_of_total >= 0.80  → 100% over rate (56/56 games)

TARGET_COLUMN = "over_total"  # game goes over the total line
TARGET_COLUMNS = {"over_total": "over_total"}  # dict for model.py compatibility

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [

    # ── Layer 1: P2S — Possession-to-Score cycle baseline ─────────────────────────
    # Offense
    "p2s_baseline",                     # avg drives between scoring drives (baseline)
    "p2s_ratio",               # current non-scoring streak / p2s_baseline
    "p2s_miss_streak",              # consecutive non-scoring drives
    "p2s_optimal_zone",             # 1 if ratio in [0.5, 1.0] — reversion zone
    "p2s_conversion_rate_5g",       # rolling 5-game scoring drive rate
    "p2s_conversion_rate_20g",      # rolling 20-game scoring drive rate (baseline)
    "p2s_cycle_position",           # autocorrelation cycle position (0.0–1.0)
    "p2s_hot_streak_prob",          # prob current hot streak continues
    "p2s_cold_streak_prob",         # prob current cold streak continues

    # Defense (opponent's L1 state — fade/confirm signal)
    "p2s_def_baseline",                 # opponent's avg drives between scoring drives
    "p2s_def_ratio",           # opponent's current ratio
    "p2s_def_optimal_zone",         # 1 if opponent offense is in reversion zone

    # ── Layer 2: P2SZ — Possession-to-Scoring Zone cycle baseline ────────────────
    # Red zone
    "p2sz_rz_baseline",                 # avg drives between red zone entries
    "p2sz_rz_ratio",           # current non-RZ streak / rz_baseline
    "p2sz_rz_optimal_zone",         # 1 if red zone entry rate in reversion zone
    "p2sz_rz_rate_5g",              # rolling 5-game red zone entry rate

    # Scoring position (inside opp 35)
    "p2sz_sp_baseline",                 # avg drives between scoring position entries
    "p2sz_sp_ratio",           # current non-SP streak / sp_baseline
    "p2sz_sp_optimal_zone",         # 1 if scoring position rate in reversion zone
    "p2sz_sp_rate_5g",              # rolling 5-game scoring position rate

    # Drive quality
    "p2sz_yards_per_drive",         # rolling 10-game yards per drive
    "p2sz_explosive_rate",          # rolling 10-game explosive drive rate (20+ yards)
    "p2sz_3rd_conv_baseline",           # Cycle baseline on 3rd down conversion rate
    "p2sz_3rd_conv_optimal_zone",   # 1 if 3rd down rate in reversion zone
    "p2sz_avg_start_yard_line",     # avg drive start position (opponent field position)

    # ── L1 × L2 Intersection — PRIMARY SIGNAL ───────────────────────────
    "p2s_p2sz_dual_optimal",        # 1 if BOTH L1 AND L2 in optimal zone
    "p2s_p2sz_signal_strength",     # sum of L1 ratio + L2 ratio (continuous)
    "matchup_dual_optimal",         # 1 if this team dual optimal AND opp defense suppressed

    # ── Layer 3: PIC — Player Impact Curve ──────────────────────────────
    "pic_qb_in_zone",               # QB personally in L1 optimal zone pre-game
    "pic_qb_career_rate",           # QB career scoring drive rate (quality baseline)
    "pic_qb_is_backup",             # 1 if backup QB detected this game
    "pic_ol_sync_rate",             # % of OL players in their personal L1 optimal zone
    "pic_ol_quality_weighted",      # snap_pct x position_weight OL composite
    "pic_skill_sync_rate",          # % of skill players in their personal L1 optimal zone
    "pic_team_alignment",           # weighted avg of all player zone states (core signal)
    "pic_present_quality",          # quality score of players who ARE present
    "pic_cycle_confirmed",          # team dual_optimal AND qb_in_zone AND ol_sync >= 0.5
    "pic_cycle_degraded",           # team dual_optimal BUT backup QB (fade signal)
    "pic_alignment_score",          # continuous: team_alignment x (1 - is_backup)
    "pic_avg_player_epa",           # avg career EPA percentile of players on field (quality signal)
    "pic_players_tracked",          # number of players with valid PIC cycles this game

    # ── Layer 4: PIM stubs (Phase 4 — not yet implemented) ──────────────
    # pim_trajectory, pim_salary_signal, pim_volatility — removed until implemented

    # ── Contextual / Standard Features ──────────────────────────────────
    "is_home",                      # 1 if this is the home team row
    "rest_days",                    # days since last game
    "rest_advantage",               # rest_days - opponent_rest_days
    "is_dome",                      # 1 if dome/climate controlled
    "temp",                         # temperature (°F); 70 if dome
    "wind_mph",                     # wind speed; 0 if dome
    "is_cold_game",                 # 1 if temp < 40°F
    "is_windy_game",                # 1 if wind > 15 mph
    "is_div_game",                  # 1 if division game
    "season_week",                  # week number (1-18)
    "is_playoffs",                  # 1 if playoff game
    "opening_total",                # opening total line
    # closing_total: REMOVED — it defines the target (over_total). Using it as a
    # feature creates a circular dependency that collapses probability spread.
    "total_movement",               # closing - opening (sharp money signal — this IS useful)
    "opening_spread",               # opening spread (home perspective)
    # closing_spread: REMOVED — same target-leakage risk as closing_total.
    "spread_movement",              # closing - opening spread (sharp money signal)
    "implied_team_score",           # from spread + total
    "vegas_win_prob",               # market-implied win probability
    "epa_offense_avg",              # rolling EPA/play offense
    "epa_defense_avg",              # rolling EPA/play defense (allowed)
    "scoring_avg_5g",               # rolling 5-game points scored
    "scoring_avg_20g",              # rolling 20-game points scored
    "scoring_allowed_avg_5g",       # rolling 5-game points allowed
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def engineer_features(
    games_df: pd.DataFrame,
    drives_df: pd.DataFrame,
    pbp_df: pd.DataFrame = None,
    players_df: pd.DataFrame = None,
    contracts_df: pd.DataFrame = None,
    closing_lines_df: pd.DataFrame = None,
    epa_df: pd.DataFrame = None,
    qb_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build all features from raw NFL data.

    Input:  games_df (game-level), drives_df (drive-level)
    Output: game-TEAM level rows (~570/season) with all features + target

    Each game becomes TWO rows — one per team (mirrors golf's per-player structure).
    Features are computed from each team's perspective.
    """
    t_start = time.time()

    logger.info("[FEAT] ============================================================")
    logger.info("[FEAT] NFL FEATURE ENGINEERING PIPELINE START")
    logger.info(f"[FEAT]   Games: {len(games_df)}, Drives: {len(drives_df)}")
    logger.info("[FEAT] ============================================================")

    if games_df.empty or drives_df.empty:
        logger.error("[FEAT] Empty input — cannot proceed")
        return pd.DataFrame()

    # Step 1: Reshape games to game-team rows
    logger.info("[FEAT] Step 1: Reshaping to game-team rows...")
    df = _reshape_to_game_team(games_df, closing_lines_df)
    logger.info(f"[FEAT]   {len(df)} game-team rows ({df['team'].nunique()} teams, {df['season'].nunique()} seasons)")

    # Step 2: Target variable
    logger.info(f"[FEAT] Step 2: Building target variable ({TARGET_COLUMN})...")
    df = _build_target(df)

    # Step 3: Layer 1 — PTS cycle baseline
    logger.info("[FEAT] Step 3: Building P2S features (Layer 1)...")
    t3 = time.time()
    df = _build_p2s_features(df, drives_df)
    logger.info(f"[FEAT]   Step 3 done in {time.time()-t3:.1f}s")

    # Step 4: Layer 2 — P2SZ cycle baseline
    logger.info("[FEAT] Step 4: Building P2SZ features (Layer 2)...")
    t4 = time.time()
    df = _build_p2sz_features(df, drives_df)
    logger.info(f"[FEAT]   Step 4 done in {time.time()-t4:.1f}s")

    # Step 5: Intersection features
    logger.info("[FEAT] Step 5: Engineering intersection features...")
    df = _build_intersection_features(df)

    # Step 6: EPA features
    logger.info("[FEAT] Step 6: Merging EPA features...")
    df = _build_epa_features(df, epa_df)

    # Step 7: QB features
    logger.info("[FEAT] Step 7: Merging QB features...")
    df = _build_qb_features(df, qb_df)

    # Step 6b: Layer 3 — PIC Player Impact Curve
    logger.info("[FEAT] Step 6b: Building PIC features (Layer 3)...")
    t6b = time.time()
    if pbp_df is not None and players_df is not None:
        try:
            from nfl_predictor.pic_features import build_pic_features
            df = build_pic_features(df, pbp_df, players_df, logger=logger)
            logger.info(f"[FEAT]   Step 6b done in {time.time()-t6b:.1f}s")
        except Exception as e:
            logger.warning(f"[FEAT]   PIC build failed (non-fatal): {e} — filling stubs")
            for col in [c for c in FEATURE_COLUMNS if c.startswith("pic_")]:
                if col not in df.columns:
                    df[col] = np.nan
    else:
        logger.warning("[FEAT]   PIC skipped — pbp_df or players_df not provided")
        for col in [c for c in FEATURE_COLUMNS if c.startswith("pic_")]:
            if col not in df.columns:
                df[col] = np.nan

    # Step 6c: Impute missing PIC values using team-season averages
    # 2013-2015 PBP personnel data (offense_players column) is sparse,
    # causing 27-37% NaN rates in PIC features for those seasons.
    # Strategy: fill NaN with team-season median where available,
    # then fill remaining NaN with global median across all valid rows.
    # This is conservative — doesn't invent signal, just reduces XGBoost
    # split noise from excessive NaN propagation.
    pic_continuous_cols = [
        "pic_qb_in_zone", "pic_qb_career_rate",
        "pic_ol_sync_rate", "pic_ol_quality_weighted",
        "pic_skill_sync_rate", "pic_team_alignment", "pic_present_quality",
        "pic_avg_player_epa", "pic_alignment_score",
    ]
    pic_cols_present = [c for c in pic_continuous_cols if c in df.columns]
    if pic_cols_present and "team" in df.columns and "season" in df.columns:
        nan_before = df[pic_cols_present].isna().sum().sum()
        # Step 1: team-season median
        team_season_medians = df.groupby(["team", "season"])[pic_cols_present].median()
        for col in pic_cols_present:
            null_mask = df[col].isna()
            if null_mask.any():
                def fill_ts(row):
                    if pd.isna(row[col]):
                        try:
                            return team_season_medians.loc[(row["team"], row["season"]), col]
                        except KeyError:
                            return np.nan
                    return row[col]
                # Vectorized version using merge
                ts_fill = df[["team", "season"]].merge(
                    team_season_medians[[col]].reset_index(),
                    on=["team", "season"], how="left"
                )[col].reset_index(drop=True)
                df[col] = df[col].reset_index(drop=True).fillna(ts_fill)
        # Step 2: global median for any remaining NaN
        for col in pic_cols_present:
            global_median = df[col].median()
            if pd.notna(global_median):
                df[col] = df[col].fillna(global_median)
        nan_after = df[pic_cols_present].isna().sum().sum()
        logger.info(
            f"[FEAT]   PIC imputation: {nan_before} NaN → {nan_after} NaN "
            f"({nan_before - nan_after} filled via team-season median)"
        )

    # Step 7b: Layer 4 — Q1 Pace Signal (PIM)
    t7b = time.time()
    if pbp_df is not None:
        try:
            df = _build_q1_pace_features(df, pbp_df, logger=logger)
            logger.info(f"[FEAT]   Step 7b (Q1 pace) done in {time.time()-t7b:.1f}s")
        except Exception as e:
            logger.warning(f"[FEAT]   Q1 pace build failed (non-fatal): {e}")
    # Fill any remaining PIM stubs
    for col in [c for c in FEATURE_COLUMNS if c.startswith("pim_")]:
        if col not in df.columns:
            df[col] = np.nan

    # Step 8: Standard / contextual features
    logger.info("[FEAT] Step 8: Building standard features...")
    df = _build_standard_features(df, games_df)

    # Step 9: Validation and cleanup
    logger.info("[FEAT] Step 9: Validation...")
    df = _validate_and_clean(df)

    total = time.time() - t_start
    available = [f for f in FEATURE_COLUMNS if f in df.columns and df[f].notna().any()]
    logger.info("[FEAT] ============================================================")
    logger.info(f"[FEAT] PIPELINE COMPLETE: {len(df)} rows, {len(available)}/{len(FEATURE_COLUMNS)} features, {total:.1f}s")
    logger.info("[FEAT] ============================================================")
    return df


# ---------------------------------------------------------------------------
# Step 1: Reshape game-level → game-team rows
# ---------------------------------------------------------------------------

def _reshape_to_game_team(
    games_df: pd.DataFrame,
    closing_lines_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Convert one-row-per-game to two-rows-per-game (one per team).
    This mirrors GolfPredictor's natural per-player structure.

    Adds:
        team, opponent, is_home, team_score, opponent_score,
        team_spread (from team perspective), team_total_implied
    """
    rows = []

    logger.info(f"[FEAT] Reshape input: {len(games_df)} games, columns: {sorted(games_df.columns.tolist())}")
    if "total_line" in games_df.columns:
        n = games_df["total_line"].notna().sum()
        logger.info(f"[FEAT] total_line: {n}/{len(games_df)} valid, sample: {games_df['total_line'].dropna().head(3).tolist()}")
    else:
        logger.warning("[FEAT] total_line NOT in games_df columns!")

    # Merge closing lines if provided
    if closing_lines_df is not None and not closing_lines_df.empty:
        merge_cols = [c for c in ["game_id", "closing_spread", "closing_total"] if c in closing_lines_df.columns]
        if "game_id" in merge_cols:
            # Drop any pre-existing closing_total/closing_spread columns from games_df
            # to avoid pandas renaming them to _x/_y on merge (games_df has NaN stubs,
            # closing_lines_df has the real data — always prefer closing_lines_df).
            drop_cols = [c for c in ["closing_total", "closing_spread"] if c in games_df.columns and c in merge_cols]
            if drop_cols:
                games_df = games_df.drop(columns=drop_cols)
            games_df = games_df.merge(closing_lines_df[merge_cols], on="game_id", how="left")
            n_merged = games_df["closing_total"].notna().sum() if "closing_total" in games_df.columns else 0
            logger.info(f"[FEAT] Closing lines merged: {n_merged}/{len(games_df)} games have closing_total")

    for _, game in games_df.iterrows():
        base = {
            "game_id": game["game_id"],
            "season": game.get("season"),
            "week": game.get("week"),
            "game_type": game.get("game_type", "REG"),
            "gameday": game.get("gameday"),
            "opening_total": pd.to_numeric(game.get("total_line"), errors="coerce"),
            # closing_total: ONLY from closing_lines_df merge above — not from total_line.
            # If closing_lines_df wasn't provided, this stays NaN and _build_target
            # will log an error and produce NaN targets (rows dropped).
            "closing_total": pd.to_numeric(game.get("closing_total"), errors="coerce"),
            "is_dome": int(game.get("is_dome", 0)),
            "temp": float(game.get("temp", 70)) if game.get("is_dome", 0) == 1 else float(game.get("temp", 55)),
            "wind_mph": float(game.get("wind", 0)) if game.get("is_dome", 0) == 1 else float(game.get("wind", 0)),
            "is_div_game": int(game.get("div_game", 0)),
            "season_week": game.get("week"),
            "is_playoffs": int(game.get("game_type", "REG") != "REG"),
        }

        # Home team row
        home_spread = pd.to_numeric(game.get("spread_line"), errors="coerce")
        closing_spread = pd.to_numeric(game.get("closing_spread", game.get("spread_line")), errors="coerce")
        home_row = {
            **base,
            "team": game["home_team"],
            "opponent": game["away_team"],
            "is_home": 1,
            "team_score": pd.to_numeric(game.get("home_score"), errors="coerce"),
            "opponent_score": pd.to_numeric(game.get("away_score"), errors="coerce"),
            "opening_spread": home_spread,   # negative = home favored
            "closing_spread": closing_spread,
        }

        # Away team row (flip spread sign)
        away_row = {
            **base,
            "team": game["away_team"],
            "opponent": game["home_team"],
            "is_home": 0,
            "team_score": pd.to_numeric(game.get("away_score"), errors="coerce"),
            "opponent_score": pd.to_numeric(game.get("home_score"), errors="coerce"),
            "opening_spread": -home_spread if pd.notna(home_spread) else np.nan,
            "closing_spread": -closing_spread if pd.notna(closing_spread) else np.nan,
        }

        rows.extend([home_row, away_row])

    df = pd.DataFrame(rows)

    # Debug: check key columns made it through
    for col in ["opening_total", "closing_total", "team_score", "opponent_score", "is_home"]:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            logger.info(f"[FEAT] Reshape check: {col} — {n_valid}/{len(df)} valid")
        else:
            logger.warning(f"[FEAT] Reshape check: {col} — MISSING from DataFrame")

    # Derived contextual features
    df["spread_movement"] = df.get("closing_spread", np.nan) - df.get("opening_spread", np.nan)
    df["total_movement"] = df.get("closing_total", np.nan) - df.get("opening_total", np.nan)
    df["total_points"] = df["team_score"] + df["opponent_score"]
    df["result"] = df["team_score"] - df["opponent_score"]  # positive = win

    # Implied team score from spread + total
    # Use closing_spread when available (Odds API 2020+),
    # fall back to opening_spread from nfl_data_py schedules (covers 2013-2019)
    closing_total = df["closing_total"].fillna(df["opening_total"]) if "opening_total" in df.columns else df.get("closing_total", np.nan)
    best_spread = df["closing_spread"].fillna(df["opening_spread"]) if "opening_spread" in df.columns else df.get("closing_spread", np.nan)
    df["implied_team_score"] = (closing_total - best_spread) / 2

    # Vegas win probability from spread
    from scipy.stats import norm
    df["vegas_win_prob"] = best_spread.apply(
        lambda s: norm.cdf(-s / 13.45) if pd.notna(s) else 0.5
    )

    # Weather adjustments
    df["is_cold_game"] = (df["temp"] < 40).astype(int)
    df["is_windy_game"] = (df["wind_mph"] > 15).astype(int)

    # Sort chronologically
    if "gameday" in df.columns:
        df = df.sort_values(["gameday", "game_id", "is_home"], ascending=[True, True, False])
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Step 2: Target variable
# ---------------------------------------------------------------------------

def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build binary target: did the game go over the total line?
    Uses closing_total where available (preferred — sharpest number),
    falls back to opening_total per-row for games without closing data.
    Tags each row with 'target_line_source' for downstream debugging.
    """
    # Build best-available total: closing where we have it, opening otherwise
    if "closing_total" in df.columns and df["closing_total"].notna().sum() > 0:
        n_closing = df["closing_total"].notna().sum()
        best_total = df["closing_total"].copy()
        # Track source per row: 'closing' or 'opening_fallback'
        df["target_line_source"] = np.where(df["closing_total"].notna(), "closing", "missing")
        if "opening_total" in df.columns:
            n_fallback = best_total.isna().sum()
            best_total = best_total.fillna(df["opening_total"])
            n_filled = n_fallback - best_total.isna().sum()
            df.loc[(df["target_line_source"] == "missing") & df["opening_total"].notna(),
                   "target_line_source"] = "opening_fallback"
            logger.info(
                f"[FEAT] Target line: {n_closing} closing + {n_filled} opening fallback "
                f"= {best_total.notna().sum()} total"
            )
        else:
            logger.info(f"[FEAT] Using closing_total for target ({n_closing} valid)")

        # Per-season breakdown so you can see exactly where gaps are
        if "season" in df.columns:
            for season, grp in df.groupby("season"):
                n_c = (grp["target_line_source"] == "closing").sum()
                n_o = (grp["target_line_source"] == "opening_fallback").sum()
                n_m = (grp["target_line_source"] == "missing").sum()
                total = len(grp)
                # A fully-covered season has ~530-570 rows with closing data.
                # Week 1 (~30-34 rows) will always fall back to opening because
                # there's no prior-week closing line to compute movement from.
                # Flag only seasons where the gap is larger than week 1.
                if n_m > 0:
                    tag = " ← MISSING DATA"
                elif n_o > 40:
                    tag = " ← NEEDS CLOSING DATA"
                elif n_o > 0:
                    tag = " (week 1 fallback only — expected)"
                else:
                    tag = ""
                logger.info(
                    f"[FEAT]   Season {season}: {n_c} closing, {n_o} opening fallback, "
                    f"{n_m} missing{tag}"
                )
    elif "opening_total" in df.columns and df["opening_total"].notna().sum() > 0:
        best_total = df["opening_total"].copy()
        df["target_line_source"] = "opening_fallback"
        logger.warning("[FEAT] closing_total unavailable — falling back to opening_total")
    elif "total_line" in df.columns and df["total_line"].notna().sum() > 0:
        best_total = df["total_line"].copy()
        df["target_line_source"] = "opening_fallback"
        logger.warning("[FEAT] closing_total unavailable — falling back to total_line (opening line)")
    else:
        logger.error("[FEAT] No total line column found — target will be all NaN")
        df[TARGET_COLUMN] = np.nan
        df["target_line_source"] = "missing"
        return df

    n_missing = best_total.isna().sum()
    if n_missing > 0:
        logger.warning(f"[FEAT] Best-available total is NaN for {n_missing}/{len(df)} rows — those rows will be dropped")

    # Carry the exact line the target is scored against, and the signed margin,
    # so the backtester can score pushes as refunds (no-action) rather than as bets.
    df["target_total"] = best_total
    df["total_margin"] = df["total_points"] - best_total          # >0 over, <0 under, 0 push
    df["is_push"] = (df["total_margin"] == 0)

    # Binary training target EXCLUDES pushes (NaN -> dropped). A push is not a clean
    # over/under example; the old `> best_total` silently mislabeled it as an under.
    df[TARGET_COLUMN] = np.where(df["total_margin"] > 0, 1.0,
                          np.where(df["total_margin"] < 0, 0.0, np.nan))
    df.loc[best_total.isna() | df["total_points"].isna(), TARGET_COLUMN] = np.nan
    df.loc[best_total.isna() | df["total_points"].isna(), "is_push"] = False

    n_push = int(df["is_push"].sum())
    pos_rate = df[TARGET_COLUMN].mean()
    n_valid = df[TARGET_COLUMN].notna().sum()
    logger.info(
        f"[FEAT]   Target '{TARGET_COLUMN}': {n_valid}/{len(df)} decided rows "
        f"({n_push} pushes excluded), {pos_rate:.3f} over rate "
        f"(expect ~0.48-0.50 vs closing line)"
    )
    return df


# ---------------------------------------------------------------------------
# Core cycle baseline engine — shared by L1 and L2
# ---------------------------------------------------------------------------

def _compute_cycle_sequence(miss_flags: np.ndarray, min_gaps: int = 3) -> tuple:
    """
    Core cycle baseline computation — same math as GolfPredictor, NFL-appropriate terminology.

    Args:
        miss_flags: binary array (1 = miss/non-event, 0 = hit/event)
        min_gaps:   minimum observed gaps before cycle baseline is valid

    Returns:
        (baseline_values, ratio_values, streak_values, zone_values) — all length n
        All values are NaN until min_gaps gaps have been observed.
    """
    n = len(miss_flags)
    baseline_values = np.full(n, np.nan)
    ratio_values = np.full(n, np.nan)
    streak_values = np.full(n, np.nan)
    zone_values = np.full(n, np.nan)

    gaps = []          # completed gaps between miss events
    current_gap = 0    # drives since last miss
    current_streak = 0 # consecutive miss events

    for i in range(n):
        v = miss_flags[i]
        if np.isnan(v):
            continue

        if v == 1:  # miss event (non-scoring drive)
            if current_gap > 0:
                gaps.append(current_gap)
            current_gap = 0
            current_streak += 1
        else:       # hit event (scoring drive)
            current_gap += 1
            current_streak = 0

        # Need min_gaps completed gaps for a valid cycle baseline baseline
        if len(gaps) >= min_gaps:
            baseline_values[i] = float(np.mean(gaps))
            if baseline_values[i] > 0:
                ratio_values[i] = current_gap / baseline_values[i]
                # Optimal zone: [0.5, 1.0] — overdue for next event, prime reversion
                zone_values[i] = 1.0 if 0.5 <= ratio_values[i] <= 1.0 else 0.0

        streak_values[i] = float(current_streak)

    return baseline_values, ratio_values, streak_values, zone_values


def _compute_bsr_features(values: np.ndarray, min_history: int = 15) -> dict:
    """
    BSR (Baseline Scoring Rhythm) — autocorrelation cycle detection.
    Mirrors GolfPredictor _player_bsr().

    Returns dict with cycle_position, hot_streak_prob, cold_streak_prob, trend.
    """
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

        # Autocorrelation for cycle detection
        centered = valid - np.mean(valid)
        if np.std(centered) > 0:
            ac = np.correlate(centered, centered, mode="full")
            ac = ac[len(ac) // 2:]
            ac = ac / ac[0] if ac[0] != 0 else ac

            peaks = [
                j for j in range(2, min(len(ac) - 1, 20))
                if ac[j] > ac[j-1] and ac[j] > ac[j+1] and ac[j] > 0.08
            ]
            if peaks:
                cycle_len = peaks[0]
                rounds_since = len(valid) % cycle_len
                cycle_position[i] = rounds_since / cycle_len
            else:
                # Fallback: use mean scoring interval as cycle estimate
                # If team scores ~38% of drives, average cycle ≈ 2.6 drives
                mean_rate = np.mean(valid) if np.mean(valid) > 0 else 0.38
                est_cycle = max(2, round(1.0 / mean_rate)) if mean_rate > 0 else 3
                rounds_since = len(valid) % est_cycle
                cycle_position[i] = rounds_since / est_cycle

        # Hot/cold streak probability
        recent = valid[-5:] if len(valid) >= 5 else valid
        good = np.sum(recent > np.mean(valid))
        bad = np.sum(recent < np.mean(valid))

        # Compute transition rates from history
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

        # Trend (slope of last 5 values)
        if len(valid) >= 5:
            last5 = valid[-5:]
            slope = np.polyfit(np.arange(5), last5, 1)[0]
            trend[i] = slope

    return {
        "cycle_position": cycle_position,
        "hot_prob": hot_prob,
        "cold_prob": cold_prob,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# Step 3: Layer 1 — P2S features
# ---------------------------------------------------------------------------

def _build_p2s_features(df: pd.DataFrame, drives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P2S (Possession-to-Score) cycle baseline for each team.

    For each game-team row, we need the cycle baseline state computed from all drives
    BEFORE this game. This is the pre-game signal.

    Drive unit: a non-garbage-time offensive possession.
    Miss:  non-scoring drive (punt, turnover, turnover on downs, missed FG).
    Hit:   scoring drive (TD or FG).
    """
    required = ["game_id", "posteam", "is_miss", "is_scoring_drive", "is_countable_drive"]
    missing = [c for c in required if c not in drives_df.columns]
    if missing:
        logger.warning(f"[FEAT] PTS: drives_df missing {missing} — filling NaN")
        for col in [c for c in FEATURE_COLUMNS if c.startswith("p2s_")]:
            df[col] = np.nan
        return df

    # Sort drives chronologically
    sort_cols = [c for c in ["game_date", "game_id", "drive_id"] if c in drives_df.columns]
    drives_sorted = drives_df.sort_values(sort_cols).copy()

    # Build per-team drive sequences and compute cycle baseline
    # Result: dict of (team, game_id) → feature dict
    team_game_features = {}
    teams_processed = 0

    for team, team_drives in drives_sorted.groupby("posteam"):
        miss_flags = team_drives["is_miss"].values.astype(float)
        score_flags = team_drives["is_scoring_drive"].values.astype(float)
        game_ids = team_drives["game_id"].values

        # Compute cycle baseline sequence over all drives (ordered chronologically)
        baseline_vals, ratio_vals, streak_vals, zone_vals = _compute_cycle_sequence(miss_flags)

        # Compute BSR on scoring rate signal
        # Convert to binary per-drive scoring rate per game first
        bsr = _compute_bsr_features(score_flags)

        # Compute rolling scoring rates
        scoring_series = pd.Series(score_flags)
        conv_5g_raw = scoring_series.rolling(window=40, min_periods=10).mean()   # ~8 drives/game × 5 games
        conv_20g_raw = scoring_series.rolling(window=160, min_periods=40).mean() # ~8 × 20 games

        # Map drive-level features back to game level
        # For each game, take the LAST drive's feature values (pre-game state would be
        # the values computed at end of previous game — but since we use shift below, this
        # represents the state at the END of the last drive of this game, and we shift by
        # one game when merging. This is correct: we want the state GOING INTO the game.)
        game_indices = {}
        for idx, (drive_idx, gid) in enumerate(zip(range(len(game_ids)), game_ids)):
            if gid not in game_indices:
                game_indices[gid] = []
            game_indices[gid].append(idx)

        for gid, idxs in game_indices.items():
            last_idx = idxs[-1]  # last drive of this game
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
            logger.info(f"[FEAT]   PTS: processed {teams_processed} teams...")

    logger.info(f"[FEAT]   PTS: computed {len(team_game_features)} team-game records")

    # Join offensive features — using PREVIOUS game's values (shift by 1 game per team)
    df = _merge_lagged_features(df, team_game_features, "p2s_", drives_df)

    # Defensive PTS: opponent's OFFENSIVE L1 state, also using PREVIOUS game lag.
    # We want to know: going into this game, how has the opponent's offense been performing?
    # This uses the SAME lagged team_game_features dict (already shifted by _merge_lagged_features)
    # but looked up from the opponent's perspective.
    #
    # IMPORTANT: We must use the opponent's state BEFORE the current game — i.e., the
    # opponent's end-of-last-game state. Since team_game_features stores end-of-game state
    # keyed by (team, game_id), and _merge_lagged_features already shifts by one game,
    # we need to do the same shift for defensive lookups.
    #
    # Build a prev_game_id map for opponents too.
    game_order = (
        drives_df.drop_duplicates(subset=["game_id", "posteam"])
        .sort_values([c for c in ["game_date", "game_id"] if c in drives_df.columns])
        [["game_id", "posteam"]]
        .rename(columns={"posteam": "team"})
    )
    game_order["prev_game_id"] = game_order.groupby("team")["game_id"].shift(1)
    prev_game_map = game_order.set_index(["team", "game_id"])["prev_game_id"].to_dict()

    # For each row, look up opponent's state from opponent's PREVIOUS game
    def _get_def_feature(row, feat_key):
        opponent = row["opponent"]
        game_id = row["game_id"]
        # Get the opponent's previous game_id
        prev_gid = prev_game_map.get((opponent, game_id))
        if prev_gid is None:
            return np.nan
        return team_game_features.get((opponent, prev_gid), {}).get(feat_key, np.nan)

    logger.info("[FEAT]   Computing defensive PTS features (opponent pre-game state)...")
    df["p2s_def_baseline"] = df.apply(lambda r: _get_def_feature(r, "p2s_baseline"), axis=1)
    df["p2s_def_ratio"] = df.apply(lambda r: _get_def_feature(r, "p2s_ratio"), axis=1)
    df["p2s_def_optimal_zone"] = df.apply(lambda r: _get_def_feature(r, "p2s_optimal_zone"), axis=1)

    def_valid = df["p2s_def_baseline"].notna().sum()
    logger.info(f"[FEAT]   Defensive PTS features: {def_valid}/{len(df)} rows have data")

    return df


# ---------------------------------------------------------------------------
# Step 4: Layer 2 — P2SZ features
# ---------------------------------------------------------------------------

def _build_p2sz_features(df: pd.DataFrame, drives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P2SZ (Possession-to-Scoring Zone) cycle baseline for each team.

    Identical computation to PTS, but applied to FIELD POSITION signals
    rather than scoring signals. This creates an independent cycle.

    Miss (red zone cycle baseline):  drive that doesn't reach opponent's 20
    Hit:                   drive that enters red zone
    Miss (SP cycle baseline):        drive that doesn't reach opponent's 35
    Hit:                   drive that reaches scoring position
    """
    required_cols = ["game_id", "posteam", "reached_redzone", "reached_scoring_position"]
    missing = [c for c in required_cols if c not in drives_df.columns]
    if missing:
        logger.warning(f"[FEAT] P2SZ: drives_df missing {missing} — filling NaN")
        for col in [c for c in FEATURE_COLUMNS if c.startswith("p2sz_")]:
            df[col] = np.nan
        return df

    sort_cols = [c for c in ["game_date", "game_id", "drive_id"] if c in drives_df.columns]
    drives_sorted = drives_df.sort_values(sort_cols).copy()

    team_game_features = {}
    teams_processed = 0

    for team, team_drives in drives_sorted.groupby("posteam"):
        # Red zone cycle baseline: miss = didn't enter red zone
        rz_miss = (1 - team_drives["reached_redzone"].values).astype(float)
        rz_baseline, rz_ratio, _, rz_zone = _compute_cycle_sequence(rz_miss)

        # Scoring position cycle baseline: miss = didn't reach inside opp 35
        sp_miss = (1 - team_drives["reached_scoring_position"].values).astype(float)
        sp_baseline, sp_ratio, _, sp_zone = _compute_cycle_sequence(sp_miss)

        # 3rd down conversion cycle baseline
        if "third_down_conv_rate" in team_drives.columns:
            td_conv = team_drives["third_down_conv_rate"].values.astype(float)
            # Miss = below-average 3rd down conversion (per-drive rate)
            td_avg = np.nanmean(td_conv)
            td_miss = (td_conv < td_avg).astype(float)
            td_miss[np.isnan(td_conv)] = np.nan
            td_baseline, td_ratio, _, td_zone = _compute_cycle_sequence(td_miss, min_gaps=2)
        else:
            n = len(team_drives)
            td_baseline = td_ratio = td_zone = np.full(n, np.nan)

        # Rolling rates
        rz_series = pd.Series(team_drives["reached_redzone"].values.astype(float))
        sp_series = pd.Series(team_drives["reached_scoring_position"].values.astype(float))
        rz_5g = rz_series.rolling(window=40, min_periods=10).mean()
        sp_5g = sp_series.rolling(window=40, min_periods=10).mean()

        # Rolling drive quality
        yds_per_drive = pd.Series(
            team_drives.get("drive_yards_gained", pd.Series(np.nan, index=team_drives.index))
        ).rolling(window=80, min_periods=16).mean()  # ~8 drives/game × 10 games

        explosive_rate = pd.Series(
            team_drives.get("is_explosive_drive", pd.Series(0, index=team_drives.index))
        ).rolling(window=80, min_periods=16).mean()

        # Group by game — take last drive of each game
        game_ids = team_drives["game_id"].values
        game_indices = {}
        for idx, gid in enumerate(game_ids):
            if gid not in game_indices:
                game_indices[gid] = []
            game_indices[gid].append(idx)

        for gid, idxs in game_indices.items():
            last_idx = idxs[-1]
            team_game_features[(team, gid)] = {
                "p2sz_rz_baseline": rz_baseline[last_idx],
                "p2sz_rz_ratio": rz_ratio[last_idx],
                "p2sz_rz_optimal_zone": rz_zone[last_idx],
                "p2sz_rz_rate_5g": rz_5g.iloc[last_idx],
                "p2sz_sp_baseline": sp_baseline[last_idx],
                "p2sz_sp_ratio": sp_ratio[last_idx],
                "p2sz_sp_optimal_zone": sp_zone[last_idx],
                "p2sz_sp_rate_5g": sp_5g.iloc[last_idx],
                "p2sz_3rd_conv_baseline": td_baseline[last_idx] if not np.isnan(td_baseline[last_idx]) else np.nan,
                "p2sz_3rd_conv_optimal_zone": td_zone[last_idx] if not np.isnan(td_zone[last_idx]) else np.nan,
                "p2sz_yards_per_drive": yds_per_drive.iloc[last_idx],
                "p2sz_explosive_rate": explosive_rate.iloc[last_idx],
                "p2sz_avg_start_yard_line": (
                    team_drives["start_yard_line"].rolling(window=80, min_periods=8).mean().iloc[last_idx]
                    if "start_yard_line" in team_drives.columns else np.nan
                ),
            }

        teams_processed += 1
        if teams_processed % 8 == 0:
            logger.info(f"[FEAT]   P2SZ: processed {teams_processed} teams...")

    logger.info(f"[FEAT]   P2SZ: computed {len(team_game_features)} team-game records")
    df = _merge_lagged_features(df, team_game_features, "p2sz_", drives_df)

    return df


# ---------------------------------------------------------------------------
# Step 5: Intersection features — THE PRIMARY SIGNAL
# ---------------------------------------------------------------------------

def _build_intersection_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer L1 × L2 interaction signals explicitly.
    Not learned — these are engineered because the interaction is the hypothesis.
    """
    # Dual optimal: both L1 (scoring suppressed) AND L2 (drive quality suppressed) together
    has_l1 = "p2s_optimal_zone" in df.columns
    has_l2 = "p2sz_sp_optimal_zone" in df.columns

    if has_l1 and has_l2:
        df["p2s_p2sz_dual_optimal"] = (
            (df["p2s_optimal_zone"] == 1.0) & (df["p2sz_sp_optimal_zone"] == 1.0)
        ).astype(float)
        df.loc[df["p2s_optimal_zone"].isna() | df["p2sz_sp_optimal_zone"].isna(), "p2s_p2sz_dual_optimal"] = np.nan

        # Signal strength: continuous version (sum of both ratios, normalized)
        has_r1 = "p2s_ratio" in df.columns
        has_r2 = "p2sz_sp_ratio" in df.columns
        if has_r1 and has_r2:
            # Clip ratios to [0, 2] range — beyond 2 means very overdue but also data may be sparse
            r1 = df["p2s_ratio"].clip(0, 2).fillna(0)
            r2 = df["p2sz_sp_ratio"].clip(0, 2).fillna(0)
            df["p2s_p2sz_signal_strength"] = (r1 + r2) / 2
        else:
            df["p2s_p2sz_signal_strength"] = np.nan

        # Matchup edge: THIS team in dual optimal AND opponent defense is suppressed
        # (opponent's L1 optimal zone means their OFFENSE hasn't been scoring — but
        # we want opponent's DEFENSE to be suppressed, which means their defense
        # hasn't been stopping scoring. Proxy: opponent's p2s_def_ratio is high)
        if "p2s_def_ratio" in df.columns:
            df["matchup_dual_optimal"] = (
                (df["p2s_p2sz_dual_optimal"] == 1.0) &
                (df["p2s_def_ratio"] > 1.0)  # opponent defense overdue for giving up points
            ).astype(float)
        else:
            df["matchup_dual_optimal"] = np.nan

    else:
        df["p2s_p2sz_dual_optimal"] = np.nan
        df["p2s_p2sz_signal_strength"] = np.nan
        df["matchup_dual_optimal"] = np.nan

    # Log intersection statistics
    if "p2s_p2sz_dual_optimal" in df.columns:
        n_dual = (df["p2s_p2sz_dual_optimal"] == 1.0).sum()
        n_valid = df["p2s_p2sz_dual_optimal"].notna().sum()
        logger.info(f"[FEAT]   Dual optimal: {n_dual}/{n_valid} game-team rows ({n_dual/n_valid*100:.1f}% if valid)")

    return df


# ---------------------------------------------------------------------------
# Step 8: Standard contextual features
# ---------------------------------------------------------------------------

def _build_epa_features(df: pd.DataFrame, epa_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge pre-game rolling EPA into game-team rows."""
    if epa_df is None or epa_df.empty:
        logger.warning("[FEAT] EPA data not provided — epa_offense_avg and epa_defense_avg will be NaN")
        df["epa_offense_avg"] = np.nan
        df["epa_defense_avg"] = np.nan
        return df

    epa_cols = [c for c in ["game_id", "team", "epa_off_5g", "epa_def_5g", "epa_off_season", "epa_def_season"]
                if c in epa_df.columns]
    epa = epa_df[epa_cols].copy()

    df = df.merge(epa, on=["game_id", "team"], how="left")
    df = df.reset_index(drop=True)

    # Use 5g rolling as primary, fall back to season avg
    df["epa_offense_avg"] = df["epa_off_5g"].fillna(df.get("epa_off_season", np.nan))
    df["epa_defense_avg"] = df["epa_def_5g"].fillna(df.get("epa_def_season", np.nan))

    n_off = df["epa_offense_avg"].notna().sum()
    n_def = df["epa_defense_avg"].notna().sum()
    logger.info(f"[FEAT]   EPA merged: offense {n_off}/{len(df)} valid, defense {n_def}/{len(df)} valid")

    return df


def _build_qb_features(df: pd.DataFrame, qb_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge pre-game QB efficiency features into game-team rows."""
    if qb_df is None or qb_df.empty:
        logger.warning("[FEAT] QB data not provided — QB features will be NaN")
        return df

    qb_cols = [c for c in ["game_id", "team", "qb_epa_5g", "qb_epa_season",
                            "qb_completion_pct", "qb_air_yards", "is_backup_qb"]
               if c in qb_df.columns]
    qb = qb_df[qb_cols].copy()

    df = df.merge(qb, on=["game_id", "team"], how="left")
    df = df.reset_index(drop=True)

    n_valid = df["qb_epa_5g"].notna().sum() if "qb_epa_5g" in df.columns else 0
    logger.info(f"[FEAT]   QB features merged: {n_valid}/{len(df)} valid")

    return df


def _build_standard_features(df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """Build EPA rolling averages, scoring averages, rest, weather."""

    # Rest days computation
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"])
        df = df.sort_values(["team", "gameday"]).reset_index(drop=True)
        df["prev_gameday"] = df.groupby("team")["gameday"].shift(1)
        df["rest_days"] = (df["gameday"] - df["prev_gameday"]).dt.days
        df["rest_days"] = df["rest_days"].fillna(7).clip(4, 21)  # cap at reasonable range

        # Rest advantage vs. opponent
        rest_map = df.set_index(["game_id", "team"])["rest_days"].to_dict()
        df["opponent_rest"] = df.apply(
            lambda r: rest_map.get((r["game_id"], r["opponent"]), np.nan), axis=1
        )
        df["rest_advantage"] = df["rest_days"] - df["opponent_rest"]

    # Build EPA rolling averages from games data — need to join EPA back
    # nflfastR pbp has EPA by team — we'll use game-level agg if available
    if "epa_offense_avg" not in df.columns:
        df["epa_offense_avg"] = np.nan
        df["epa_defense_avg"] = np.nan

    # Rolling scoring averages (from team_score column)
    if "team_score" in df.columns:
        df["scoring_avg_5g"] = df.groupby("team")["team_score"].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).mean()
        )
        df["scoring_avg_20g"] = df.groupby("team")["team_score"].transform(
            lambda x: x.shift(1).rolling(window=20, min_periods=5).mean()
        )

    if "opponent_score" in df.columns:
        df["scoring_allowed_avg_5g"] = df.groupby("team")["opponent_score"].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).mean()
        )

    return df


# ---------------------------------------------------------------------------
# Merge helper — join lagged features to game-team rows
# ---------------------------------------------------------------------------

def _merge_lagged_features(
    df: pd.DataFrame,
    team_game_features: dict,
    prefix: str,
    drives_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge drive-level computed features onto game-team rows.

    CRITICAL: We want the state BEFORE the current game, not including it.
    So we look up the feature values from the PREVIOUS game's end state.

    For each (team, game_id) row, look up features from (team, prev_game_id).
    """
    # Build ordered game list per team from drives
    game_order = (
        drives_df.drop_duplicates(subset=["game_id", "posteam"])
        .sort_values([c for c in ["game_date", "game_id"] if c in drives_df.columns])
        [["game_id", "posteam"]]
        .rename(columns={"posteam": "team"})
    )

    # Create prev_game_id mapping
    game_order["prev_game_id"] = game_order.groupby("team")["game_id"].shift(1)
    prev_game_map = game_order.set_index(["team", "game_id"])["prev_game_id"].to_dict()

    # Map features from previous game's state
    feature_cols = [c for c in list(team_game_features.values())[0].keys()] if team_game_features else []

    for col in feature_cols:
        if col.startswith(prefix.rstrip("_")) or col.startswith(prefix):
            col_data = []
            for _, row in df.iterrows():
                team = row["team"]
                gid = row["game_id"]
                prev_gid = prev_game_map.get((team, gid))
                if prev_gid is not None:
                    feats = team_game_features.get((team, prev_gid), {})
                else:
                    feats = {}
                col_data.append(feats.get(col, np.nan))
            df[col] = col_data

    # Log coverage
    if feature_cols:
        sample_col = feature_cols[0]
        if sample_col in df.columns:
            valid = df[sample_col].notna().sum()
            logger.info(f"[FEAT]   {prefix}* features merged: {valid}/{len(df)} rows have data")

    return df


# ---------------------------------------------------------------------------
# Step 9: Validation and cleanup
# ---------------------------------------------------------------------------

def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Final NaN audit and row filtering."""
    initial = len(df)

    # Drop rows without target (can't train/evaluate)
    df = df.dropna(subset=[TARGET_COLUMN])
    after_target = len(df)
    logger.info(f"[FEAT]   Dropped {initial - after_target} rows with NaN target")

    # Feature coverage report
    available = [f for f in FEATURE_COLUMNS if f in df.columns]
    logger.info(f"[FEAT]   Feature coverage: {len(available)}/{len(FEATURE_COLUMNS)}")
    for col in available:
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100
        if n_valid == 0:
            logger.warning(f"[FEAT]     {col}: 0 valid values")
        elif pct < 50:
            logger.warning(f"[FEAT]     {col}: {pct:.0f}% valid ({n_valid}/{len(df)})")
        else:
            logger.debug(f"[FEAT]     {col}: {pct:.0f}% valid")

    # L1/L2 intersection diagnostic
    if "p2s_p2sz_dual_optimal" in df.columns:
        dual = df[df["p2s_p2sz_dual_optimal"] == 1.0]
        if len(dual) > 10:
            target_rate_overall = df[TARGET_COLUMN].mean()
            target_rate_dual = dual[TARGET_COLUMN].mean()
            logger.info(f"[FEAT]   INTERSECTION DIAGNOSTIC:")
            logger.info(f"[FEAT]     Overall {TARGET_COLUMN} rate: {target_rate_overall:.3f}")
            logger.info(f"[FEAT]     Dual optimal {TARGET_COLUMN} rate: {target_rate_dual:.3f}")
            logger.info(f"[FEAT]     Lift: {(target_rate_dual - target_rate_overall):.3f} ({(target_rate_dual/target_rate_overall - 1)*100:+.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Layer 4: Q1 Pace Signal (PIM — Player Impact Momentum)
# ---------------------------------------------------------------------------

def _build_q1_pace_features(
    df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    logger=None,
) -> pd.DataFrame:
    """
    Compute Q1 and H1 pace signals from play-by-play and merge onto game-team rows.

    The core insight (validated on 887 games, 2016-2019 SBR data):
      Q1 pace = combined Q1 score / closing_total
      - Q1 pct <= 10% → over rate 25% (strong under signal)
      - Q1 pct >= 25% → over rate 69-82% (strong over signal)
      - H1 pct <  30% → over rate 11% (near-certain under)
      - H1 pct >= 80% → over rate 100% (56/56 games)

    These are historical features — Q1 data from the game BEING PREDICTED
    is used at training time. At inference time this is a live signal
    (read after Q1 ends). The model learns the historical correlation.

    Features added (per team-game row):
      pim_q1_pct_of_total     - game's combined Q1 score / closing_total
      pim_q1_epa_per_drive    - avg EPA per Q1 drive for this team
      pim_q1_scoring_drives   - number of scoring drives in Q1 for this team
      pim_q1_cycle_confirmed  - 1 if dual_optimal AND q1_pct >= 0.20
      pim_q1_cycle_contradicted - 1 if dual_optimal AND q1_pct <= 0.10
      pim_h1_pct_of_total     - game's combined H1 score / closing_total
    """
    if pbp_df is None or len(pbp_df) == 0:
        return df

    required = {"qtr", "game_id", "total_home_score", "total_away_score"}
    if not required.issubset(pbp_df.columns):
        if logger:
            missing = required - set(pbp_df.columns)
            logger.warning(f"[PIM]   Q1 pace skipped — missing PBP columns: {missing}")
        return df

    if logger:
        logger.info("[PIM] Computing Q1/H1 pace features...")

    # ── Game-level Q1 combined score ─────────────────────────────────────────
    # Use last play of Q1 per game — total_home_score + total_away_score
    q1_plays = pbp_df[pbp_df["qtr"] == 1].copy()
    h1_plays = pbp_df[pbp_df["qtr"].isin([1, 2])].copy()

    def get_combined_at_end(plays):
        if len(plays) == 0:
            return pd.DataFrame(columns=["game_id", "combined"])
        last = plays.groupby("game_id").last()[
            ["total_home_score", "total_away_score"]
        ].reset_index()
        last["combined"] = last["total_home_score"] + last["total_away_score"]
        return last[["game_id", "combined"]]

    q1_combined = get_combined_at_end(q1_plays).rename(columns={"combined": "q1_combined"})
    h1_combined = get_combined_at_end(h1_plays).rename(columns={"combined": "h1_combined"})

    # ── Team-level Q1 EPA and drive stats ────────────────────────────────────
    if "epa" in q1_plays.columns and "posteam" in q1_plays.columns:
        # EPA per drive per team in Q1
        q1_drive_epa = (
            q1_plays[q1_plays["epa"].notna()]
            .groupby(["game_id", "posteam", "fixed_drive"])["epa"]
            .sum()
            .reset_index()
            .groupby(["game_id", "posteam"])["epa"]
            .mean()
            .reset_index()
            .rename(columns={"epa": "pim_q1_epa_per_drive", "posteam": "team"})
        )
    else:
        q1_drive_epa = None

    # Scoring drives in Q1 per team
    if "fixed_drive_result" in q1_plays.columns and "posteam" in q1_plays.columns:
        q1_last_play = q1_plays.groupby(
            ["game_id", "posteam", "fixed_drive"]
        ).last().reset_index()
        q1_last_play["is_scoring"] = q1_last_play["fixed_drive_result"].isin(
            ["Touchdown", "Field goal"]
        ).astype(int)
        q1_scoring = (
            q1_last_play.groupby(["game_id", "posteam"])["is_scoring"]
            .sum()
            .reset_index()
            .rename(columns={"is_scoring": "pim_q1_scoring_drives", "posteam": "team"})
        )
    else:
        q1_scoring = None

    # ── Merge game-level Q1/H1 pace onto df ──────────────────────────────────
    df = df.merge(q1_combined, on="game_id", how="left")
    df = df.merge(h1_combined, on="game_id", how="left")

    # Compute pace as fraction of closing_total
    if "closing_total" in df.columns:
        df["pim_q1_pct_of_total"] = df["q1_combined"] / df["closing_total"].replace(0, np.nan)
        df["pim_h1_pct_of_total"] = df["h1_combined"] / df["closing_total"].replace(0, np.nan)
    else:
        df["pim_q1_pct_of_total"] = np.nan
        df["pim_h1_pct_of_total"] = np.nan

    df.drop(columns=["q1_combined", "h1_combined"], inplace=True, errors="ignore")

    # ── Merge team-level Q1 stats ─────────────────────────────────────────────
    if q1_drive_epa is not None:
        df = df.merge(
            q1_drive_epa,
            left_on=["game_id", "team"],
            right_on=["game_id", "team"],
            how="left"
        )
    else:
        df["pim_q1_epa_per_drive"] = np.nan

    if q1_scoring is not None:
        df = df.merge(
            q1_scoring,
            left_on=["game_id", "team"],
            right_on=["game_id", "team"],
            how="left"
        )
    else:
        df["pim_q1_scoring_drives"] = np.nan

    # ── Cycle × Q1 multiplicative interaction ────────────────────────────────
    # This is the lighthouse signal: pre-game cycle prediction confirmed or
    # contradicted by Q1 pace. When both point the same direction, edge is
    # dramatically larger than either signal alone.
    #
    # Dual optimal = team in L1+L2 scoring cycle alignment pre-game
    # Q1 confirmed = pace >= 20% (game on track to go over)
    # Q1 contradicted = pace <= 10% (game tracking under)
    #
    # Historical pattern: dual_optimal + Q1 confirmed → ~75%+ over rate
    #                     dual_optimal + Q1 contradicted → ~20% over rate
    if "p2s_p2sz_dual_optimal" in df.columns:
        dual = (df["p2s_p2sz_dual_optimal"] == 1.0)
        q1_over_pace = (df["pim_q1_pct_of_total"] >= 0.20)
        q1_under_pace = (df["pim_q1_pct_of_total"] <= 0.10)

        df["pim_q1_cycle_confirmed"] = (dual & q1_over_pace).astype(float)
        df["pim_q1_cycle_contradicted"] = (dual & q1_under_pace).astype(float)

        # Set to NaN where Q1 pace is missing (no PBP data for that game)
        q1_missing = df["pim_q1_pct_of_total"].isna()
        df.loc[q1_missing, "pim_q1_cycle_confirmed"] = np.nan
        df.loc[q1_missing, "pim_q1_cycle_contradicted"] = np.nan
    else:
        df["pim_q1_cycle_confirmed"] = np.nan
        df["pim_q1_cycle_contradicted"] = np.nan

    # ── Logging ──────────────────────────────────────────────────────────────
    if logger:
        n_valid = df["pim_q1_pct_of_total"].notna().sum()
        n_total = len(df)
        logger.info(f"[PIM]   Q1 pace: {n_valid}/{n_total} games have Q1 data")
        if n_valid > 0:
            q1 = df["pim_q1_pct_of_total"].dropna()
            logger.info(
                f"[PIM]   Q1 pct distribution: "
                f"mean={q1.mean():.3f}, "
                f"p25={q1.quantile(0.25):.3f}, "
                f"p50={q1.quantile(0.50):.3f}, "
                f"p75={q1.quantile(0.75):.3f}"
            )
            # Diagnostic: over rate by Q1 bucket
            if "over_total" in df.columns:
                buckets = [0, 0.10, 0.20, 0.25, 0.30, 1.0]
                labels = ["<=10%", "10-20%", "20-25%", "25-30%", ">30%"]
                df["_q1_bucket"] = pd.cut(
                    df["pim_q1_pct_of_total"], bins=buckets, labels=labels
                )
                bucket_stats = df.groupby("_q1_bucket", observed=False)["over_total"].agg(["mean", "count"])
                logger.info("[PIM]   Q1 pace → over rate (training set):")
                for label, row in bucket_stats.iterrows():
                    if row["count"] >= 10:
                        logger.info(
                            f"[PIM]     {label}: {row['mean']*100:.1f}% over "
                            f"(n={int(row['count'])})"
                        )
                df.drop(columns=["_q1_bucket"], inplace=True, errors="ignore")

        # Cycle confirmation diagnostic
        n_confirmed = df["pim_q1_cycle_confirmed"].sum()
        n_contradicted = df["pim_q1_cycle_contradicted"].sum()
        logger.info(
            f"[PIM]   Cycle+Q1 confirmed: {n_confirmed:.0f} games | "
            f"contradicted: {n_contradicted:.0f} games"
        )

    return df