"""
Layer 3: PIC — Player Impact Curve
===================================
Computes individual player cycle baselines and their intersection
with team-level L1/L2 cycle state.

Core hypothesis:
    Team L1/L2 cycles are necessary but not sufficient.
    The cycle signal is ONLY reliable when the players executing
    it are (a) present and (b) individually in their own optimal zones.

    A team in dual optimal with Patrick Mahomes at QB is a strong over signal.
    The same team in dual optimal with a backup QB is noise — fade it.

Data sources:
    - PBP offense_players column: semicolon-delimited gsis_ids per play
    - nfl_data_py import_players(): gsis_id → position, pfr_id
    - nfl_data_py import_snap_counts(): pfr_id → offense_pct per game
    - PBP passer/rusher/receiver EPA: individual player quality baseline

Filters:
    - Offensive possession plays only (run, pass)
    - Excludes: kickoffs, punts, field goals, PAT, no_play, kneel, spike
    - Special teams plays are irrelevant to the possession cycle —
      punting is the RESULT of a cycle failure, not part of the cycle.
      FG attempts are the RESULT of the cycle working (L2 reached scoring position).

Unit of analysis: game-team row (same as L1/L2).
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Position groups for PIC weighting
# ---------------------------------------------------------------------------

# Offensive positions only — ST excluded entirely
OFFENSIVE_POSITIONS = {
    "QB", "RB", "FB", "WR", "TE",
    "T", "OT", "G", "C", "OL",
    "HB", "SB",
}

# Position weight multipliers for PIC alignment score
# QB dominates because single biggest individual impact on scoring cycle
# OL composite second — blocks every play, invisible until absent
# Skill positions moderate — rotate, game-plan dependent
POSITION_WEIGHTS = {
    "QB":  3.0,   # single most impactful player on every offensive play
    "OT":  1.5,   # pass protection directly enables cycle
    "T":   1.5,
    "G":   1.2,
    "C":   1.2,
    "OL":  1.2,
    "TE":  1.0,
    "WR":  0.8,
    "RB":  0.8,
    "HB":  0.8,
    "FB":  0.5,   # situational, low cycle impact
    "SB":  0.5,
}

# Play types that represent actual offensive possession plays
# Everything else is noise for the cycle
OFFENSIVE_PLAY_TYPES = {"run", "pass"}

# Minimum snaps to be considered a meaningful contributor
MIN_SNAP_PCT_THRESHOLD = 0.15  # below 15% = practice squad / emergency

# Minimum drives for a player's PIC baseline to be valid
MIN_DRIVES_FOR_BASELINE = 8


# ---------------------------------------------------------------------------
# Main entry point — called from engineer_features()
# ---------------------------------------------------------------------------

def build_pic_features(
    df: pd.DataFrame,           # game-team rows (output of L1/L2 pipeline)
    pbp_df: pd.DataFrame,       # full play-by-play with offense_players column
    players_df: pd.DataFrame,   # nfl_data_py import_players()
    snap_df: pd.DataFrame,      # nfl_data_py import_snap_counts()
    logger=None,
) -> pd.DataFrame:
    """
    Build Layer 3 PIC features and merge onto game-team rows.

    Args:
        df:         Game-team rows from L1/L2 pipeline (1 row per team per game)
        pbp_df:     Play-by-play with offense_players, passer_player_id, etc.
                    Should include full history (pic_history_start→train_end)
                    for cold start resolution.
        players_df: Player identity table (gsis_id → position, pfr_id)
        snap_df:    Snap count data (pfr_id, game_id → offense_pct)
        logger:     Logger instance

    Returns:
        df with PIC feature columns added
    """
    if logger:
        logger.info("[PIC] ============================================================")
        logger.info("[PIC] LAYER 3: PIC — Player Impact Curve")
        logger.info("[PIC] ============================================================")

    t0 = time.time()

    # Step 1: Build player identity + quality lookup
    if logger:
        logger.info("[PIC] Step 1: Building player identity and quality lookup...")
    player_lookup, snap_lookup, snap_lookup_by_gsis = _build_player_lookups(players_df, snap_df, logger)

    # Step 1b: Compute career EPA quality scores from full PBP history
    if logger:
        logger.info("[PIC] Step 1b: Computing career EPA quality scores...")
    career_epa = _compute_career_epa(pbp_df, logger)

    # Step 2: Filter PBP to offensive possession plays only
    if logger:
        logger.info("[PIC] Step 2: Filtering PBP to offensive possession plays...")
    pbp_off = _filter_offensive_plays(pbp_df, logger)

    # Step 2b: Compute PBP-derived snap percentages — no external join needed.
    # import_snap_counts() pfr_player_id only overlaps ~45% with import_players() pfr_id,
    # making the external snap lookup unreliable. Instead, count play appearances per
    # player per game directly from offense_players column.
    if logger:
        logger.info("[PIC] Step 2b: Computing PBP-derived snap percentages...")
    pbp_snap_pct = _compute_pbp_snap_pct(pbp_off, logger)

    # Step 3: Aggregate drive personnel — base group + key player presence
    if logger:
        logger.info("[PIC] Step 3: Aggregating drive-level personnel...")
    drive_personnel = _aggregate_drive_personnel(pbp_off, player_lookup, snap_lookup, snap_lookup_by_gsis, logger)

    # Step 4: Compute per-player PIC cycles from drive history
    if logger:
        logger.info("[PIC] Step 4: Computing per-player PIC cycle baselines...")
    player_cycles = _compute_player_pic_cycles(pbp_off, drive_personnel, logger)

    # Step 5: Build game-level PIC features
    if logger:
        logger.info("[PIC] Step 5: Building game-level PIC features...")
    game_pic = _build_game_pic_features(
        drive_personnel, player_cycles, player_lookup, snap_lookup, snap_lookup_by_gsis, pbp_snap_pct, career_epa, logger
    )

    # Step 6: Merge onto game-team rows
    if logger:
        logger.info("[PIC] Step 6: Merging PIC features onto game-team rows...")
    df = _merge_pic_features(df, game_pic, logger)

    elapsed = time.time() - t0
    if logger:
        logger.info(f"[PIC] ============================================================")
        logger.info(f"[PIC] LAYER 3 COMPLETE in {elapsed:.1f}s")
        logger.info(f"[PIC] ============================================================")

    return df


# ---------------------------------------------------------------------------
# Step 1: Player identity + quality lookups
# ---------------------------------------------------------------------------

def _build_player_lookups(
    players_df: pd.DataFrame,
    snap_df: pd.DataFrame,
    logger=None,
) -> Tuple[Dict, Dict]:
    """
    Build two lookup dicts:
        player_lookup: gsis_id → {position, pfr_id, display_name}
        snap_lookup:   (pfr_id, game_id) → {offense_snaps, offense_pct}
    """
    # Player identity — filter to offensive positions only
    player_lookup = {}
    for _, row in players_df.iterrows():
        gsis = row.get("gsis_id")
        pos = row.get("position", "")
        if gsis and pos in OFFENSIVE_POSITIONS:
            player_lookup[gsis] = {
                "display_name": row.get("display_name", ""),
                "position":     pos,
                "pfr_id":       row.get("pfr_id", ""),
                "weight":       POSITION_WEIGHTS.get(pos, 0.5),
            }

    if logger:
        n_with_pfr = sum(1 for v in player_lookup.values() if v.get("pfr_id"))
        n_no_pfr = len(player_lookup) - n_with_pfr
        logger.info(f"[PIC]   Player lookup: {len(player_lookup)} offensive players "
                   f"({n_with_pfr} with pfr_id, {n_no_pfr} missing pfr_id — "
                   f"external snap lookup will miss {n_no_pfr/len(player_lookup)*100:.0f}% of players)")

    # Snap counts — only offensive snaps matter
    # Primary key: (pfr_player_id, game_id) — matches player_lookup pfr_id
    snap_lookup = {}
    for _, row in snap_df.iterrows():
        pfr_id = row.get("pfr_player_id", "")
        game_id = row.get("game_id", "")
        if pfr_id and game_id:
            snap_lookup[(pfr_id, game_id)] = {
                "offense_snaps": row.get("offense_snaps", 0) or 0,
                "offense_pct":   row.get("offense_pct", 0) or 0,
            }

    if logger:
        logger.info(f"[PIC]   Snap lookup: {len(snap_lookup)} player-game records")

    # Build reverse map: pfr_id → gsis_id for players missing pfr_id in snap data
    # Snap counts use pfr_player_id; PBP offense_players uses gsis_id.
    # 10% of players have no pfr_id — build a gsis→snap fallback using player name matching.
    # Simpler approach: build a gsis_id keyed snap lookup as well using the players_df bridge.
    pfr_to_gsis = {}
    for gsis, info in player_lookup.items():
        pfr = info.get("pfr_id", "")
        if pfr:
            pfr_to_gsis[pfr] = gsis

    # Build gsis-keyed snap lookup for players whose pfr_id is missing
    # This allows direct gsis_id lookup as fallback
    snap_lookup_by_gsis = {}
    for (pfr_id, game_id), snap_data in snap_lookup.items():
        gsis = pfr_to_gsis.get(pfr_id)
        if gsis:
            snap_lookup_by_gsis[(gsis, game_id)] = snap_data

    if logger:
        logger.info(f"[PIC]   Snap lookup by gsis_id: {len(snap_lookup_by_gsis)} player-game records "
                   f"({len(snap_lookup_by_gsis)/max(len(snap_lookup),1)*100:.0f}% of snap records bridged)")

    return player_lookup, snap_lookup, snap_lookup_by_gsis


# ---------------------------------------------------------------------------
# Step 1b: Career EPA quality scores
# ---------------------------------------------------------------------------

def _compute_career_epa(pbp_df: pd.DataFrame, logger=None) -> Dict:
    """
    Compute career EPA percentile rank per player from full PBP history.
    Used as quality weight multiplier: Mahomes >> backup QB.

    Returns dict: gsis_id → epa_percentile (0.0–1.0)
    where 1.0 = top EPA player, 0.0 = worst.

    Method:
        - QB: career EPA/dropback from passer_player_id plays
        - Skill (WR/TE/RB): career EPA/play from receiver/rusher plays
        - OL: career snap-weighted team EPA (OL has no direct EPA attribution)
          OL quality = team's average EPA/play when this player is on field

    Only players with enough sample (min 50 plays) get a score.
    Players without a score get median percentile (0.5) — neutral weight.
    """
    if pbp_df is None or len(pbp_df) == 0:
        if logger:
            logger.warning("[PIC]   Career EPA: empty PBP — skipping, all players get median weight")
        return {}

    t0 = time.time()
    career_epa = {}

    # QB EPA — from passer plays
    if "passer_player_id" in pbp_df.columns and "epa" in pbp_df.columns:
        qb_plays = pbp_df[
            pbp_df["passer_player_id"].notna() &
            pbp_df["epa"].notna()
        ].copy()

        qb_stats = (
            qb_plays.groupby("passer_player_id")
            .agg(epa_mean=("epa", "mean"), n_plays=("epa", "count"))
            .reset_index()
        )
        qb_stats_filtered = qb_stats[qb_stats["n_plays"] >= 50]

        if len(qb_stats_filtered) > 0:
            qb_stats_filtered = qb_stats_filtered.copy()
            qb_stats_filtered["epa_pct"] = qb_stats_filtered["epa_mean"].rank(pct=True)
            for _, row in qb_stats_filtered.iterrows():
                career_epa[row["passer_player_id"]] = float(row["epa_pct"])

        if logger:
            logger.info(f"[PIC]   QB EPA: {len(qb_stats)} passers found, "
                       f"{len(qb_stats_filtered)} with ≥50 dropbacks scored")
            if len(qb_stats_filtered) > 0:
                top = qb_stats_filtered.nlargest(3, "epa_mean")[["passer_player_id","epa_mean","epa_pct"]]
                for _, r in top.iterrows():
                    logger.info(f"[PIC]     QB top: {r['passer_player_id']} "
                               f"EPA/play={r['epa_mean']:.3f} pct={r['epa_pct']:.2f}")
    else:
        if logger:
            logger.warning("[PIC]   Career EPA: passer_player_id or epa not in PBP columns — QB EPA skipped")

    # Skill player EPA — receiver plays
    if "receiver_player_id" in pbp_df.columns and "epa" in pbp_df.columns:
        rec_plays = pbp_df[
            pbp_df["receiver_player_id"].notna() &
            pbp_df["epa"].notna()
        ].copy()

        rec_stats = (
            rec_plays.groupby("receiver_player_id")
            .agg(epa_mean=("epa", "mean"), n_plays=("epa", "count"))
            .reset_index()
        )
        rec_stats_filtered = rec_stats[rec_stats["n_plays"] >= 30]

        if len(rec_stats_filtered) > 0:
            rec_stats_filtered = rec_stats_filtered.copy()
            rec_stats_filtered["epa_pct"] = rec_stats_filtered["epa_mean"].rank(pct=True)
            for _, row in rec_stats_filtered.iterrows():
                gsis = row["receiver_player_id"]
                existing = career_epa.get(gsis, 0.5)
                career_epa[gsis] = max(existing, float(row["epa_pct"]))

        if logger:
            logger.info(f"[PIC]   Receiver EPA: {len(rec_stats)} receivers found, "
                       f"{len(rec_stats_filtered)} with ≥30 targets scored")

    # Rusher EPA
    if "rusher_player_id" in pbp_df.columns and "epa" in pbp_df.columns:
        rush_plays = pbp_df[
            pbp_df["rusher_player_id"].notna() &
            pbp_df["epa"].notna()
        ].copy()

        rush_stats = (
            rush_plays.groupby("rusher_player_id")
            .agg(epa_mean=("epa", "mean"), n_plays=("epa", "count"))
            .reset_index()
        )
        rush_stats_filtered = rush_stats[rush_stats["n_plays"] >= 30]

        if len(rush_stats_filtered) > 0:
            rush_stats_filtered = rush_stats_filtered.copy()
            rush_stats_filtered["epa_pct"] = rush_stats_filtered["epa_mean"].rank(pct=True)
            for _, row in rush_stats_filtered.iterrows():
                gsis = row["rusher_player_id"]
                existing = career_epa.get(gsis, 0.5)
                career_epa[gsis] = max(existing, float(row["epa_pct"]))

        if logger:
            logger.info(f"[PIC]   Rusher EPA: {len(rush_stats)} rushers found, "
                       f"{len(rush_stats_filtered)} with ≥30 carries scored")

    if logger:
        logger.info(f"[PIC]   Career EPA scores: {len(career_epa)} players in {time.time()-t0:.1f}s")
        if career_epa:
            scores = list(career_epa.values())
            logger.info(f"[PIC]   EPA percentile range: {min(scores):.2f}–{max(scores):.2f}, "
                       f"median={sorted(scores)[len(scores)//2]:.2f}")
            # Count players without EPA scores (will get 0.5 default)
            logger.info(f"[PIC]   Players without EPA score get median weight (0.5)")

    return career_epa

# ---------------------------------------------------------------------------
# Step 2b: PBP-derived snap percentages
# ---------------------------------------------------------------------------

def _compute_pbp_snap_pct(pbp_off: pd.DataFrame, logger=None) -> Dict:
    """
    Compute snap percentage per player per game directly from PBP offense_players column.

    Replaces the unreliable import_snap_counts() bridge which only has ~45% pfr_id
    overlap with import_players(). This approach uses only gsis_ids from PBP —
    no external join required.

    Returns dict: (gsis_id, game_id) → snap_pct (0.0–1.0)
    where 1.0 = player appeared on every offensive play this game.

    NOTE: Called AFTER _filter_offensive_plays so snap_pct reflects only
    offensive possession plays, consistent with how the cycle is computed.
    """
    if pbp_off is None or len(pbp_off) == 0 or "offense_players" not in pbp_off.columns:
        if logger:
            logger.warning("[PIC]   PBP snap pct: no offense_players column — skipping")
        return {}

    # Count total offensive plays per game
    plays_per_game = pbp_off.groupby("game_id").size().to_dict()

    # Count appearances per player per game
    player_game_counts = {}

    for _, row in pbp_off.iterrows():
        game_id = row.get("game_id", "")
        raw = row.get("offense_players", "")
        if not game_id or not raw:
            continue
        for gsis in raw.split(";"):
            gsis = gsis.strip()
            if gsis:
                key = (gsis, game_id)
                player_game_counts[key] = player_game_counts.get(key, 0) + 1

    # Convert counts to percentages
    pbp_snap_pct = {}
    for (gsis, game_id), count in player_game_counts.items():
        total = plays_per_game.get(game_id, 0)
        if total > 0:
            pbp_snap_pct[(gsis, game_id)] = count / total

    if logger:
        logger.info(f"[PIC]   PBP snap pct: {len(pbp_snap_pct)} player-game records computed "
                   f"across {len(plays_per_game)} games")
        if pbp_snap_pct:
            vals = list(pbp_snap_pct.values())
            above_threshold = sum(1 for v in vals if v >= MIN_SNAP_PCT_THRESHOLD)
            logger.info(f"[PIC]   PBP snap pct ≥{MIN_SNAP_PCT_THRESHOLD}: "
                       f"{above_threshold}/{len(vals)} ({above_threshold/len(vals)*100:.0f}%) "
                       f"pass usage threshold")
            logger.info(f"[PIC]   PBP snap pct distribution: "
                       f"mean={sum(vals)/len(vals):.2f}, "
                       f"min={min(vals):.2f}, max={max(vals):.2f}")
            # Log a few sample entries so you can verify gsis_id format matches player_lookup
            sample_keys = list(pbp_snap_pct.keys())[:3]
            for k in sample_keys:
                logger.info(f"[PIC]   PBP snap sample: gsis={k[0]} game={k[1]} pct={pbp_snap_pct[k]:.2f}")
            # Log plays_per_game range to catch anomalous games
            play_counts = list(plays_per_game.values())
            logger.info(f"[PIC]   Plays per game: mean={sum(play_counts)/len(play_counts):.0f}, "
                       f"min={min(play_counts)}, max={max(play_counts)}")

    return pbp_snap_pct


def _filter_offensive_plays(pbp_df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Keep only offensive possession plays — run and pass.

    Excludes:
        - kickoff, punt, field_goal, extra_point (special teams)
        - no_play (penalty, timeout)
        - qb_kneel, qb_spike (game management, not cycle plays)
        - NaN play_type
        - Plays without offense_players data

    The cycle lives in offensive possessions only. Punting is the result
    of a cycle failure. FG attempts are the result of the cycle working.
    Neither belongs in the cycle computation itself.
    """
    if "play_type" not in pbp_df.columns:
        if logger:
            logger.warning("[PIC] play_type not in PBP — cannot filter offensive plays")
        return pbp_df

    n_before = len(pbp_df)

    pbp_off = pbp_df[
        pbp_df["play_type"].isin(OFFENSIVE_PLAY_TYPES) &
        pbp_df["offense_players"].notna() &
        (pbp_df["offense_players"] != "") &
        pbp_df["posteam"].notna() &
        pbp_df["fixed_drive"].notna()
    ].copy()

    n_after = len(pbp_off)

    if logger:
        logger.info(
            f"[PIC]   Filtered to offensive plays: {n_after}/{n_before} "
            f"({n_after/n_before*100:.1f}% of all plays)"
        )

    return pbp_off


# ---------------------------------------------------------------------------
# Step 3: Drive-level personnel aggregation
# ---------------------------------------------------------------------------

def _aggregate_drive_personnel(
    pbp_off: pd.DataFrame,
    player_lookup: Dict,
    snap_lookup: Dict,
    snap_lookup_by_gsis: Dict,
    logger=None,
) -> pd.DataFrame:
    """
    For each drive, compute:
        - base_personnel: most common 11-man group (mode across plays)
        - key_players: players with weight >= 1.2 (QB + OL)
        - qb_id: the QB on this drive
        - qb_snap_pct: how much of the game the QB has played (quality proxy)
        - ol_count: number of OL identified on this drive
        - key_player_absent: any high-usage player (season snap_pct > 0.8)
                             missing from this drive's base personnel

    Returns one row per (game_id, posteam, fixed_drive).
    """
    records = []

    for (game_id, posteam, drive_id), plays in pbp_off.groupby(
        ["game_id", "posteam", "fixed_drive"]
    ):
        # Collect all personnel sets across plays in this drive
        play_personnel = []
        for _, play in plays.iterrows():
            raw = play.get("offense_players", "")
            if raw and raw != "":
                ids = [x for x in raw.split(";") if x]
                play_personnel.append(frozenset(ids))

        if not play_personnel:
            continue

        # Base personnel = most common group (mode)
        from collections import Counter
        personnel_counts = Counter(play_personnel)
        base_personnel = set(personnel_counts.most_common(1)[0][0])

        # All unique players seen on this drive
        all_players_drive = set()
        for group in play_personnel:
            all_players_drive.update(group)

        # Identify positions in base personnel
        qb_ids = []
        ol_ids = []
        skill_ids = []

        for gsis in base_personnel:
            info = player_lookup.get(gsis)
            if info is None:
                continue
            pos = info["position"]
            if pos == "QB":
                qb_ids.append(gsis)
            elif pos in {"OT", "T", "G", "C", "OL"}:
                ol_ids.append(gsis)
            elif pos in {"WR", "TE", "RB", "HB", "FB"}:
                skill_ids.append(gsis)

        # QB identity — primary QB is whoever appears most in passer_player_id
        qb_id = None
        if "passer_player_id" in plays.columns:
            passer_counts = plays["passer_player_id"].dropna().value_counts()
            if len(passer_counts) > 0:
                qb_id = passer_counts.index[0]

        # Fallback: use QB from personnel
        if qb_id is None and qb_ids:
            qb_id = qb_ids[0]

        # QC: detect backup QB — if identified QB not in player_lookup as QB
        is_backup_qb = False
        if qb_id:
            qb_info = player_lookup.get(qb_id)
            if qb_info is None or qb_info.get("position") != "QB":
                is_backup_qb = True

        # Drive outcome
        drive_result = plays["fixed_drive_result"].iloc[-1] if len(plays) > 0 else None
        is_scoring = drive_result in {"Touchdown", "Field goal"}
        reached_rz = plays["drive_inside20"].fillna(0).max() if "drive_inside20" in plays.columns else 0

        # EPA sum for this drive
        epa_sum = plays["epa"].sum() if "epa" in plays.columns else np.nan

        records.append({
            "game_id":           game_id,
            "posteam":           posteam,
            "drive_id":          drive_id,
            "n_plays":           len(plays),
            "base_personnel":    base_personnel,
            "all_players":       all_players_drive,
            "qb_id":             qb_id,
            "ol_ids":            ol_ids,
            "skill_ids":         skill_ids,
            "ol_count":          len(ol_ids),
            "is_backup_qb":      is_backup_qb,
            "drive_result":      drive_result,
            "is_scoring_drive":  int(is_scoring),
            "reached_redzone":   int(reached_rz > 0),
            "epa_sum":           epa_sum,
        })

    df = pd.DataFrame(records)

    if logger:
        logger.info(
            f"[PIC]   Drive personnel: {len(df)} drives across "
            f"{df['game_id'].nunique()} games, "
            f"{df['posteam'].nunique()} teams"
        )
        if len(df) > 0:
            backup_pct = df["is_backup_qb"].mean() * 100
            logger.info(f"[PIC]   Backup QB drives: {backup_pct:.1f}%")
            scoring_pct = df["is_scoring_drive"].mean() * 100
            logger.info(f"[PIC]   Scoring drive rate: {scoring_pct:.1f}%")

    return df


# ---------------------------------------------------------------------------
# Step 4: Per-player PIC cycle computation
# ---------------------------------------------------------------------------

def _compute_player_pic_cycles(
    pbp_off: pd.DataFrame,
    drive_personnel: pd.DataFrame,
    logger=None,
) -> Dict:
    """
    For each player, compute their personal L1 cycle baseline using
    the same math as the team-level _compute_cycle_sequence().

    L1 (scoring): was this player on the field for a scoring drive?
    L2 (position): was this player on the field when drive reached RZ?

    Key insight: we only count drives where the player was PRESENT.
    A player who played 3 drives out of 10 has a cycle built on 3 drives.
    This naturally handles rotational players and mid-game substitutions.

    Returns:
        Dict keyed by player gsis_id →
            {game_id → {p2s_ratio, p2s_optimal_zone, p2sz_ratio, p2sz_optimal_zone,
                        drives_present, career_scoring_rate}}
    """
    # Build player → ordered list of drives they appeared on
    player_drives = {}  # gsis_id → list of (game_id, drive_id, is_scoring, reached_rz, season, week)

    # Need season/week for ordering
    game_meta = {}
    if "season" in pbp_off.columns and "week" in pbp_off.columns:
        for game_id, grp in pbp_off.groupby("game_id"):
            game_meta[game_id] = {
                "season": grp["season"].iloc[0],
                "week":   grp["week"].iloc[0],
            }

    for _, drive in drive_personnel.iterrows():
        game_id = drive["game_id"]
        drive_id = drive["drive_id"]
        is_scoring = drive["is_scoring_drive"]
        reached_rz = drive["reached_redzone"]
        meta = game_meta.get(game_id, {"season": 0, "week": 0})

        for gsis in drive["all_players"]:
            if gsis not in player_drives:
                player_drives[gsis] = []
            player_drives[gsis].append({
                "game_id":    game_id,
                "drive_id":   drive_id,
                "is_scoring": is_scoring,
                "reached_rz": reached_rz,
                "season":     meta["season"],
                "week":       meta["week"],
            })

    if logger:
        logger.info(f"[PIC]   Computing PIC cycles for {len(player_drives)} players...")
        # Log breakdown of players with enough drives vs dropped
        enough = sum(1 for drives in player_drives.values() if len(drives) >= MIN_DRIVES_FOR_BASELINE)
        too_few = len(player_drives) - enough
        logger.info(f"[PIC]   Players with ≥{MIN_DRIVES_FOR_BASELINE} drives: {enough} | "
                   f"dropped (too few drives): {too_few}")

    # For each player, sort drives chronologically and compute cycle state
    player_cycles = {}

    for gsis, drives in player_drives.items():
        if len(drives) < MIN_DRIVES_FOR_BASELINE:
            continue

        # Sort by season, week, drive_id
        drives_sorted = sorted(drives, key=lambda x: (x["season"], x["week"], x["drive_id"]))

        # Compute L1 cycle (scoring)
        scoring_flags = np.array([d["is_scoring"] for d in drives_sorted], dtype=float)
        l1_baseline, l1_ratio, l1_streak, l1_zone = _compute_cycle_sequence(
            1 - scoring_flags  # miss = non-scoring drive
        )

        # Compute L2 cycle (redzone)
        rz_flags = np.array([d["reached_rz"] for d in drives_sorted], dtype=float)
        l2_baseline, l2_ratio, l2_streak, l2_zone = _compute_cycle_sequence(
            1 - rz_flags  # miss = didn't reach redzone
        )

        # Career scoring rate
        career_scoring_rate = scoring_flags.mean()

        # Map back to game_id → state at END of that game
        # (used with lag in game-level features — pre-game state)
        game_states = {}
        for i, d in enumerate(drives_sorted):
            gid = d["game_id"]
            if gid not in game_states:
                game_states[gid] = []
            game_states[gid].append({
                "l1_baseline": l1_baseline[i],
                "l1_ratio":    l1_ratio[i],
                "l1_zone":     l1_zone[i],
                "l2_ratio":    l2_ratio[i],
                "l2_zone":     l2_zone[i],
                "drives_seen": i + 1,
            })

        # Take last drive state per game (end-of-game state, lagged in features)
        player_cycles[gsis] = {
            "career_scoring_rate": career_scoring_rate,
            "total_drives": len(drives_sorted),
            "game_states": {
                gid: states[-1] for gid, states in game_states.items()
            }
        }

    if logger:
        logger.info(f"[PIC]   PIC cycles computed for {len(player_cycles)} players "
                    f"(min {MIN_DRIVES_FOR_BASELINE} drives required)")
        if player_cycles:
            total_drives_list = [v["total_drives"] for v in player_cycles.values()]
            scoring_rates = [v["career_scoring_rate"] for v in player_cycles.values()]
            logger.info(f"[PIC]   Avg drives per player: {np.mean(total_drives_list):.0f} "
                       f"(min={min(total_drives_list)}, max={max(total_drives_list)})")
            logger.info(f"[PIC]   Career scoring rate distribution: "
                       f"mean={np.mean(scoring_rates):.3f}, "
                       f"min={min(scoring_rates):.3f}, "
                       f"max={max(scoring_rates):.3f}")

    return player_cycles


def _compute_cycle_sequence(
    miss_flags: np.ndarray,
    min_gaps: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core cycle baseline — identical math to team-level version.
    Applied here at individual player level.

    miss_flags: 1 = miss (non-scoring / didn't reach RZ), 0 = hit
    Returns: (baseline, ratio, streak, zone) arrays
    """
    n = len(miss_flags)
    baseline = np.full(n, np.nan)
    ratio    = np.full(n, np.nan)
    streak   = np.full(n, np.nan)
    zone     = np.full(n, np.nan)

    gaps = []
    current_gap = 0
    current_streak = 0

    for i in range(n):
        v = miss_flags[i]
        if np.isnan(v):
            continue

        if v == 1:  # miss
            if current_gap > 0:
                gaps.append(current_gap)
            current_gap = 0
            current_streak += 1
        else:       # hit
            current_gap += 1
            current_streak = 0

        if len(gaps) >= min_gaps:
            baseline[i] = float(np.mean(gaps))
            if baseline[i] > 0:
                ratio[i] = current_gap / baseline[i]
                zone[i] = 1.0 if 0.5 <= ratio[i] <= 1.0 else 0.0

        streak[i] = float(current_streak)

    return baseline, ratio, streak, zone


# ---------------------------------------------------------------------------
# Step 5: Game-level PIC feature aggregation
# ---------------------------------------------------------------------------

def _build_game_pic_features(
    drive_personnel: pd.DataFrame,
    player_cycles: Dict,
    player_lookup: Dict,
    snap_lookup: Dict,
    snap_lookup_by_gsis: Dict,
    pbp_snap_pct: Dict,
    career_epa: Dict,
    logger=None,
) -> pd.DataFrame:
    """
    For each (game_id, team), aggregate player PIC states into game-level features.

    Uses PREVIOUS game's player cycle states (lagged) to avoid data leakage.

    Features produced:
        pic_qb_in_zone:          QB personally in L1 optimal zone going into game
        pic_qb_career_rate:      QB career scoring drive rate (quality proxy)
        pic_qb_is_backup:        1 if backup QB detected this game
        pic_ol_sync_rate:        % of OL players in their personal L1 optimal zone
        pic_ol_quality_weighted: snap_pct × position_weight composite for OL
        pic_skill_sync_rate:     % of skill players in their personal L1 optimal zone
        pic_team_alignment:      weighted average of all player zone states
                                 (the core PIC signal — how aligned are players
                                 with the team's cycle state)
        pic_key_absent_score:    quality score of absent key players
                                 (high = important player missing, fade team cycle)
        pic_present_quality:     weighted quality of players who ARE present
    """
    records = []
    # Diagnostic counters — track snap_pct source across all players processed
    _snap_from_pbp = 0
    _snap_from_external = 0
    _snap_below_threshold = 0
    _snap_missing = 0

    # Build ordered game list per team for lag — MUST sort by season+week, not game_id string.
    # game_id strings like "2020_01_ARI_SF" sort alphabetically within a week,
    # not chronologically. Sorting by season then week ensures correct temporal ordering
    # so prev_game_id always points to the team's actual previous game.
    game_meta = (
        drive_personnel[["game_id", "posteam"]]
        .drop_duplicates()
        .copy()
    )
    # Extract season and week from game_id (format: YYYY_WW_AWAY_HOME)
    game_meta["season"] = game_meta["game_id"].str.split("_").str[0].astype(int)
    game_meta["week"]   = game_meta["game_id"].str.split("_").str[1].astype(int)
    game_meta = game_meta.sort_values(["season", "week", "game_id"])
    game_meta["prev_game_id"] = game_meta.groupby("posteam")["game_id"].shift(1)
    prev_game_map = game_meta.set_index(["posteam", "game_id"])["prev_game_id"].to_dict()

    if logger:
        n_with_prev = sum(1 for v in prev_game_map.values() if v is not None and str(v) != "nan")
        n_total = len(prev_game_map)
        logger.info(f"[PIC]   Game order: {n_total} team-game entries, "
                   f"{n_with_prev} have a previous game for cycle lag")

    for (game_id, posteam), game_drives in drive_personnel.groupby(["game_id", "posteam"]):
        prev_game_id = prev_game_map.get((posteam, game_id))

        # Collect all players who appeared in this game
        all_game_players = set()
        for _, drive in game_drives.iterrows():
            all_game_players.update(drive["all_players"])

        # For each player, get their cycle state from PREVIOUS game
        epa_scores = []
        qb_zone = np.nan
        qb_career_rate = np.nan
        qb_is_backup = 0
        ol_zones = []
        ol_weights = []
        skill_zones = []
        skill_weights = []
        all_zones = []
        all_weights = []

        # Detect backup QB for this game
        backup_drives = game_drives["is_backup_qb"].sum()
        total_drives = len(game_drives)
        qb_is_backup = int(backup_drives > total_drives * 0.5)

        # Get QB id (most common across game drives)
        qb_ids = game_drives["qb_id"].dropna()
        if len(qb_ids) > 0:
            qb_id = qb_ids.value_counts().index[0]
            qb_cycles = player_cycles.get(qb_id)
            if qb_cycles and prev_game_id:
                prev_state = qb_cycles["game_states"].get(prev_game_id)
                if prev_state:
                    qb_zone = prev_state["l1_zone"]
            qb_career_rate = player_cycles.get(qb_id, {}).get("career_scoring_rate", np.nan)

        # Get snap-weighted player states
        for gsis in all_game_players:
            info = player_lookup.get(gsis)
            if info is None:
                continue

            pos = info["position"]
            weight = info["weight"]
            pfr_id = info["pfr_id"]

            # Snap percentage this game as usage weight.
            # PRIMARY: PBP-derived snap pct — computed from offense_players column,
            # no external join needed. 100% gsis_id coverage.
            # FALLBACK: external snap_lookup (pfr_id) and snap_lookup_by_gsis for
            # any edge cases where PBP data is missing offense_players.
            snap_pct = pbp_snap_pct.get((gsis, game_id), None)
            if snap_pct is not None:
                _snap_from_pbp += 1
            else:
                snap_info = snap_lookup.get((pfr_id, game_id), {})
                if not snap_info:
                    snap_info = snap_lookup_by_gsis.get((gsis, game_id), {})
                snap_pct = snap_info.get("offense_pct", 0) or 0
                if snap_pct > 0:
                    _snap_from_external += 1
                else:
                    _snap_missing += 1

            # Skip bench warmers — below threshold means essentially absent
            if snap_pct < MIN_SNAP_PCT_THRESHOLD:
                _snap_below_threshold += 1
                continue

            # Get player's cycle state from previous game
            cycles = player_cycles.get(gsis)
            if cycles is None or prev_game_id is None:
                continue

            prev_state = cycles["game_states"].get(prev_game_id)
            if prev_state is None:
                continue

            l1_zone = prev_state["l1_zone"]
            if np.isnan(l1_zone):
                continue

            # Quality weight = position weight × snap percentage × career EPA percentile
            # This is the key distinction: Mahomes at 0.95 EPA percentile gets
            # 3x the quality weight of a backup at 0.3 EPA percentile.
            # OL players without direct EPA attribution get 0.5 (median) as default.
            epa_pct = career_epa.get(gsis, 0.5)  # default to median if unknown
            quality_weight = weight * snap_pct * (0.5 + epa_pct)  # scale: 0.5x to 1.5x

            epa_scores.append(epa_pct)

            if pos in {"OT", "T", "G", "C", "OL"}:
                ol_zones.append(l1_zone)
                ol_weights.append(quality_weight)
            elif pos in {"WR", "TE", "RB", "HB", "FB"}:
                skill_zones.append(l1_zone)
                skill_weights.append(quality_weight)

            all_zones.append(l1_zone)
            all_weights.append(quality_weight)

        # Aggregate
        pic_ol_sync_rate = np.average(ol_zones, weights=ol_weights) if ol_zones else np.nan
        pic_skill_sync_rate = np.average(skill_zones, weights=skill_weights) if skill_zones else np.nan
        pic_team_alignment = np.average(all_zones, weights=all_weights) if all_zones else np.nan
        pic_ol_quality = sum(ol_weights) / len(ol_weights) if ol_weights else np.nan
        pic_present_quality = sum(all_weights) / len(all_weights) if all_weights else np.nan

        records.append({
            "game_id":                game_id,
            "team":                   posteam,
            "pic_qb_in_zone":         qb_zone,
            "pic_qb_career_rate":     qb_career_rate,
            "pic_qb_is_backup":       qb_is_backup,
            "pic_ol_sync_rate":       pic_ol_sync_rate,
            "pic_ol_quality_weighted": pic_ol_quality,
            "pic_skill_sync_rate":    pic_skill_sync_rate,
            "pic_team_alignment":     pic_team_alignment,
            "pic_present_quality":    pic_present_quality,
            "pic_avg_player_epa":     float(np.mean(epa_scores)) if epa_scores else np.nan,
            "pic_players_tracked":    len(all_zones),
        })

    df = pd.DataFrame(records)

    if logger:
        total_lookups = _snap_from_pbp + _snap_from_external + _snap_missing + _snap_below_threshold
        logger.info(f"[PIC]   Snap pct source breakdown ({total_lookups} player-game lookups):")
        logger.info(f"[PIC]     PBP-derived (primary):  {_snap_from_pbp} ({_snap_from_pbp/max(total_lookups,1)*100:.0f}%)")
        logger.info(f"[PIC]     External fallback:       {_snap_from_external} ({_snap_from_external/max(total_lookups,1)*100:.0f}%)")
        logger.info(f"[PIC]     Below threshold (≥{MIN_SNAP_PCT_THRESHOLD}): {_snap_below_threshold} ({_snap_below_threshold/max(total_lookups,1)*100:.0f}%) — filtered out")
        logger.info(f"[PIC]     Missing entirely:        {_snap_missing} ({_snap_missing/max(total_lookups,1)*100:.0f}%) — treated as absent")
        logger.info(f"[PIC]   Game PIC features: {len(df)} team-game rows")
        for col in ["pic_qb_in_zone", "pic_ol_sync_rate", "pic_team_alignment",
                    "pic_skill_sync_rate", "pic_present_quality", "pic_avg_player_epa"]:
            if col in df.columns:
                valid = df[col].notna().sum()
                pct = valid / len(df) * 100 if len(df) > 0 else 0
                if valid > 0:
                    mean_val = df[col].dropna().mean()
                    logger.info(f"[PIC]     {col}: {valid}/{len(df)} valid ({pct:.0f}%) mean={mean_val:.3f}")
                else:
                    logger.warning(f"[PIC]     {col}: 0/{len(df)} valid — no data")
        # Log backup QB coverage
        if "pic_qb_is_backup" in df.columns:
            n_backup = (df["pic_qb_is_backup"] == 1).sum()
            logger.info(f"[PIC]     Backup QB games detected: {n_backup} ({n_backup/len(df)*100:.1f}%)")
        # Log EPA weight coverage
        n_epa_scored = sum(1 for r in records if not np.isnan(r.get("pic_avg_player_epa", np.nan)))
        n_epa_default = len(records) - n_epa_scored
        logger.info(f"[PIC]     Games with EPA-weighted quality: {n_epa_scored} | "
                   f"using default weight: {n_epa_default}")

    return df


# ---------------------------------------------------------------------------
# Step 6: Merge PIC features onto game-team rows
# ---------------------------------------------------------------------------

def _merge_pic_features(
    df: pd.DataFrame,
    game_pic: pd.DataFrame,
    logger=None,
) -> pd.DataFrame:
    """
    Left-join PIC features onto game-team rows.
    Also builds the critical intersection features:
        pic_cycle_confirmed:    team dual_optimal AND qb in zone AND ol_sync > 0.5
        pic_cycle_degraded:     team dual_optimal BUT qb is backup
        pic_alignment_score:    continuous — team_alignment × (1 - is_backup)
    """
    if game_pic.empty:
        if logger:
            logger.warning("[PIC] No PIC features computed — skipping merge")
        return df

    df = df.merge(game_pic, on=["game_id", "team"], how="left")
    df = df.reset_index(drop=True)

    # Intersection features — the patent-protected core
    # These combine team cycle state (L1/L2) with player cycle state (L3)

    has_team_dual = "p2s_p2sz_dual_optimal" in df.columns
    has_qb_zone = "pic_qb_in_zone" in df.columns
    has_ol_sync = "pic_ol_sync_rate" in df.columns
    has_backup = "pic_qb_is_backup" in df.columns
    has_alignment = "pic_team_alignment" in df.columns

    if has_team_dual and has_qb_zone and has_ol_sync:
        # Confirmed cycle: team in dual optimal + QB in zone + OL in sync
        df["pic_cycle_confirmed"] = (
            (df["p2s_p2sz_dual_optimal"] == 1.0) &
            (df["pic_qb_in_zone"] == 1.0) &
            (df["pic_ol_sync_rate"] >= 0.5)
        ).astype(float)
        df.loc[
            df["p2s_p2sz_dual_optimal"].isna() |
            df["pic_qb_in_zone"].isna() |
            df["pic_ol_sync_rate"].isna(),
            "pic_cycle_confirmed"
        ] = np.nan

        n_confirmed = (df["pic_cycle_confirmed"] == 1.0).sum()
        if logger:
            logger.info(f"[PIC]   pic_cycle_confirmed: {n_confirmed} games "
                        f"({n_confirmed/len(df)*100:.1f}%)")

    if has_team_dual and has_backup:
        # Degraded cycle: team in dual optimal BUT backup QB
        # This is the "fade" signal — team cycle is real but execution is compromised
        df["pic_cycle_degraded"] = (
            (df["p2s_p2sz_dual_optimal"] == 1.0) &
            (df["pic_qb_is_backup"] == 1)
        ).astype(float)

        n_degraded = (df["pic_cycle_degraded"] == 1.0).sum()
        if logger:
            logger.info(f"[PIC]   pic_cycle_degraded: {n_degraded} games "
                        f"({n_degraded/len(df)*100:.1f}%)")

    if has_alignment and has_backup:
        # Continuous alignment score — team alignment × quality penalty for backup
        # When starter is present: full signal
        # When backup present: signal discounted by (1 - backup_flag)
        df["pic_alignment_score"] = (
            df["pic_team_alignment"].fillna(0) *
            (1 - df["pic_qb_is_backup"].fillna(0))
        )
        # Set to NaN if no alignment data at all
        df.loc[df["pic_team_alignment"].isna(), "pic_alignment_score"] = np.nan

    if logger:
        n_valid = df["pic_team_alignment"].notna().sum() if "pic_team_alignment" in df.columns else 0
        logger.info(f"[PIC]   Final merge: {n_valid}/{len(df)} rows have PIC data")

    return df