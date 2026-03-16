"""
Q1 Cycle Confirmation Analysis
================================
Tests the hypothesis: when pre-game cycle signal (p2s_ratio, dual optimal)
is CONFIRMED by Q1 drive behavior, does the over hit at a higher rate?

This is the foundation for Layer 4 PIM (Player Impact Momentum).

Run this as a standalone analysis against the training parquet after a
full feature-engineering pass. It does NOT modify the model — it tells
you whether the Q1 signal is strong enough to ADD as a live feature.

Usage:
    python q1_cycle_analysis.py

Input:
    - Needs access to PBP data (2020-2022 or wider) and the feature matrix
    - Run after a training pass so features are computed
"""

import pandas as pd
import numpy as np


# ─── HYPOTHESIS ───────────────────────────────────────────────────────────────
#
# Pre-game cycle prediction (L1+L2) fires on ~12% of games (dual optimal).
# Those games hit over at 43.3% — actually LOWER than baseline (46.4%).
# Something is inverting the signal.
#
# Q1 confirmation hypothesis:
#   1. A team in dual optimal pre-game that ALSO shows Q1 cycle behavior
#      (efficient drives, scoring or near-scoring, QB in rhythm)
#      should hit over at 55%+.
#
#   2. A team showing Q1 cycle CONTRADICTION (3-and-outs, no EPA)
#      should hit under at 55%+.
#
#   3. The RATIO of Q1 points to closing_total is a numeric signal:
#      - Q1_pct = q1_combined_score / closing_total
#      - Historically: what Q1_pct thresholds predict over vs under?
#      - Hypothesis: Q1_pct > 0.20 → strong over signal
#                    Q1_pct < 0.08 → strong under signal


# ─── Q1 METRICS TO COMPUTE FROM PBP ─────────────────────────────────────────

def compute_q1_metrics(pbp_df, games_df):
    """
    For each game, compute Q1 drive metrics:
    
    - q1_drives_team: number of Q1 drives per team
    - q1_scoring_drives: scoring drives in Q1
    - q1_points: actual Q1 points scored by each team
    - q1_combined: total Q1 points in game
    - q1_pct_of_total: q1_combined / combined_final_score
    - q1_pct_of_closing: q1_combined / closing_total
    - q1_epa_per_drive: avg EPA per Q1 drive (efficiency without scoring)
    - q1_yards_per_drive: avg yards per Q1 drive
    - q1_rz_reached: did either team reach redzone in Q1?
    - q1_cycle_confirmed: pre-game cycle prediction AND Q1 shows positive EPA
    
    The key metric is q1_pct_of_closing — what fraction of the
    closing total was scored in Q1. If teams are in cycle, Q1 should
    account for more of the total.
    
    Historical baseline to measure against:
    - Average Q1 contributes ~22-25% of a typical NFL game total
    - Games where Q1 > 28% of closing total → over hits at ?%
    - Games where Q1 < 15% of closing total → under hits at ?%
    """
    
    q1_plays = pbp_df[pbp_df['qtr'] == 1].copy()
    
    # Q1 points per game (sum of scoring plays in Q1)
    if 'total_home_score' in q1_plays.columns:
        # Use last play of Q1 per game to get end-of-Q1 score
        q1_end = q1_plays.groupby('game_id').last()[['total_home_score', 'total_away_score']].reset_index()
        q1_end['q1_combined'] = q1_end['total_home_score'] + q1_end['total_away_score']
    else:
        q1_end = None
    
    # Q1 EPA per game (efficiency signal even without scoring)
    q1_epa = q1_plays.groupby(['game_id', 'posteam'])['epa'].mean().reset_index()
    q1_epa.columns = ['game_id', 'team', 'q1_epa_per_play']
    
    # Q1 drive count and scoring
    q1_drives = q1_plays.groupby(['game_id', 'posteam', 'fixed_drive']).agg(
        q1_drive_yards=('yards_gained', 'sum'),
        q1_drive_epa=('epa', 'sum'),
        q1_drive_result=('fixed_drive_result', 'last'),
    ).reset_index()
    q1_drives['q1_scoring'] = q1_drives['q1_drive_result'].isin(['Touchdown', 'Field goal']).astype(int)
    
    q1_summary = q1_drives.groupby(['game_id', 'posteam']).agg(
        q1_drives_n=('fixed_drive', 'count'),
        q1_scoring_drives=('q1_scoring', 'sum'),
        q1_yards_per_drive=('q1_drive_yards', 'mean'),
        q1_epa_per_drive=('q1_drive_epa', 'mean'),
    ).reset_index()
    
    return q1_end, q1_summary


def analyze_q1_over_correlation(pbp_df, games_df, closing_lines_df):
    """
    Core analysis: bucket games by Q1 scoring percentage of closing total,
    then measure over rate in each bucket.
    
    Expected output:
    
    Q1_pct_bucket  | games | over_rate | vs_baseline
    ─────────────────────────────────────────────────
    <10%           |  xxx  |   35%     |   -11pts
    10-15%         |  xxx  |   41%     |   -5pts
    15-20%         |  xxx  |   46%     |   baseline
    20-25%         |  xxx  |   52%     |   +6pts  ← signal
    25-30%         |  xxx  |   58%     |   +12pts ← strong
    >30%           |  xxx  |   65%     |   +19pts ← very strong
    
    If this pattern holds, Q1_pct becomes a live feature that, combined
    with pre-game cycle signal, creates the PIM layer.
    """
    # Placeholder — run against actual data
    # Requires: pbp with Q1 plays, games with final scores, closing totals
    pass


# ─── INTEGRATION POINT ───────────────────────────────────────────────────────
#
# How this plugs into the model as PIM features:
#
# PRE-GAME (static, computed at training/inference time):
#   pim_q1_cycle_primed = 1 if team in dual optimal AND p2s_ratio > 0.7
#
# LIVE (computed after Q1 ends, for in-game betting):
#   pim_q1_pct_of_closing   = q1_combined / closing_total
#   pim_q1_epa_per_drive    = avg EPA per Q1 drive (both teams)
#   pim_q1_confirmed        = pim_q1_cycle_primed AND q1_epa_per_drive > 0
#   pim_q1_contradicted     = pim_q1_cycle_primed AND q1_epa_per_drive < -0.5
#
# For static model (training on historical data):
#   - Compute Q1 metrics from historical PBP
#   - Include as features — model learns the correlation
#   - This is supervised learning on a leaky feature ONLY if you ensure
#     test set uses Q1 from the game being predicted (which is valid for
#     live betting, but NOT for pre-game prediction)
#
# ⚠ IMPORTANT: Q1 features are LIVE features, not pre-game features.
#   They belong in a SECOND model tier: pre-game model fires a prediction,
#   then after Q1 ends the live model either confirms or revises it.
#   This is the two-tier architecture: PIC (pre-game) + PIM (live).

print("Q1 Cycle Analysis framework loaded.")
print("Next steps:")
print("  1. Run compute_q1_metrics() against 2020-2022 PBP")
print("  2. Run analyze_q1_over_correlation() to find the threshold buckets")
print("  3. If pattern holds: add q1_pct_of_closing as PIM feature")
print("  4. This becomes the live betting tier of the platform")