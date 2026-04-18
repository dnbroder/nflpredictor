"""
Backtester -- Rolling walk-forward validation vs. closing line.

FIXES vs original:
  - Imports TARGET_COLUMNS dict (was undefined)
  - _compute_clv uses correct target key check ("over_total" not "total")
  - engineer_features call passes target= param
"""

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from nfl_predictor.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS

logger = logging.getLogger("nflpredictor")

BREAK_EVEN = 0.5238


def run_backtest(featured_data, target="over_total", train_seasons=4, test_seasons=1,
                 min_confidence=0.0, bet_fraction=1.0, random_state=42):
    from nfl_predictor.model import train_model

    target_col = TARGET_COLUMNS[target]
    available = [f for f in FEATURE_COLUMNS if f in featured_data.columns]

    logger.info("[BT] ============================================================")
    logger.info(f"[BT] BACKTEST -- target={target}, train_window={train_seasons}s, min_conf={min_confidence}")
    logger.info("[BT] ============================================================")

    if "season" not in featured_data.columns:
        logger.error("[BT] No 'season' column")
        return {}

    seasons = sorted(featured_data["season"].unique())
    logger.info(f"[BT] Available seasons: {seasons}")

    window_results = []
    all_bets = []

    for i in range(train_seasons, len(seasons) - test_seasons + 1):
        train_range = seasons[i - train_seasons:i]
        test_range = seasons[i:i + test_seasons]

        train_data = featured_data[
            featured_data["season"].isin(train_range) & featured_data[target_col].notna()
        ].copy()
        test_data = featured_data[
            featured_data["season"].isin(test_range) & featured_data[target_col].notna()
        ].copy()

        if len(train_data) < 100 or len(test_data) < 10:
            logger.warning(f"[BT] Skipping window train={train_range}, test={test_range}: insufficient data")
            continue

        logger.info(f"[BT] Window: train={train_range}, test={test_range} "
                    f"({len(train_data)} train, {len(test_data)} test)")

        try:
            model, train_metrics = train_model(train_data, target=target, calibrate=True,
                                               random_state=random_state)
        except Exception as e:
            logger.error(f"[BT] Training failed: {e}")
            continue

        X_test = test_data[available].copy()
        y_test = test_data[target_col].values
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        clv_values = _compute_clv(test_data, y_proba, target)
        bets = _simulate_bets(test_data, y_proba, y_pred, y_test,
                              clv_values, target, min_confidence, bet_fraction)
        all_bets.extend(bets)

        window_metrics = _compute_window_metrics(y_test, y_pred, y_proba, bets,
                                                  train_range, test_range)
        window_results.append(window_metrics)

        logger.info(f"[BT] Window {test_range}: ROI={window_metrics['roi_pct']:+.1f}%, "
                    f"AUC={window_metrics['auc']:.4f}, Acc={window_metrics['accuracy']:.4f} "
                    f"({window_metrics['bets_placed']} bets)")

    summary = _compute_aggregate(all_bets, window_results, target)

    logger.info("[BT] ============================================================")
    logger.info(f"[BT] AGGREGATE -- target={target}")
    logger.info(f"[BT]   Total bets: {summary['total_bets']}")
    logger.info(f"[BT]   Win rate:   {summary['win_rate']:.4f} (break-even: {BREAK_EVEN:.4f})")
    logger.info(f"[BT]   ROI:        {summary['overall_roi_pct']:+.2f}%")
    logger.info(f"[BT]   CLV (avg):  {summary.get('avg_clv', 0):+.4f}")
    logger.info(f"[BT]   AUC (avg):  {summary['avg_auc']:.4f}")
    logger.info(f"[BT]   Brier:      {summary['avg_brier']:.4f}")

    for tier, data in summary.get("confidence_tiers", {}).items():
        if data["n"] >= 10:
            logger.info(f"[BT]     P>={tier}: {data['win_rate']:.4f} win rate, "
                        f"{data['roi_pct']:+.1f}% ROI ({data['n']} bets)")
    logger.info("[BT] ============================================================")

    return {"windows": window_results, "summary": summary, "bets": all_bets,
            "target": target, "min_confidence": min_confidence}


def _compute_clv(test_data, y_proba, target):
    clv = np.zeros(len(test_data))
    if target == "over_total" and "total_movement" in test_data.columns:
        line_mov = test_data["total_movement"].fillna(0).values
        for i, (prob, mov) in enumerate(zip(y_proba, line_mov)):
            if prob > BREAK_EVEN:
                clv[i] = prob - (0.5 + mov * 0.02)
            else:
                clv[i] = (1 - prob) - (0.5 - mov * 0.02)
    return clv


def _simulate_bets(test_data, y_proba, y_pred, y_actual, clv_values,
                   target, min_confidence, bet_fraction):
    bets = []
    for i in range(len(y_proba)):
        prob = y_proba[i]
        confidence = max(prob, 1 - prob)
        if confidence < max(min_confidence, BREAK_EVEN):
            continue

        bet_on_positive = prob >= 0.5
        outcome = bool(y_actual[i])
        won = (bet_on_positive == outcome)
        stake = 1.0 * bet_fraction
        profit = stake * (100 / 110) if won else -stake

        bets.append({
            "index": i, "predicted_prob": round(float(prob), 4),
            "confidence": round(float(confidence), 4),
            "bet_on": "over/home" if bet_on_positive else "under/away",
            "won": bool(won), "profit": round(profit, 4),
            "clv": round(float(clv_values[i]), 4) if clv_values is not None else 0.0,
            "season": test_data["season"].iloc[i] if "season" in test_data.columns else None,
            "game_id": test_data["game_id"].iloc[i] if "game_id" in test_data.columns else None,
        })
    return bets


def _compute_window_metrics(y_test, y_pred, y_proba, bets, train_seasons, test_seasons):
    placed = list(bets)
    wins = [b for b in placed if b["won"]]
    total_profit = sum(b["profit"] for b in placed)
    total_staked = len(placed)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0

    return {
        "train_seasons": train_seasons, "test_seasons": test_seasons,
        "total_games": len(y_test), "bets_placed": len(placed),
        "wins": len(wins),
        "win_rate": len(wins) / len(placed) if placed else 0.0,
        "roi_pct": round(roi, 2), "total_profit_units": round(total_profit, 2),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "avg_clv": round(float(np.mean([b["clv"] for b in placed])), 4) if placed else 0.0,
    }


def _compute_aggregate(all_bets, window_results, target):
    if not all_bets:
        return {"total_bets": 0, "win_rate": 0, "overall_roi_pct": 0,
                "avg_auc": 0, "avg_brier": 0, "avg_clv": 0, "confidence_tiers": {}}

    total_profit = sum(b["profit"] for b in all_bets)
    wins = sum(1 for b in all_bets if b["won"])

    confidence_tiers = {}
    for thresh in [0.52, 0.53, 0.55, 0.57, 0.60, 0.65]:
        tier_bets = [b for b in all_bets if b["confidence"] >= thresh]
        if tier_bets:
            tier_wins = sum(1 for b in tier_bets if b["won"])
            tier_profit = sum(b["profit"] for b in tier_bets)
            confidence_tiers[str(thresh)] = {
                "n": len(tier_bets), "wins": tier_wins,
                "win_rate": round(tier_wins / len(tier_bets), 4),
                "roi_pct": round(tier_profit / len(tier_bets) * 100, 2),
            }

    win_rate = wins / len(all_bets)
    roi_pct = total_profit / len(all_bets) * 100
    benchmarks = {
        "spread":     {"break_even": BREAK_EVEN, "best_public": 0.53},
        "over_total": {"break_even": BREAK_EVEN, "best_public": 0.535},
        "moneyline":  {"break_even": BREAK_EVEN, "best_public": 0.67},
        "team_total_over": {"break_even": BREAK_EVEN, "best_public": 0.54},
    }
    bench = benchmarks.get(target, {})

    return {
        "total_bets": len(all_bets), "wins": wins,
        "win_rate": round(win_rate, 4), "overall_roi_pct": round(roi_pct, 2),
        "vs_break_even": round(win_rate - BREAK_EVEN, 4),
        "vs_best_public": round(win_rate - bench.get("best_public", BREAK_EVEN), 4),
        "avg_auc": round(float(np.mean([w.get("auc", 0.5) for w in window_results])), 4),
        "avg_brier": round(float(np.mean([w.get("brier", 0.25) for w in window_results])), 4),
        "avg_clv": round(float(np.mean([b["clv"] for b in all_bets])), 4),
        "confidence_tiers": confidence_tiers,
        "target": target, "benchmarks": bench,
    }


if __name__ == "__main__":
    import argparse, json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
    parser = argparse.ArgumentParser(description="Run NFL model backtest")
    parser.add_argument("--target", default="over_total", choices=list(TARGET_COLUMNS.keys()))
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2016, 2025)))
    parser.add_argument("--train-window", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--no-pic", action="store_true", help="Skip Layer 3 PIC (faster, L1+L2 only)")
    args = parser.parse_args()

    from nfl_predictor.nfl_data import load_games, load_drives, load_plays, load_players
    from nfl_predictor.feature_engineering import engineer_features

    logger.info("Loading data...")
    games = load_games(args.seasons)
    drives = load_drives(args.seasons)

    # Load PBP + players for Layer 3 PIC (unless --no-pic)
    pbp = None
    players = None
    if not args.no_pic:
        logger.info("Loading PBP for Layer 3 PIC...")
        pbp = load_plays(args.seasons)
        players = load_players()
        logger.info(f"PBP: {len(pbp)} plays, Players: {len(players)} records")
    else:
        logger.info("Skipping Layer 3 PIC (--no-pic flag)")

    featured = engineer_features(games, drives, pbp_df=pbp, players_df=players)

    results = run_backtest(featured, target=args.target, train_seasons=args.train_window,
                           min_confidence=args.min_confidence)

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    summary = results.get("summary", {})
    for k, v in summary.items():
        if k not in ("confidence_tiers", "benchmarks"):
            print(f"  {k}: {v}")
    print("\nConfidence Tiers:")
    for tier, data in summary.get("confidence_tiers", {}).items():
        if data["n"] >= 10:
            print(f"  P>={tier}: win={data['win_rate']:.4f}, ROI={data['roi_pct']:+.1f}%, n={data['n']}")