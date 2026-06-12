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

    # Pushes (game total lands exactly on the line) are NO-ACTION: excluded from
    # training, but kept in the TEST set so they score as refunds, not as wins/losses.
    has_push = "total_margin" in featured_data.columns

    for i in range(train_seasons, len(seasons) - test_seasons + 1):
        train_range = seasons[i - train_seasons:i]
        test_range = seasons[i:i + test_seasons]

        train_data = featured_data[
            featured_data["season"].isin(train_range) & featured_data[target_col].notna()
        ].copy()
        if has_push:
            test_data = featured_data[
                featured_data["season"].isin(test_range) & featured_data["total_margin"].notna()
            ].copy()
        else:
            test_data = featured_data[
                featured_data["season"].isin(test_range) & featured_data[target_col].notna()
            ].copy()

        if len(train_data) < 100 or len(test_data) < 10:
            logger.warning(f"[BT] Skipping window train={train_range}, test={test_range}: insufficient data")
            continue

        logger.info(f"[BT] Window: train={train_range}, test={test_range} "
                    f"({len(train_data)} train, {len(test_data)} test)")

        try:
            # holdout=0.0: train on the ENTIRE training window. The old default did an
            # internal random 80/20 split and trained on only 80%, wasting a fifth of
            # each fold's data (the real OOS test is the held-out future season below).
            model_result, train_metrics = train_model(train_data, holdout=0.0,
                                                       random_state=random_state)
        except Exception as e:
            logger.error(f"[BT] Training failed: {e}")
            continue

        X_test = test_data[available].copy()
        y_test = test_data[target_col].values            # binary over/under; NaN on pushes
        margin = test_data["total_margin"].values if has_push else None

        # Support both two-stage (dict with logistic_model) and legacy XGBoost
        if isinstance(model_result, dict) and "logistic_model" in model_result:
            # Two-stage model: use calibrated probabilities when available
            lr_features = model_result["logistic_features"]
            lr_scaler = model_result["scaler"]
            medians = model_result.get("train_medians", {})
            X_lr = test_data[[f for f in lr_features if f in test_data.columns]].copy()
            for col in X_lr.columns:
                if col in medians:
                    X_lr[col] = X_lr[col].fillna(medians[col])
            X_lr_scaled = lr_scaler.transform(X_lr)
            # Prefer calibrated model (Platt scaling) over raw logistic
            if "calibrated_model" in model_result:
                y_proba = model_result["calibrated_model"].predict_proba(X_lr_scaled)[:, 1]
            else:
                y_proba = model_result["logistic_model"].predict_proba(X_lr_scaled)[:, 1]
        else:
            # Legacy XGBoost
            y_proba = model_result.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= 0.5).astype(int)

        clv_values = _compute_clv(test_data, y_proba, target)
        bets = _simulate_bets(test_data, y_proba, y_pred, y_test,
                              clv_values, target, min_confidence, bet_fraction,
                              margin=margin)
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
    logger.info(f"[BT]   Record:     {summary.get('wins',0)}-{summary.get('losses',0)}-"
                f"{summary.get('pushes',0)} (W-L-P, pushes refunded/no-action)")
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
                   target, min_confidence, bet_fraction, margin=None):
    bets = []
    for i in range(len(y_proba)):
        prob = y_proba[i]
        confidence = max(prob, 1 - prob)
        if confidence < max(min_confidence, BREAK_EVEN):
            continue

        bet_on_positive = prob >= 0.5
        stake = 1.0 * bet_fraction

        # Push handling: if the game landed exactly on the line, the bet is refunded
        # (no-action) — zero profit, and excluded from win-rate / ROI denominators.
        if margin is not None and margin[i] == 0:
            push = True
            won = None
            profit = 0.0
        else:
            push = False
            # Outcome from the true signed margin when available, else the binary label.
            if margin is not None:
                outcome = bool(margin[i] > 0)
            else:
                outcome = bool(y_actual[i])
            won = (bet_on_positive == outcome)
            profit = stake * (100 / 110) if won else -stake

        bets.append({
            "index": i, "predicted_prob": round(float(prob), 4),
            "confidence": round(float(confidence), 4),
            "bet_on": "over/home" if bet_on_positive else "under/away",
            "won": won, "push": push, "profit": round(profit, 4),
            "clv": round(float(clv_values[i]), 4) if clv_values is not None else 0.0,
            "season": test_data["season"].iloc[i] if "season" in test_data.columns else None,
            "game_id": test_data["game_id"].iloc[i] if "game_id" in test_data.columns else None,
        })
    return bets


def _compute_window_metrics(y_test, y_pred, y_proba, bets, train_seasons, test_seasons):
    placed = list(bets)
    decided = [b for b in placed if not b.get("push")]      # pushes are no-action
    wins = [b for b in decided if b["won"]]
    pushes = [b for b in placed if b.get("push")]
    total_profit = sum(b["profit"] for b in placed)         # pushes contribute 0
    total_staked = len(decided)                             # pushes risk nothing
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0

    # AUC/accuracy on DECIDED games only (drop NaN labels from pushes)
    yt = np.asarray(y_test, dtype=float)
    valid = ~np.isnan(yt)
    yv, pv, prv = yt[valid].astype(int), np.asarray(y_pred)[valid], np.asarray(y_proba)[valid]

    return {
        "train_seasons": train_seasons, "test_seasons": test_seasons,
        "total_games": len(y_test), "bets_placed": len(decided),
        "pushes": len(pushes), "wins": len(wins),
        "win_rate": len(wins) / len(decided) if decided else 0.0,
        "roi_pct": round(roi, 2), "total_profit_units": round(total_profit, 2),
        "accuracy": float(accuracy_score(yv, pv)) if len(yv) else 0.0,
        "auc": float(roc_auc_score(yv, prv)) if len(np.unique(yv)) > 1 else 0.5,
        "brier": float(brier_score_loss(yv, prv)) if len(yv) else 0.25,
        "log_loss": float(log_loss(yv, prv)) if len(np.unique(yv)) > 1 else 0.0,
        "avg_clv": round(float(np.mean([b["clv"] for b in placed])), 4) if placed else 0.0,
    }


def _compute_aggregate(all_bets, window_results, target):
    if not all_bets:
        return {"total_bets": 0, "win_rate": 0, "overall_roi_pct": 0, "pushes": 0,
                "avg_auc": 0, "avg_brier": 0, "avg_clv": 0, "confidence_tiers": {}}

    decided = [b for b in all_bets if not b.get("push")]
    pushes = [b for b in all_bets if b.get("push")]
    total_profit = sum(b["profit"] for b in all_bets)
    wins = sum(1 for b in decided if b["won"])

    confidence_tiers = {}
    for thresh in [0.52, 0.53, 0.55, 0.57, 0.60, 0.65]:
        tier = [b for b in decided if b["confidence"] >= thresh]   # exclude pushes
        if tier:
            tier_wins = sum(1 for b in tier if b["won"])
            tier_profit = sum(b["profit"] for b in tier)
            confidence_tiers[str(thresh)] = {
                "n": len(tier), "wins": tier_wins,
                "win_rate": round(tier_wins / len(tier), 4),
                "roi_pct": round(tier_profit / len(tier) * 100, 2),
            }

    win_rate = wins / len(decided) if decided else 0.0
    roi_pct = total_profit / len(decided) * 100 if decided else 0.0
    benchmarks = {
        "spread":     {"break_even": BREAK_EVEN, "best_public": 0.53},
        "over_total": {"break_even": BREAK_EVEN, "best_public": 0.535},
        "moneyline":  {"break_even": BREAK_EVEN, "best_public": 0.67},
        "team_total_over": {"break_even": BREAK_EVEN, "best_public": 0.54},
    }
    bench = benchmarks.get(target, {})

    return {
        "total_bets": len(decided), "wins": wins, "losses": len(decided) - wins,
        "pushes": len(pushes),
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