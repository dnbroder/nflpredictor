"""
NFL Model -- XGBoost training, hyperparameter tuning, save/load.

FIXES vs original:
  - Import TARGET_COLUMNS (dict) instead of nonexistent symbol
  - Fixed feature_importance bug: 'sorted_imp' checked via local var not dir()
  - Added NaN target filtering before train/test split
"""

import logging
import os
import pickle
import time
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    log_loss, roc_auc_score, brier_score_loss,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from xgboost import XGBClassifier

from nfl_predictor.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS

logger = logging.getLogger("nflpredictor")

MODEL_FILENAME = "nfl_predictor.pkl"
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILENAME}"
S3_BUCKET = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
S3_MODEL_KEY = f"models/{MODEL_FILENAME}"

DEFAULT_PARAMS = {
    "n_estimators": 400, "max_depth": 5, "learning_rate": 0.04,
    "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 7,
    "reg_alpha": 0.1, "reg_lambda": 1.5, "scale_pos_weight": 1.0,
}

PARAM_GRID = {
    "n_estimators": [200, 300, 400, 600, 800, 1000],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    "min_child_weight": [3, 5, 7, 10, 15],
    "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 5.0],
}


def train_model(featured_data, target="over_total", test_size=0.2,
                random_state=42, params=None, calibrate=True):
    logger.info("[MODEL] ============================================================")
    logger.info(f"[MODEL] TRAINING XGBoost -- target={target}, rows={len(featured_data)}")
    logger.info("[MODEL] ============================================================")

    target_col = TARGET_COLUMNS[target]
    hp = params or DEFAULT_PARAMS

    available = [f for f in FEATURE_COLUMNS if f in featured_data.columns]
    missing = [f for f in FEATURE_COLUMNS if f not in featured_data.columns]
    if missing:
        logger.warning(f"[MODEL] Missing {len(missing)} features: {missing[:5]}...")

    logger.info(f"[MODEL] Using {len(available)} features")
    logger.info(f"[MODEL] Hyperparameters: {hp}")

    X = featured_data[available].copy()
    y = featured_data[target_col].copy()

    # Drop NaN targets
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    # NaN audit
    nan_counts = X.isnull().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if len(cols_with_nans) > 0:
        logger.info(f"[MODEL] NaN audit -- {len(cols_with_nans)} cols with NaN:")
        for col, cnt in cols_with_nans.head(10).items():
            logger.info(f"[MODEL]   {col}: {cnt} ({cnt/len(X)*100:.1f}%)")

    # Correlations
    logger.info("[MODEL] Feature-target correlations (top 15):")
    correlations = {}
    for col in available:
        mask = X[col].notna() & y.notna()
        if mask.sum() > 100:
            correlations[col] = X.loc[mask, col].corr(y[mask])
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs[:15]:
        logger.info(f"[MODEL]   {feat}: {corr:+.4f}")

    logger.info(f"[MODEL] Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"[MODEL] Target positive rate: {y.mean():.3f}")

    # Split: 60/20/20
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=random_state, stratify=y_trainval)
    logger.info(f"[MODEL] Split: train={len(X_train)}, cal={len(X_cal)}, test={len(X_test)}")

    # Train
    t_start = time.time()
    logger.info("[MODEL] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=hp.get("n_estimators", 400),
        max_depth=hp.get("max_depth", 5),
        learning_rate=hp.get("learning_rate", 0.04),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.75),
        min_child_weight=hp.get("min_child_weight", 7),
        reg_alpha=hp.get("reg_alpha", 0.1),
        reg_lambda=hp.get("reg_lambda", 1.5),
        scale_pos_weight=hp.get("scale_pos_weight", 1.0),
        random_state=random_state,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
    t_train = time.time() - t_start
    logger.info(f"[MODEL] Training done in {t_train:.2f}s, "
                f"best_iteration={getattr(xgb, 'best_iteration', 'N/A')}")

    # Calibration
    final_model = xgb
    if calibrate:
        logger.info("[MODEL] Applying Platt scaling...")
        calibrated = CalibratedClassifierCV(xgb, cv="prefit", method="sigmoid")
        calibrated.fit(X_cal, y_cal)
        final_model = calibrated
        logger.info("[MODEL] Platt scaling applied")

    # Evaluate
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    logger.info("[MODEL] ---- Test Set Results ----")
    logger.info(f"[MODEL]   Accuracy:    {accuracy:.4f}  (break-even: 0.5238)")
    logger.info(f"[MODEL]   AUC-ROC:     {auc:.4f}")
    logger.info(f"[MODEL]   Log Loss:    {ll:.4f}")
    logger.info(f"[MODEL]   Brier Score: {brier:.4f}")

    report = classification_report(y_test, y_pred)
    for line in report.split("\n"):
        if line.strip():
            logger.info(f"[MODEL]   {line}")

    # Feature importance (from base xgb, not calibrated wrapper)
    sorted_imp = None
    if hasattr(xgb, "feature_importances_"):
        importance = dict(zip(available, xgb.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("[MODEL] Feature Importance (top 20):")
        for rank, (feat, imp) in enumerate(sorted_imp[:20], 1):
            logger.info(f"[MODEL]   #{rank:2d} {feat}: {imp:.4f}")

    # Confidence tiers
    confidences = np.maximum(y_proba, 1 - y_proba)
    correct = (y_pred == y_test.values).astype(int)
    logger.info("[MODEL] Confidence tier accuracy (test set):")
    for tier_label, tier_thresh in [("55+", 0.55), ("57+", 0.57), ("60+", 0.60), ("65+", 0.65)]:
        mask = confidences >= tier_thresh
        if mask.sum() >= 10:
            tier_acc = correct[mask].mean()
            status = "BEAT" if tier_acc > 0.5238 else "below"
            logger.info(f"[MODEL]   P>={tier_label}: {correct[mask].sum()}/{mask.sum()} "
                        f"({tier_acc*100:.1f}%) -- {status} break-even")
        else:
            logger.info(f"[MODEL]   P>={tier_label}: <10 samples")

    # Prob distribution
    logger.info("[MODEL] Prediction probability distribution (test set):")
    for thresh in [0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
        count_over = (y_proba >= thresh).sum()
        logger.info(f"[MODEL]   P >= {thresh}: {count_over}/{len(y_proba)} "
                    f"({count_over/len(y_proba)*100:.1f}%)")

    _save_model(final_model, available, target)

    metrics = {
        "target": target, "target_col": target_col,
        "accuracy": round(accuracy, 4), "auc_roc": round(auc, 4),
        "log_loss": round(ll, 4), "brier_score": round(brier, 4),
        "beat_break_even": bool(accuracy > 0.5238),
        "train_rows": len(X_train), "cal_rows": len(X_cal), "test_rows": len(X_test),
        "features_used": len(available), "calibrated": calibrate,
        "hyperparameters": hp, "timestamp": datetime.utcnow().isoformat(),
        "confidence_tiers": {
            str(t): {
                "n": int((confidences >= t).sum()),
                "accuracy": float(correct[confidences >= t].mean()) if (confidences >= t).sum() >= 10 else None,
            } for t in [0.55, 0.57, 0.60, 0.65]
        },
    }
    if sorted_imp is not None:
        metrics["feature_importance"] = {k: round(v, 4) for k, v in sorted_imp[:30]}

    logger.info("[MODEL] ============================================================")
    logger.info(f"[MODEL] COMPLETE -- Acc={accuracy:.4f}, AUC={auc:.4f}, Brier={brier:.4f}")
    logger.info("[MODEL] ============================================================")
    return final_model, metrics


def tune_hyperparameters(featured_data, target="over_total", n_iter=50,
                         cv_folds=5, random_state=42):
    import random as rng
    logger.info("[TUNE] ============================================================")
    logger.info(f"[TUNE] Hyperparameter search: {n_iter} iterations, {cv_folds}-fold CV")
    logger.info("[TUNE] ============================================================")

    target_col = TARGET_COLUMNS[target]
    available = [f for f in FEATURE_COLUMNS if f in featured_data.columns]
    X = featured_data[available].copy()
    y = featured_data[target_col].copy()
    valid_mask = y.notna()
    X, y = X[valid_mask], y[valid_mask]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rng.seed(random_state)

    logger.info("[TUNE] Baseline (default params)...")
    baseline_model = XGBClassifier(**DEFAULT_PARAMS, random_state=random_state, eval_metric="logloss")
    baseline_scores = cross_val_score(baseline_model, X, y, cv=cv, scoring="roc_auc")
    baseline_mean = baseline_scores.mean()
    logger.info(f"[TUNE] Baseline AUC: {baseline_mean:.4f} +/- {baseline_scores.std():.4f}")

    all_results = []
    best_score = 0.0
    best_params = None

    for i in range(n_iter):
        trial_params = {k: rng.choice(v) for k, v in PARAM_GRID.items()}
        try:
            model = XGBClassifier(**trial_params, random_state=random_state, eval_metric="logloss")
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            mean_score = scores.mean()
            all_results.append({"iteration": i+1, "params": trial_params,
                                "mean_auc": round(mean_score, 4), "std_auc": round(scores.std(), 4)})
            if mean_score > best_score:
                best_score = mean_score
                best_params = trial_params.copy()
                logger.info(f"[TUNE] Iter {i+1}/{n_iter}: NEW BEST AUC={mean_score:.4f}")
            elif (i+1) % 10 == 0:
                logger.info(f"[TUNE] Iter {i+1}/{n_iter}: AUC={mean_score:.4f} | Best={best_score:.4f}")
        except Exception as e:
            logger.warning(f"[TUNE] Iter {i+1} failed: {e}")

    all_results.sort(key=lambda x: x["mean_auc"], reverse=True)
    improvement = best_score - baseline_mean
    logger.info(f"[TUNE] Baseline: {baseline_mean:.4f}, Best: {best_score:.4f} (delta={improvement:+.4f})")
    logger.info(f"[TUNE] Best params: {best_params}")

    return {
        "baseline": {"params": DEFAULT_PARAMS, "mean_auc": round(baseline_mean, 4)},
        "best": {"params": best_params, "mean_auc": round(best_score, 4),
                 "improvement": round(improvement, 4)},
        "top_10": all_results[:10], "total_iterations": len(all_results),
    }


def _save_model(model, feature_columns, target):
    package = {"model": model, "features": feature_columns, "target": target,
               "saved_at": datetime.utcnow().isoformat()}
    with open(LOCAL_MODEL_PATH, "wb") as f:
        pickle.dump(package, f)
    logger.info(f"[MODEL] Saved locally: {LOCAL_MODEL_PATH}")
    try:
        s3 = boto3.client("s3")
        s3.upload_file(LOCAL_MODEL_PATH, S3_BUCKET, S3_MODEL_KEY)
        logger.info(f"[MODEL] Saved to S3: s3://{S3_BUCKET}/{S3_MODEL_KEY}")
    except Exception as e:
        logger.warning(f"[MODEL] S3 save failed (non-fatal): {e}")


def load_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"[MODEL] Loading from local: {LOCAL_MODEL_PATH}")
        with open(LOCAL_MODEL_PATH, "rb") as f:
            pkg = pickle.load(f)
        return pkg["model"], pkg["features"], pkg.get("target", "over_total")
    try:
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        with open(LOCAL_MODEL_PATH, "rb") as f:
            pkg = pickle.load(f)
        return pkg["model"], pkg["features"], pkg.get("target", "over_total")
    except Exception as e:
        logger.warning(f"[MODEL] No model found: {e}")
        return None, None, None


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
    parser = argparse.ArgumentParser(description="Train NFL prediction model")
    parser.add_argument("--target", default="over_total", choices=list(TARGET_COLUMNS.keys()))
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2018, 2025)))
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    from nfl_predictor.nfl_data import load_games, load_drives
    from nfl_predictor.feature_engineering import engineer_features
    logger.info(f"Loading data for seasons: {args.seasons}")
    games = load_games(args.seasons)
    drives = load_drives(args.seasons)
    featured = engineer_features(games, drives, target=args.target)

    if args.tune:
        tune_results = tune_hyperparameters(featured, target=args.target, n_iter=30)
        best_params = tune_results["best"]["params"]
    else:
        best_params = None

    model, metrics = train_model(featured, target=args.target,
                                 calibrate=not args.no_calibrate, params=best_params)
    logger.info(f"Training complete: {metrics}")