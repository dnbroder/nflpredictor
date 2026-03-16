"""
NFL Model — XGBoost training, hyperparameter tuning, save/load.
"""

from nfl_predictor.logger import get_logger
import os
import pickle
import boto3
import numpy as np
import pandas as pd
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)

from nfl_predictor.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_FILENAME = "nfl_scoring_predictor.pkl"
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILENAME}"
S3_MODEL_KEY = f"models/{MODEL_FILENAME}"

# Default hyperparameters — tuned for NFL totals (~1600 rows, noisy signal)
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,          # shallower — prevents overfitting on noisy signal
    "learning_rate": 0.02,   # slower learning — more trees at lower lr
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,   # was 5/7 — too aggressive, caused best_iteration=3/4
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
}

# Hyperparameter search space
PARAM_GRID = {
    "n_estimators": [300, 500, 700, 1000],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.01, 0.05, 0.1],
    "reg_lambda": [0.5, 1.0, 2.0],
}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_model(
    featured_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    params: dict = None,
) -> tuple:
    """
    Train XGBoost classifier on featured data.

    Args:
        featured_data: DataFrame with feature columns and target
        test_size: Fraction held out for validation
        random_state: Random seed
        params: Optional hyperparameters (uses DEFAULT_PARAMS if None)

    Returns:
        (model, metrics_dict) or (None, error_dict)
    """
    logger.info("[MODEL] ============================================================")
    logger.info(f"[MODEL] TRAINING XGBoost -- target={TARGET_COLUMN}, rows={len(featured_data)}")
    logger.info("[MODEL] ============================================================")

    hp = params or DEFAULT_PARAMS

    # Select available features
    available_features = [f for f in FEATURE_COLUMNS if f in featured_data.columns]
    missing_features = [f for f in FEATURE_COLUMNS if f not in featured_data.columns]

    if missing_features:
        logger.warning(f"[MODEL] Missing features (will skip): {missing_features}")

    logger.info(f"[MODEL] Using {len(available_features)} features")
    logger.info(f"[MODEL] Hyperparameters: {hp}")

    X = featured_data[available_features].copy()
    y = featured_data[TARGET_COLUMN].copy()

    # Closing lines entry diagnostic — confirm closing_total value at model boundary
    if "closing_total" in X.columns:
        ct_valid = X["closing_total"].notna().sum()
        ct_sample = X["closing_total"].dropna().head(3).tolist()
        logger.info(f"[MODEL] closing_total in X: {ct_valid}/{len(X)} valid, sample={ct_sample}")
    else:
        logger.error("[MODEL] closing_total NOT in X — dropped during feature selection")

    # Pre-training NaN audit
    nan_counts = X.isnull().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if len(cols_with_nans) > 0:
        logger.info(f"[MODEL] NaN audit -- {len(cols_with_nans)} cols with NaN:")
        for col, count in cols_with_nans.items():
            logger.info(f"[MODEL]   {col}: {count} ({count/len(X)*100:.1f}%)")
    else:
        logger.info("[MODEL] NaN audit: all features clean")

    # Feature correlation with target
    logger.info("[MODEL] Feature-target correlations (top 15):")
    correlations = {}
    import warnings
    for col in available_features:
        valid_mask = X[col].notna() & y.notna()
        if valid_mask.sum() > 100:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = X.loc[valid_mask, col].corr(y[valid_mask])
            if not np.isnan(corr):
                correlations[col] = corr
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs[:15]:
        logger.info(f"[MODEL]   {feat}: {corr:+.4f}")

    logger.info(f"[MODEL] Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"[MODEL] Target positive rate: {y.mean():.3f}")

    # Three-way split: train / calibration / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp,
    )
    logger.info(f"[MODEL] Split: train={len(X_train)}, cal={len(X_cal)}, test={len(X_test)}")

    # Train XGBoost
    import time as _time
    train_start = _time.time()
    logger.info("[MODEL] Training XGBoost...")

    xgb_model = XGBClassifier(
        n_estimators=hp.get("n_estimators", 500),
        max_depth=hp.get("max_depth", 4),
        learning_rate=hp.get("learning_rate", 0.02),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.7),
        min_child_weight=hp.get("min_child_weight", 3),
        reg_alpha=hp.get("reg_alpha", 0.05),
        reg_lambda=hp.get("reg_lambda", 1.0),
        scale_pos_weight=hp.get("scale_pos_weight", 1.0),
        random_state=random_state,
        eval_metric="auc",        # AUC more meaningful than logloss for this signal
        early_stopping_rounds=50, # give model room before stopping
        n_jobs=-1,
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_cal, y_cal)],
        verbose=False,
    )

    train_elapsed = _time.time() - train_start
    best_iteration = getattr(xgb_model, "best_iteration", None)
    best_score = getattr(xgb_model, "best_score", None)
    logger.info(f"[MODEL] Training done in {train_elapsed:.2f}s, best_iteration={best_iteration}")
    if best_score:
        logger.info(f"[MODEL] Best eval AUC: {best_score:.6f}")

    # Platt scaling calibration on held-out cal set
    # GUARD: skip Platt if XGBoost barely trained (best_iteration <= 10).
    # With 1-2 trees, raw scores are nearly constant — Platt fits a near-zero
    # coefficient and collapses all predictions to the mean (~0.46), making
    # every prediction class 0. Raw probabilities are more useful in that case.
    from sklearn.linear_model import LogisticRegression
    platt = None
    use_platt = best_iteration is not None and best_iteration > 10
    if use_platt:
        logger.info(f"[MODEL] Applying Platt scaling (best_iteration={best_iteration} > 10)...")
        raw_cal = xgb_model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        platt = LogisticRegression(C=1.0)
        platt.fit(raw_cal, y_cal)
        logger.info("[MODEL] Platt scaling applied")
    else:
        logger.warning(f"[MODEL] Skipping Platt scaling — best_iteration={best_iteration} <= 10 "
                       "(too few trees; raw probabilities more reliable than calibrated flat scores)")

    # Evaluate on test set
    raw_test = xgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    if platt is not None:
        y_proba = platt.predict_proba(raw_test)[:, 1]
    else:
        y_proba = raw_test.ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    brier = float(np.mean((y_proba - y_test.values) ** 2))

    logger.info("[MODEL] ---- Test Set Results ----")
    logger.info(f"[MODEL]   Accuracy:    {accuracy:.4f}  (break-even: 0.5238)")
    logger.info(f"[MODEL]   AUC-ROC:     {auc:.4f}")
    logger.info(f"[MODEL]   Log Loss:    {logloss:.4f}")
    logger.info(f"[MODEL]   Brier Score: {brier:.4f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    for line in report.split("\n"):
        if line.strip():
            logger.info(f"[MODEL]   {line}")

    # Feature importance
    importance = dict(zip(available_features, xgb_model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("[MODEL] Feature Importance (top 20):")
    for rank, (feat, imp) in enumerate(sorted_importance[:20], 1):
        logger.info(f"[MODEL]   # {rank} {feat}: {imp:.4f}")

    # Confidence tier accuracy
    confidences = np.maximum(y_proba, 1 - y_proba)
    correct = (y_pred == y_test.values).astype(int)
    logger.info("[MODEL] Confidence tier accuracy (test set):")
    for tier_label, tier_thresh in [("P>=55%", 0.55), ("P>=57%", 0.57), ("P>=60%", 0.60), ("P>=65%", 0.65)]:
        mask = confidences >= tier_thresh
        if mask.sum() >= 10:
            tier_acc = correct[mask].mean()
            be = "BEAT break-even" if tier_acc >= 0.5238 else "below break-even"
            logger.info(f"[MODEL]   {tier_label}: {correct[mask].sum()}/{mask.sum()} ({tier_acc*100:.1f}%) -- {be}")
        else:
            logger.info(f"[MODEL]   {tier_label}: <10 samples")

    # Prediction probability distribution
    logger.info("[MODEL] Prediction probability distribution (test set):")
    for threshold in [0.45, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7]:
        count = (y_proba >= threshold).sum()
        logger.info(f"[MODEL]   P >= {threshold}: {count}/{len(y_proba)} ({count/len(y_proba)*100:.1f}%)")

    # L1/L2 intersection diagnostic
    if "p2s_p2sz_dual_optimal" in X_test.columns:
        dual_mask = X_test["p2s_p2sz_dual_optimal"] == 1.0
        if dual_mask.sum() > 5:
            acc_dual = correct[dual_mask.values].mean()
            acc_non = correct[~dual_mask.values].mean()
            logger.info(f"[MODEL] Intersection diagnostic (L1×L2 dual optimal):")
            logger.info(f"[MODEL]   Dual optimal: {acc_dual:.4f} (n={dual_mask.sum()})")
            logger.info(f"[MODEL]   Non-dual: {acc_non:.4f} | Lift: {acc_dual - acc_non:+.4f}")

    # Save model + platt scaler
    _save_model(xgb_model, platt, available_features)

    metrics = {
        "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "log_loss": round(logloss, 4),
        "brier_score": round(brier, 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": len(available_features),
        "best_iteration": best_iteration,
        "feature_importance": {k: round(v, 4) for k, v in sorted_importance},
        "positive_rate": round(float(y.mean()), 4),
        "hyperparameters": hp,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info("[MODEL] ============================================================")
    logger.info(f"[MODEL] COMPLETE -- Acc={accuracy:.4f}, AUC={auc:.4f}, Brier={brier:.4f}")
    logger.info("[MODEL] ============================================================")

    return xgb_model, metrics


# ---------------------------------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------------------------------
def tune_hyperparameters(
    featured_data: pd.DataFrame,
    n_iter: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Random search over hyperparameter space with cross-validation."""
    import random as rng

    logger.info("[TUNE] ============================================================")
    logger.info(f"[TUNE] Hyperparameter tuning: {n_iter} iterations, {cv_folds}-fold CV")
    logger.info("[TUNE] ============================================================")

    available_features = [f for f in FEATURE_COLUMNS if f in featured_data.columns]
    X = featured_data[available_features].copy()
    y = featured_data[TARGET_COLUMN].copy()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rng.seed(random_state)

    logger.info("[TUNE] Scoring baseline (default params)...")
    baseline_model = XGBClassifier(**{k: v for k, v in DEFAULT_PARAMS.items() if k != "scale_pos_weight"},
                                   random_state=random_state, eval_metric="auc")
    baseline_scores = cross_val_score(baseline_model, X, y, cv=cv, scoring="roc_auc")
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    logger.info(f"[TUNE] Baseline AUC: {baseline_mean:.4f} ± {baseline_std:.4f}")

    all_results = []
    best_score = 0
    best_params = None

    for i in range(n_iter):
        params = {key: rng.choice(values) for key, values in PARAM_GRID.items()}
        try:
            model = XGBClassifier(**params, random_state=random_state, eval_metric="auc")
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            mean_score = scores.mean()
            std_score = scores.std()
            result = {
                "iteration": i + 1,
                "params": params,
                "mean_auc": round(mean_score, 4),
                "std_auc": round(std_score, 4),
            }
            all_results.append(result)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                logger.info(f"[TUNE] Iter {i+1}/{n_iter}: NEW BEST AUC {mean_score:.4f} ± {std_score:.4f}")
            elif (i + 1) % 10 == 0:
                logger.info(f"[TUNE] Iter {i+1}/{n_iter}: {mean_score:.4f} | Best: {best_score:.4f}")
        except Exception as e:
            logger.warning(f"[TUNE] Iter {i+1} failed: {e}")

    all_results.sort(key=lambda x: x["mean_auc"], reverse=True)
    improvement = best_score - baseline_mean

    logger.info("[TUNE] ============================================================")
    logger.info(f"[TUNE] Baseline AUC: {baseline_mean:.4f} ± {baseline_std:.4f}")
    logger.info(f"[TUNE] Best AUC:     {best_score:.4f}")
    logger.info(f"[TUNE] Improvement:  {improvement:+.4f}")
    logger.info(f"[TUNE] Best params:  {best_params}")
    logger.info("[TUNE] ============================================================")

    return {
        "baseline": {
            "params": DEFAULT_PARAMS,
            "mean_auc": round(baseline_mean, 4),
            "std_auc": round(baseline_std, 4),
        },
        "best": {
            "params": best_params,
            "mean_auc": round(best_score, 4),
            "improvement": round(improvement, 4),
        },
        "top_10": all_results[:10],
        "total_iterations": len(all_results),
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def _save_model(model, platt, feature_columns: list):
    """Save model + platt scaler + feature list to local disk and S3."""
    model_package = {
        "model": model,
        "platt": platt,
        "features": feature_columns,
        "saved_at": datetime.utcnow().isoformat(),
    }

    with open(LOCAL_MODEL_PATH, "wb") as f:
        pickle.dump(model_package, f)
    logger.info(f"[MODEL] Saved locally: {LOCAL_MODEL_PATH}")

    try:
        bucket = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
        s3 = boto3.client("s3")
        s3.upload_file(LOCAL_MODEL_PATH, bucket, S3_MODEL_KEY)
        logger.info(f"[MODEL] Saved to S3: s3://{bucket}/{S3_MODEL_KEY}")
    except Exception as e:
        logger.warning(f"[MODEL] S3 save failed (non-fatal): {e}")


def load_model() -> tuple:
    """Load model from local disk or S3."""
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"[MODEL] Loading from local: {LOCAL_MODEL_PATH}")
        with open(LOCAL_MODEL_PATH, "rb") as f:
            package = pickle.load(f)
        return package["model"], package.get("features")

    try:
        bucket = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
        s3 = boto3.client("s3")
        s3.download_file(bucket, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        logger.info(f"[MODEL] Downloaded from S3: s3://{bucket}/{S3_MODEL_KEY}")
        with open(LOCAL_MODEL_PATH, "rb") as f:
            package = pickle.load(f)
        return package["model"], package.get("features")
    except Exception as e:
        logger.warning(f"[MODEL] No model found: {e}")
        return None, None