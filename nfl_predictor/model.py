"""
NFL Model — XGBoost training, hyperparameter tuning, save/load.
Mirrors GolfPredictor model.py exactly.
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

# Default hyperparameters (baseline)
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# Hyperparameter search space
PARAM_GRID = {
    "n_estimators": [200, 300, 500, 700, 1000],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
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
    logger.info(f"[MODEL] TRAINING XGBoost on {len(featured_data)} rows")
    logger.info("[MODEL] ============================================================")

    hp = params or DEFAULT_PARAMS

    # Select available features
    available_features = [f for f in FEATURE_COLUMNS if f in featured_data.columns]
    missing_features = [f for f in FEATURE_COLUMNS if f not in featured_data.columns]

    if missing_features:
        logger.warning(f"[MODEL] Missing features (will skip): {missing_features}")

    logger.info(f"[MODEL] Using {len(available_features)} features: {available_features}")
    logger.info(f"[MODEL] Hyperparameters: {hp}")

    X = featured_data[available_features].copy()
    y = featured_data[TARGET_COLUMN].copy()

    # Pre-training NaN audit
    nan_counts = X.isnull().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if len(cols_with_nans) > 0:
        logger.warning(f"[MODEL] Pre-train NaN audit ({len(cols_with_nans)} features have NaN):")
        for col, count in cols_with_nans.items():
            logger.warning(f"[MODEL]   {col}: {count} NaN ({count/len(X)*100:.1f}%)")
    else:
        logger.info("[MODEL] Pre-train NaN audit: all features clean")

    # Feature correlation with target (quick diagnostic)
    logger.info("[MODEL] Feature-target correlations (top 10):")
    correlations = {}
    for col in available_features:
        valid_mask = X[col].notna() & y.notna()
        if valid_mask.sum() > 100:
            correlations[col] = X.loc[valid_mask, col].corr(y[valid_mask])
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs[:10]:
        logger.info(f"[MODEL]   {feat}: {corr:+.4f}")

    logger.info(f"[MODEL] Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"[MODEL] Target balance: {y.mean():.3f} positive rate")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    logger.info(f"[MODEL] Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Train XGBoost
    import time as _time
    train_start = _time.time()
    logger.info("[MODEL] Training XGBoost classifier...")
    logger.info(f"[MODEL]   n_estimators={hp.get('n_estimators')}, max_depth={hp.get('max_depth')}, "
                f"lr={hp.get('learning_rate')}, subsample={hp.get('subsample')}")
    xgb_model = XGBClassifier(
        n_estimators=hp.get("n_estimators", 300),
        max_depth=hp.get("max_depth", 5),
        learning_rate=hp.get("learning_rate", 0.05),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.8),
        min_child_weight=hp.get("min_child_weight", 5),
        reg_alpha=hp.get("reg_alpha", 0.1),
        reg_lambda=hp.get("reg_lambda", 1.0),
        random_state=random_state,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    train_elapsed = _time.time() - train_start
    logger.info(f"[MODEL] Training completed in {train_elapsed:.2f}s")

    # Early stopping info
    best_iteration = getattr(xgb_model, "best_iteration", None)
    best_score = getattr(xgb_model, "best_score", None)
    if best_iteration is not None:
        logger.info(f"[MODEL] Early stopping: best iteration = {best_iteration}/{hp.get('n_estimators', 300)}")
        logger.info(f"[MODEL] Early stopping: best logloss = {best_score:.6f}")
    else:
        logger.info(f"[MODEL] No early stopping triggered — used all {hp.get('n_estimators', 300)} estimators")

    # Evaluate
    logger.info("[MODEL] Evaluating model...")
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)

    logger.info(f"[MODEL] Results:")
    logger.info(f"[MODEL]   Accuracy: {accuracy:.4f}")
    logger.info(f"[MODEL]   AUC-ROC:  {auc:.4f}")
    logger.info(f"[MODEL]   Log Loss: {logloss:.4f}")

    report = classification_report(y_test, y_pred)
    for line in report.split("\n"):
        if line.strip():
            logger.info(f"[MODEL]   {line}")

    # Feature importance
    importance = dict(zip(available_features, xgb_model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("[MODEL] Feature Importance (ALL features, ranked):")
    for rank, (feat, imp) in enumerate(sorted_importance, 1):
        logger.info(f"[MODEL]   #{rank} {feat}: {imp:.4f}")

    # Prediction probability distribution
    logger.info("[MODEL] Prediction probability distribution:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (y_proba >= threshold).sum()
        logger.info(f"[MODEL]   P >= {threshold}: {count} predictions ({count/len(y_proba)*100:.1f}%)")

    # Per-confidence-tier accuracy on test set
    confidences = np.maximum(y_proba, 1 - y_proba)
    correct = (y_pred == y_test.values).astype(int)
    logger.info("[MODEL] Confidence tier accuracy (test set):")
    for tier_label, tier_thresh in [("55+", 0.55), ("60+", 0.60), ("65+", 0.65), ("70+", 0.70), ("75+", 0.75)]:
        mask = confidences >= tier_thresh
        if mask.sum() > 0:
            tier_acc = correct[mask].mean()
            logger.info(f"[MODEL]   {tier_label}: {correct[mask].sum()}/{mask.sum()} ({tier_acc*100:.1f}%)")
        else:
            logger.info(f"[MODEL]   {tier_label}: 0 predictions at this confidence")

    # L1/L2 intersection diagnostic
    if "p2s_p2sz_dual_optimal" in X_test.columns:
        dual_mask = X_test["p2s_p2sz_dual_optimal"] == 1.0
        if dual_mask.sum() > 5:
            acc_dual = correct[dual_mask.values].mean()
            acc_non = correct[~dual_mask.values].mean()
            logger.info(f"[MODEL] Intersection diagnostic (L1×L2 dual optimal):")
            logger.info(f"[MODEL]   Dual optimal: {acc_dual:.4f} (n={dual_mask.sum()})")
            logger.info(f"[MODEL]   Non-dual: {acc_non:.4f} | Lift: {acc_dual - acc_non:+.4f}")

    # Save model
    _save_model(xgb_model, available_features)

    metrics = {
        "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "log_loss": round(logloss, 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": len(available_features),
        "feature_importance": {k: round(v, 4) for k, v in sorted_importance},
        "positive_rate": round(float(y.mean()), 4),
        "hyperparameters": hp,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info("[MODEL] ============================================================")
    logger.info(f"[MODEL] Training complete. Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
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
    baseline_model = XGBClassifier(**DEFAULT_PARAMS, random_state=random_state, eval_metric="logloss")
    baseline_scores = cross_val_score(baseline_model, X, y, cv=cv, scoring="accuracy")
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    logger.info(f"[TUNE] Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")

    all_results = []
    best_score = 0
    best_params = None

    for i in range(n_iter):
        params = {key: rng.choice(values) for key, values in PARAM_GRID.items()}
        try:
            model = XGBClassifier(**params, random_state=random_state, eval_metric="logloss")
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            mean_score = scores.mean()
            std_score = scores.std()
            result = {
                "iteration": i + 1,
                "params": params,
                "mean_accuracy": round(mean_score, 4),
                "std_accuracy": round(std_score, 4),
                "cv_scores": [round(s, 4) for s in scores],
            }
            all_results.append(result)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                logger.info(f"[TUNE] Iter {i+1}/{n_iter}: NEW BEST {mean_score:.4f} ± {std_score:.4f}")
            elif (i + 1) % 10 == 0:
                logger.info(f"[TUNE] Iter {i+1}/{n_iter}: {mean_score:.4f} | Best: {best_score:.4f}")
        except Exception as e:
            logger.warning(f"[TUNE] Iter {i+1} failed: {e}")

    all_results.sort(key=lambda x: x["mean_accuracy"], reverse=True)
    improvement = best_score - baseline_mean

    logger.info("[TUNE] ============================================================")
    logger.info(f"[TUNE] Baseline:    {baseline_mean:.4f} ± {baseline_std:.4f}")
    logger.info(f"[TUNE] Best tuned:  {best_score:.4f}")
    logger.info(f"[TUNE] Improvement: {improvement:+.4f}")
    logger.info(f"[TUNE] Best params: {best_params}")
    logger.info("[TUNE] ============================================================")

    return {
        "baseline": {
            "params": DEFAULT_PARAMS,
            "mean_accuracy": round(baseline_mean, 4),
            "std_accuracy": round(baseline_std, 4),
        },
        "best": {
            "params": best_params,
            "mean_accuracy": round(best_score, 4),
            "improvement": round(improvement, 4),
        },
        "top_10": all_results[:10],
        "total_iterations": len(all_results),
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def _save_model(model, feature_columns: list):
    """Save model + feature list to local disk and S3."""
    model_package = {
        "model": model,
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
        return package["model"], package["features"]

    try:
        bucket = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
        s3 = boto3.client("s3")
        s3.download_file(bucket, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        logger.info(f"[MODEL] Downloaded from S3: s3://{bucket}/{S3_MODEL_KEY}")
        with open(LOCAL_MODEL_PATH, "rb") as f:
            package = pickle.load(f)
        return package["model"], package["features"]
    except Exception as e:
        logger.warning(f"[MODEL] No model found: {e}")
        return None, None