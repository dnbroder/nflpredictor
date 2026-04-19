"""
NFL Model — Two-Stage Architecture for Over/Under Prediction.

Stage 1: Rule-based filter identifies candidate games where cycle alignment
         suggests the over is likely (dual optimal, PIC confirmed, etc.)
Stage 2: Logistic regression scores filtered candidates using the full
         feature set to produce calibrated probabilities.

WHY NOT XGBOOST:
  XGBoost with early stopping on AUC collapses to 3-8 trees against a
  closing-line target because individual feature correlations are 0.02-0.08.
  The gradient is too flat for boosting to find a stable path before early
  stopping kills it. The dual optimal signal (57.9% accuracy, +6.4% lift)
  is real but sparse (14% of games) — logistic regression handles sparse
  binary signals better than tree ensembles because it doesn't need to
  find a split threshold, it just learns a coefficient.

ARCHITECTURE:
  - Stage 1 filters to games with positive cycle alignment (bet-the-over
    candidates) or negative alignment (bet-the-under candidates).
  - Stage 2 logistic regression on the full training set produces
    calibrated probabilities.
  - Final bet decision: Stage 1 filter passes AND Stage 2 probability
    exceeds the confidence threshold.
"""

from nfl_predictor.logger import get_logger
import os
import pickle
import boto3
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


# ---------------------------------------------------------------------------
# Stage 1: Rule-based filter
# ---------------------------------------------------------------------------
OVER_FILTER_RULES = {
    "dual_optimal": {
        "col": "p2s_p2sz_dual_optimal",
        "op": "==",
        "val": 1.0,
        "label": "L1xL2 dual optimal zone",
    },
}

OVER_BOOST_SIGNALS = {
    "pic_confirmed": {
        "col": "pic_cycle_confirmed",
        "op": "==",
        "val": 1.0,
        "weight": 0.10,
        "label": "PIC all-layer alignment",
    },
    "matchup_dual": {
        "col": "matchup_dual_optimal",
        "op": "==",
        "val": 1.0,
        "weight": 0.05,
        "label": "Matchup dual optimal (opp def suppressed)",
    },
    "not_windy": {
        "col": "wind_mph",
        "op": "<=",
        "val": 15.0,
        "weight": 0.03,
        "label": "Not windy (favorable scoring conditions)",
    },
    "dome": {
        "col": "is_dome",
        "op": "==",
        "val": 1.0,
        "weight": 0.03,
        "label": "Dome game (controlled environment)",
    },
}

UNDER_FILTER_RULES = {
    "cold_streak": {
        "col": "p2s_cold_streak_prob",
        "op": ">=",
        "val": 0.6,
        "label": "Cold streak probability high",
    },
    "low_conversion": {
        "col": "p2s_conversion_rate_5g",
        "op": "<=",
        "val": 0.25,
        "label": "Low recent scoring drive rate",
    },
}


def apply_filter(df: pd.DataFrame, rules: dict) -> pd.Series:
    """Apply rule-based filter, returning a boolean mask. All rules must pass (AND logic)."""
    mask = pd.Series(True, index=df.index)
    for rule_name, rule in rules.items():
        col = rule["col"]
        if col not in df.columns:
            continue
        op = rule["op"]
        val = rule["val"]
        if op == "==":
            mask &= (df[col] == val)
        elif op == ">=":
            mask &= (df[col] >= val)
        elif op == "<=":
            mask &= (df[col] <= val)
        elif op == ">":
            mask &= (df[col] > val)
        elif op == "<":
            mask &= (df[col] < val)
    return mask


def compute_boost_score(df: pd.DataFrame, signals: dict) -> pd.Series:
    """Compute additive boost from strengthening signals."""
    boost = pd.Series(0.0, index=df.index)
    for sig_name, sig in signals.items():
        col = sig["col"]
        if col not in df.columns:
            continue
        op = sig["op"]
        val = sig["val"]
        weight = sig["weight"]
        if op == "==":
            boost += (df[col] == val).astype(float) * weight
        elif op == ">=":
            boost += (df[col] >= val).astype(float) * weight
        elif op == "<=":
            boost += (df[col] <= val).astype(float) * weight
    return boost


# ---------------------------------------------------------------------------
# Stage 2: Logistic Regression — feature subset
# ---------------------------------------------------------------------------
LOGISTIC_FEATURES = [
    # Weather / venue — drop is_windy_game (multicollinear with wind_mph, opposite signs)
    "wind_mph", "temp", "is_dome", "is_cold_game",
    # Market signals
    "opening_total", "total_movement", "opening_spread", "spread_movement",
    # Cycle state (L1) — drop hot/cold streak probs (low coefficient, noisy)
    "p2s_ratio", "p2s_optimal_zone", "p2s_conversion_rate_5g",
    # Cycle state (L2) — drop p2sz_rz_rate_5g (redundant with p2sz_rz_ratio)
    "p2sz_rz_ratio", "p2sz_rz_optimal_zone", "p2sz_sp_baseline",
    # Intersection
    "p2s_p2sz_dual_optimal", "p2s_p2sz_signal_strength", "matchup_dual_optimal",
    # Defense
    "p2s_def_ratio", "p2s_def_optimal_zone",
    # EPA
    "epa_offense_avg", "epa_defense_avg",
    # PIC (Layer 3) — keep all, these are the strongest features
    "pic_qb_in_zone", "pic_qb_career_rate", "pic_team_alignment",
    "pic_skill_sync_rate", "pic_avg_player_epa", "pic_present_quality",
    "pic_cycle_confirmed", "pic_qb_is_backup",
    # Scoring trends
    "scoring_avg_5g", "scoring_allowed_avg_5g",
    # Context — drop rest_advantage and is_div_game (near-zero coefficients)
    "is_home",
]


# ---------------------------------------------------------------------------
# Legacy constants (kept for backward compat with tune endpoint)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "n_estimators": 1500, "max_depth": 3, "learning_rate": 0.008,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
    "reg_alpha": 0.1, "reg_lambda": 2.0, "scale_pos_weight": 1.0,
}

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
    Train two-stage model: rule filter + logistic regression.
    Returns (model_package, metrics_dict).
    """
    logger.info("[MODEL] ============================================================")
    logger.info(f"[MODEL] TRAINING Two-Stage Model -- target={TARGET_COLUMN}, rows={len(featured_data)}")
    logger.info("[MODEL] ============================================================")

    available_features = [f for f in LOGISTIC_FEATURES if f in featured_data.columns]
    all_features = [f for f in FEATURE_COLUMNS if f in featured_data.columns]
    missing = [f for f in LOGISTIC_FEATURES if f not in featured_data.columns]
    if missing:
        logger.warning(f"[MODEL] Missing logistic features (will skip): {missing}")

    logger.info(f"[MODEL] Logistic features: {len(available_features)}/{len(LOGISTIC_FEATURES)}")
    logger.info(f"[MODEL] Architecture: Stage 1 (rule filter) + Stage 2 (logistic regression)")

    X = featured_data[available_features].copy()
    y = featured_data[TARGET_COLUMN].copy()

    # NaN audit
    nan_counts = X.isnull().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if len(cols_with_nans) > 0:
        logger.info(f"[MODEL] NaN audit -- {len(cols_with_nans)} cols with NaN:")
        for col, count in cols_with_nans.items():
            logger.info(f"[MODEL]   {col}: {count} ({count/len(X)*100:.1f}%)")

    # Feature correlations
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

    # Train/test split
    X_full = featured_data[all_features].copy()
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train = X_train_full[available_features].copy()
    X_test = X_test_full[available_features].copy()
    logger.info(f"[MODEL] Split: train={len(X_train)}, test={len(X_test)}")

    # ── Stage 1: Filter evaluation ──
    logger.info("[MODEL] -- Stage 1: Rule Filter Evaluation --")

    over_mask_train = apply_filter(X_train_full, OVER_FILTER_RULES)
    over_mask_test = apply_filter(X_test_full, OVER_FILTER_RULES)

    n_over_train = over_mask_train.sum()
    n_over_test = over_mask_test.sum()
    logger.info(f"[MODEL]   Over filter (train): {n_over_train}/{len(X_train)} ({n_over_train/len(X_train)*100:.1f}%)")
    logger.info(f"[MODEL]   Over filter (test):  {n_over_test}/{len(X_test)} ({n_over_test/len(X_test)*100:.1f}%)")

    filter_acc = None
    if n_over_test > 5:
        filter_acc = float((y_test[over_mask_test] == 1).mean())
        base_rate = float(y_test.mean())
        lift = filter_acc - base_rate
        logger.info(f"[MODEL]   Filter accuracy (test): {filter_acc:.4f} (n={n_over_test})")
        logger.info(f"[MODEL]   Base rate:              {base_rate:.4f}")
        logger.info(f"[MODEL]   Filter lift:            {lift:+.4f} ({lift/base_rate*100:+.1f}%)")

    # Boost diagnostic
    boost_test = compute_boost_score(X_test_full, OVER_BOOST_SIGNALS)
    if n_over_test > 5:
        boosted_mask = over_mask_test & (boost_test > 0.05)
        if boosted_mask.sum() > 5:
            boosted_acc = float((y_test[boosted_mask] == 1).mean())
            logger.info(f"[MODEL]   Filter + boost (test):  {boosted_acc:.4f} (n={boosted_mask.sum()})")

    # ── Stage 2: Logistic Regression ──
    logger.info("[MODEL] -- Stage 2: Logistic Regression --")

    import time as _time
    train_start = _time.time()

    # Compute train medians BEFORE imputation for reproducibility
    train_medians = X_train.median()

    X_train_imputed = X_train.fillna(train_medians)
    X_test_imputed = X_test.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    lr_model = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs", max_iter=1000, random_state=random_state,
    )
    lr_model.fit(X_train_scaled, y_train)

    train_elapsed = _time.time() - train_start
    logger.info(f"[MODEL] Logistic regression trained in {train_elapsed:.2f}s")

    # Coefficients
    coef_dict = dict(zip(available_features, lr_model.coef_[0]))
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info("[MODEL] Top 20 logistic coefficients (magnitude):")
    for rank, (feat, coef) in enumerate(sorted_coefs[:20], 1):
        logger.info(f"[MODEL]   # {rank} {feat}: {coef:+.4f}")

    # ── Evaluate ──
    y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    brier = float(np.mean((y_proba - y_test.values) ** 2))

    logger.info("[MODEL] ---- Test Set Results (all games) ----")
    logger.info(f"[MODEL]   Accuracy:    {accuracy:.4f}  (break-even: 0.5238)")
    logger.info(f"[MODEL]   AUC-ROC:     {auc:.4f}")
    logger.info(f"[MODEL]   Log Loss:    {logloss:.4f}")
    logger.info(f"[MODEL]   Brier Score: {brier:.4f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    for line in report.split("\n"):
        if line.strip():
            logger.info(f"[MODEL]   {line}")

    # Probability spread
    logger.info("[MODEL] Probability spread diagnostic (test set):")
    logger.info(f"[MODEL]   Logistic — mean={y_proba.mean():.3f}, "
               f"std={y_proba.std():.3f}, min={y_proba.min():.3f}, max={y_proba.max():.3f}, "
               f"above_0.5={(y_proba >= 0.5).sum()}/{len(y_proba)}")
    percentiles = [10, 25, 50, 75, 90]
    pct_vals = np.percentile(y_proba, percentiles)
    pct_str = ", ".join(f"p{p}={v:.3f}" for p, v in zip(percentiles, pct_vals))
    logger.info(f"[MODEL]   Percentiles: {pct_str}")

    # ── Two-stage combined evaluation ──
    logger.info("[MODEL] -- Two-Stage Combined Results --")

    over_bettable = over_mask_test & (y_proba >= 0.50)
    if over_bettable.sum() > 5:
        over_acc = float((y_test[over_bettable] == 1).mean())
        logger.info(f"[MODEL]   OVER bets (filter + P>=0.50): {over_acc:.4f} "
                   f"(n={over_bettable.sum()}, lift={over_acc - float(y_test.mean()):+.4f})")

    logger.info("[MODEL]   Confidence tiers (filtered over candidates):")
    for thresh_label, thresh in [("P>=0.50", 0.50), ("P>=0.52", 0.52),
                                  ("P>=0.55", 0.55), ("P>=0.57", 0.57), ("P>=0.60", 0.60)]:
        tier_mask = over_mask_test & (y_proba >= thresh)
        n_tier = tier_mask.sum()
        if n_tier >= 5:
            tier_acc = float((y_test[tier_mask] == 1).mean())
            logger.info(f"[MODEL]     {thresh_label}: {tier_acc:.4f} win rate "
                       f"(n={n_tier}, ROI={_roi_from_winrate(tier_acc):+.1f}%)")
        else:
            logger.info(f"[MODEL]     {thresh_label}: <5 samples")

    under_mask_test = apply_filter(X_test_full, UNDER_FILTER_RULES)
    under_bettable = under_mask_test & (y_proba < 0.50)
    if under_bettable.sum() > 5:
        under_acc = float((y_test[under_bettable] == 0).mean())
        logger.info(f"[MODEL]   UNDER bets (filter + P<0.50): {under_acc:.4f} "
                   f"(n={under_bettable.sum()}, lift={under_acc - (1 - float(y_test.mean())):+.4f})")

    all_bettable = over_bettable | under_bettable
    if all_bettable.sum() > 5:
        over_correct = int((y_test[over_bettable] == 1).sum()) if over_bettable.sum() > 0 else 0
        under_correct = int((y_test[under_bettable] == 0).sum()) if under_bettable.sum() > 0 else 0
        total_bets = int(all_bettable.sum())
        total_correct = over_correct + under_correct
        combined_acc = total_correct / total_bets
        logger.info(f"[MODEL]   ALL bets combined: {combined_acc:.4f} win rate "
                   f"(n={total_bets}, ROI={_roi_from_winrate(combined_acc):+.1f}%)")

    # Intersection diagnostic (comparability with XGBoost runs)
    if "p2s_p2sz_dual_optimal" in X_test_full.columns:
        dual_mask = X_test_full["p2s_p2sz_dual_optimal"] == 1.0
        if dual_mask.sum() > 5:
            acc_dual_raw = float((y_test[dual_mask] == 1).mean())
            acc_non = float((y_test[~dual_mask] == 1).mean())
            logger.info(f"[MODEL] Intersection diagnostic (L1xL2 dual optimal):")
            logger.info(f"[MODEL]   Dual optimal (raw): {acc_dual_raw:.4f} (n={dual_mask.sum()})")
            logger.info(f"[MODEL]   Non-dual: {acc_non:.4f} | Lift: {acc_dual_raw - acc_non:+.4f}")

            if "pic_ol_sync_rate" in X_test_full.columns:
                pic_complete = X_test_full["pic_ol_sync_rate"].notna()
                dual_pic = dual_mask & pic_complete
                non_dual_pic = ~dual_mask & pic_complete
                if dual_pic.sum() > 10 and non_dual_pic.sum() > 10:
                    acc_dual_pic = float((y_test[dual_pic] == 1).mean())
                    acc_non_pic = float((y_test[non_dual_pic] == 1).mean())
                    logger.info(f"[MODEL]   Dual optimal (PIC-complete, n={dual_pic.sum()}):")
                    logger.info(f"[MODEL]     Accuracy: {acc_dual_pic:.4f} | Lift: {acc_dual_pic - acc_non_pic:+.4f}")

    # ── Save ──
    model_package = {
        "type": "two_stage_v1",
        "scaler": scaler,
        "logistic_model": lr_model,
        "logistic_features": available_features,
        "all_features": all_features,
        "train_medians": train_medians.to_dict(),
        "over_filter_rules": OVER_FILTER_RULES,
        "over_boost_signals": OVER_BOOST_SIGNALS,
        "under_filter_rules": UNDER_FILTER_RULES,
        "saved_at": datetime.utcnow().isoformat(),
    }
    _save_model_package(model_package)

    metrics = {
        "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "log_loss": round(logloss, 4),
        "brier_score": round(brier, 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": len(available_features),
        "filter_pass_rate": round(float(n_over_test) / len(X_test), 4) if len(X_test) > 0 else 0,
        "filter_accuracy": round(filter_acc, 4) if filter_acc is not None else None,
        "coefficients": {k: round(v, 4) for k, v in sorted_coefs[:20]},
        "positive_rate": round(float(y.mean()), 4),
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info("[MODEL] ============================================================")
    logger.info(f"[MODEL] COMPLETE -- Acc={accuracy:.4f}, AUC={auc:.4f}, Brier={brier:.4f}")
    logger.info("[MODEL] ============================================================")

    return model_package, metrics


def _roi_from_winrate(win_rate: float) -> float:
    """Convert win rate to ROI assuming -110 odds."""
    if win_rate <= 0:
        return -100.0
    return (win_rate * (100 / 110) - (1 - win_rate)) * 100


# ---------------------------------------------------------------------------
# Prediction (production)
# ---------------------------------------------------------------------------
def predict(model_package: dict, game_features: pd.DataFrame) -> pd.DataFrame:
    """Score games using the two-stage model."""
    lr_model = model_package["logistic_model"]
    scaler = model_package["scaler"]
    features = model_package["logistic_features"]
    medians = model_package.get("train_medians", {})

    X = game_features[features].copy()
    for col in features:
        if col in medians:
            X[col] = X[col].fillna(medians[col])
    X_scaled = scaler.transform(X)
    proba = lr_model.predict_proba(X_scaled)[:, 1]

    over_mask = apply_filter(game_features, model_package.get("over_filter_rules", OVER_FILTER_RULES))
    under_mask = apply_filter(game_features, model_package.get("under_filter_rules", UNDER_FILTER_RULES))
    boost = compute_boost_score(game_features, model_package.get("over_boost_signals", OVER_BOOST_SIGNALS))

    result = pd.DataFrame({
        "over_probability": proba,
        "over_filter": over_mask,
        "under_filter": under_mask,
        "boost_score": boost,
    }, index=game_features.index)

    result["bet_recommendation"] = "no_bet"
    result.loc[over_mask & (proba >= 0.50), "bet_recommendation"] = "over"
    result.loc[under_mask & (proba < 0.50), "bet_recommendation"] = "under"

    return result


# ---------------------------------------------------------------------------
# Tune (logistic C parameter)
# ---------------------------------------------------------------------------
def tune_hyperparameters(
    featured_data: pd.DataFrame,
    n_iter: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Tune logistic regression C parameter via cross-validation.
    
    Tests a range of C values and reports both AUC and estimated ROI
    at the P>=0.52 threshold for each.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    logger.info("[TUNE] ============================================================")
    logger.info(f"[TUNE] Logistic C tuning: {n_iter} values, {cv_folds}-fold CV")
    logger.info("[TUNE] ============================================================")

    available_features = [f for f in LOGISTIC_FEATURES if f in featured_data.columns]
    logger.info(f"[TUNE] Features: {len(available_features)}/{len(LOGISTIC_FEATURES)}")
    
    X = featured_data[available_features].fillna(featured_data[available_features].median())
    y = featured_data[TARGET_COLUMN].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Test a broad range plus extra density around typical sweet spots
    c_values = np.logspace(-3, 2, n_iter)
    best_score = 0
    best_c = None
    results = []

    for c_val in c_values:
        lr = LogisticRegression(C=c_val, penalty="l2", solver="lbfgs",
                                max_iter=1000, random_state=random_state)
        scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")
        mean_score = scores.mean()
        std_score = scores.std()
        results.append({
            "C": round(float(c_val), 6),
            "mean_auc": round(float(mean_score), 4),
            "std_auc": round(float(std_score), 4),
        })
        if mean_score > best_score:
            best_score = mean_score
            best_c = c_val
            logger.info(f"[TUNE] C={c_val:.4f}: NEW BEST AUC {mean_score:.4f} (±{std_score:.4f})")

    # Sort and show top 10
    results_sorted = sorted(results, key=lambda x: x["mean_auc"], reverse=True)
    logger.info("[TUNE] ── Top 10 C values ──")
    for i, r in enumerate(results_sorted[:10]):
        marker = " ◀ BEST" if r["C"] == round(float(best_c), 6) else ""
        logger.info(f"[TUNE]   #{i+1} C={r['C']:.6f}: AUC={r['mean_auc']:.4f} (±{r['std_auc']:.4f}){marker}")

    logger.info("[TUNE] ============================================================")
    logger.info(f"[TUNE] Best C: {best_c:.6f}, AUC: {best_score:.4f}")
    logger.info(f"[TUNE] Current C in model: 0.1")
    logger.info("[TUNE] ============================================================")

    return {
        "best": {"C": float(best_c), "mean_auc": round(float(best_score), 4)},
        "all_results": results_sorted[:10],
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def _save_model_package(model_package: dict):
    """Save two-stage model package to local disk and S3."""
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


def _save_model(model, calibrator, feature_columns: list):
    """Legacy compatibility wrapper."""
    _save_model_package({
        "type": "legacy_xgboost",
        "model": model,
        "calibrator": calibrator,
        "features": feature_columns,
        "saved_at": datetime.utcnow().isoformat(),
    })


def load_model() -> tuple:
    """
    Load model. Returns (model_package, features, None) for two-stage,
    or (xgb_model, features, calibrator) for legacy.
    """
    def _load_from_path(path):
        with open(path, "rb") as f:
            package = pickle.load(f)
        if isinstance(package, dict) and package.get("type", "").startswith("two_stage"):
            return package, package.get("logistic_features") or package.get("all_features"), None
        calibrator = package.get("calibrator") or package.get("platt")
        return package.get("model") or package, package.get("features"), calibrator

    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"[MODEL] Loading from local: {LOCAL_MODEL_PATH}")
        return _load_from_path(LOCAL_MODEL_PATH)

    try:
        bucket = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
        s3 = boto3.client("s3")
        s3.download_file(bucket, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        logger.info(f"[MODEL] Downloaded from S3: s3://{bucket}/{S3_MODEL_KEY}")
        return _load_from_path(LOCAL_MODEL_PATH)
    except Exception as e:
        logger.warning(f"[MODEL] No model found: {e}")
        return None, None, None