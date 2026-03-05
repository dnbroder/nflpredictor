"""
NFL SHAP Analysis — Feature importance and interaction analysis.
Mirrors GolfPredictor shap_analysis.py exactly.
Uses SHAP (SHapley Additive exPlanations) to explain model predictions.
"""

from nfl_predictor.logger import get_logger
import os
import json
import time
import boto3
import numpy as np
import pandas as pd
from datetime import datetime

from nfl_predictor.model import load_model
from nfl_predictor.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

logger = get_logger(__name__)


def run_shap_analysis(
    featured_data: pd.DataFrame,
    max_samples: int = 2000,
) -> dict:
    """
    Run SHAP analysis on trained model.

    Computes:
    1. Global feature importance (mean |SHAP value|)
    2. Directional impact (over vs under)
    3. Per-confidence-tier SHAP breakdowns
    4. L1 cycle baseline deep dive
    5. L2 P2SZ feature deep dive
    6. L1×L2 intersection signal analysis
    """
    import shap

    total_start = time.time()

    logger.info("[SHAP] ============================================================")
    logger.info("[SHAP] SHAP ANALYSIS START")
    logger.info(f"[SHAP]   Input rows: {len(featured_data)}")
    logger.info(f"[SHAP]   Max samples: {max_samples}")
    logger.info(f"[SHAP]   SHAP version: {shap.__version__}")
    logger.info("[SHAP] ============================================================")

    model, feature_cols = load_model()
    if model is None:
        logger.error("[SHAP] No trained model found — cannot run SHAP")
        return {"error": "No trained model. Run /train first."}

    logger.info(f"[SHAP] Model loaded. Features expected: {len(feature_cols)}")

    available_features = [f for f in feature_cols if f in featured_data.columns]
    missing_features = [f for f in feature_cols if f not in featured_data.columns]
    if missing_features:
        logger.warning(f"[SHAP] Missing features (model expects but data lacks): {missing_features}")
    logger.info(f"[SHAP] Features available: {len(available_features)}/{len(feature_cols)}")

    X = featured_data[available_features].copy()

    # Sample for speed
    if len(X) > max_samples:
        logger.info(f"[SHAP] Sampling {max_samples}/{len(X)} rows for speed")
        X = X.sample(max_samples, random_state=42)

    logger.info(f"[SHAP] Computing SHAP values for {len(X)} samples...")
    shap_start = time.time()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classifier, shap_values may be list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    shap_elapsed = time.time() - shap_start
    logger.info(f"[SHAP] SHAP values computed in {shap_elapsed:.1f}s")

    # 1. Global feature importance
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_importance = dict(zip(available_features, mean_abs_shap))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    logger.info("[SHAP] Global feature importance (mean |SHAP|):")
    for rank, (feat, imp) in enumerate(sorted_importance, 1):
        logger.info(f"[SHAP]   #{rank:02d} {feat}: {imp:.4f}")

    # 2. Directional impact
    mean_shap = sv.mean(axis=0)
    directional = dict(zip(available_features, mean_shap))
    sorted_directional = sorted(directional.items(), key=lambda x: abs(x[1]), reverse=True)

    logger.info("[SHAP] Directional impact (mean SHAP — positive = increases over probability):")
    for feat, val in sorted_directional[:15]:
        direction = "↑ over" if val > 0 else "↓ under"
        logger.info(f"[SHAP]   {feat}: {val:+.4f} ({direction})")

    # 3. L1/L2 cycle baseline deep dive
    l1_features = [f for f in available_features if f.startswith("p2s_")]
    l2_features = [f for f in available_features if f.startswith("p2sz_")]
    intersection_features = [f for f in available_features if "dual_optimal" in f or "signal_strength" in f]

    logger.info("[SHAP] L1 PTS feature SHAP values:")
    for feat in l1_features:
        if feat in feature_importance:
            logger.info(f"[SHAP]   {feat}: importance={feature_importance[feat]:.4f}, "
                        f"directional={directional.get(feat, 0):+.4f}")

    logger.info("[SHAP] L2 P2SZ feature SHAP values:")
    for feat in l2_features:
        if feat in feature_importance:
            logger.info(f"[SHAP]   {feat}: importance={feature_importance[feat]:.4f}, "
                        f"directional={directional.get(feat, 0):+.4f}")

    logger.info("[SHAP] Intersection signal SHAP values:")
    for feat in intersection_features:
        if feat in feature_importance:
            logger.info(f"[SHAP]   {feat}: importance={feature_importance[feat]:.4f}, "
                        f"directional={directional.get(feat, 0):+.4f}")

    # 4. Per-confidence-tier SHAP
    y_proba = model.predict_proba(X)[:, 1]
    confidences = np.maximum(y_proba, 1 - y_proba)

    tier_shap = {}
    for label, thresh in [("55+", 0.55), ("65+", 0.65), ("75+", 0.75)]:
        mask = confidences >= thresh
        if mask.sum() > 10:
            tier_sv = sv[mask]
            tier_importance = np.abs(tier_sv).mean(axis=0)
            tier_feat_imp = dict(zip(available_features, tier_importance))
            top5 = sorted(tier_feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
            tier_shap[label] = {k: round(float(v), 4) for k, v in top5}
            logger.info(f"[SHAP] Top features at confidence {label} (n={mask.sum()}):")
            for feat, imp in top5:
                logger.info(f"[SHAP]   {feat}: {imp:.4f}")

    total_elapsed = time.time() - total_start

    result = {
        "feature_importance": {k: round(float(v), 4) for k, v in sorted_importance},
        "directional_impact": {k: round(float(v), 4) for k, v in sorted_directional},
        "l1_features": {f: round(float(feature_importance.get(f, 0)), 4) for f in l1_features},
        "l2_features": {f: round(float(feature_importance.get(f, 0)), 4) for f in l2_features},
        "intersection_features": {f: round(float(feature_importance.get(f, 0)), 4) for f in intersection_features},
        "confidence_tier_shap": tier_shap,
        "samples_analyzed": len(X),
        "elapsed_seconds": round(total_elapsed, 1),
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info("[SHAP] ============================================================")
    logger.info(f"[SHAP] Analysis complete in {total_elapsed:.1f}s")
    logger.info("[SHAP] ============================================================")

    # Write to S3
    try:
        bucket = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
        key = f"shap/nfl_shap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(result, default=str))
        logger.info(f"[SHAP] Results saved to s3://{bucket}/{key}")
    except Exception as e:
        logger.warning(f"[SHAP] S3 write failed (non-fatal): {e}")

    return result