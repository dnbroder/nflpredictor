"""
NFLPredictor API — main.py
Entrypoint for uvicorn (main:app).
"""

import os
import time
import uuid
import threading
import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from nfl_predictor.__version__ import __version__ as VERSION
from nfl_predictor.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Imports — wrapped individually so any failure is immediately visible in logs
# ---------------------------------------------------------------------------
logger.info("[BOOT] Importing nfl_predictor modules...")

try:
    from nfl_predictor.nfl_data import load_games, load_drives, load_players, load_injuries, CURRENT_SEASON
    logger.info("[BOOT] nfl_data OK")
except Exception as e:
    logger.critical(f"[BOOT] FAILED to import nfl_data: {e}", exc_info=True)
    raise

try:
    from nfl_predictor.feature_engineering import engineer_features, FEATURE_COLUMNS, TARGET_COLUMN
    logger.info(f"[BOOT] feature_engineering OK — {len(FEATURE_COLUMNS)} features, target={TARGET_COLUMN}")
except Exception as e:
    logger.critical(f"[BOOT] FAILED to import feature_engineering: {e}", exc_info=True)
    raise

try:
    from nfl_predictor.model import train_model, tune_hyperparameters, load_model
    logger.info("[BOOT] model OK")
except Exception as e:
    logger.critical(f"[BOOT] FAILED to import model: {e}", exc_info=True)
    raise

try:
    from nfl_predictor.backtester import run_backtest
    logger.info("[BOOT] backtester OK")
except Exception as e:
    logger.critical(f"[BOOT] FAILED to import backtester: {e}", exc_info=True)
    raise

try:
    from nfl_predictor.shap_analysis import run_shap_analysis
    logger.info("[BOOT] shap_analysis OK")
except Exception as e:
    logger.critical(f"[BOOT] FAILED to import shap_analysis: {e}", exc_info=True)
    raise

logger.info("[BOOT] All imports successful")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
S3_BUCKET = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")

app = FastAPI(
    title="NFLPredictor API",
    version=VERSION,
    description=f"Four-Layer Cyclical P2S/P2SZ NFL prediction system. Environment: {ENVIRONMENT}",
)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = str(uuid.uuid4())[:8]
        start = time.time()
        logger.info(
            f"[REQ {req_id}] --> {request.method} {request.url.path}"
            + (f"?{request.url.query}" if request.url.query else "")
        )
        try:
            response = await call_next(request)
            elapsed = (time.time() - start) * 1000
            logger.info(f"[REQ {req_id}] <-- {response.status_code} ({elapsed:.0f}ms)")
            return response
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"[REQ {req_id}] !!! EXCEPTION ({elapsed:.0f}ms): {e}", exc_info=True)
            raise

app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info(f"[STARTUP] NFLPredictor API v{VERSION}")
    logger.info(f"[STARTUP] Environment: {ENVIRONMENT}")
    logger.info(f"[STARTUP] S3 bucket:   {S3_BUCKET}")
    logger.info(f"[STARTUP] Target:      {TARGET_COLUMN}")
    logger.info(f"[STARTUP] Features:    {len(FEATURE_COLUMNS)}")
    logger.info(f"[STARTUP] Season:      {CURRENT_SEASON}")
    logger.info("=" * 60)
    model, features = load_model()
    if model is not None:
        logger.info(f"[STARTUP] Model loaded OK — {len(features)} features")
    else:
        logger.warning("[STARTUP] No model found. Call POST /train before predicting.")
    logger.info("[STARTUP] API ready")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "app": "NFLPredictor",
        "version": VERSION,
        "environment": ENVIRONMENT,
        "methodology": "Four-Layer Cyclical — P2S / P2SZ / PIC / PIM",
        "target": TARGET_COLUMN,
        "features": len(FEATURE_COLUMNS),
        "endpoints": {
            "GET /":                    "This page",
            "GET /health":              "Health check + model status",
            "GET /schema":              "Inspect data schema for a season",
            "POST /train":              "Train model synchronously",
            "POST /backtest":           "Full train + backtest (synchronous)",
            "POST /backtest-async":     "Full train + backtest (background)",
            "POST /tune-async":         "Hyperparameter tuning (background)",
            "POST /shap-async":         "SHAP feature analysis (background)",
            "GET /predict/week/{week}": "Predictions for a given week",
        },
    }


@app.get("/health")
def health():
    model, features = load_model()
    model_loaded = model is not None
    return {
        "status": "ok",
        "version": VERSION,
        "environment": ENVIRONMENT,
        "model_loaded": model_loaded,
        "target": TARGET_COLUMN if model_loaded else None,
        "features_count": len(features) if model_loaded else None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/schema")
def schema(season: int = Query(default=CURRENT_SEASON)):
    logger.info(f"[SCHEMA] Loading schema for season {season}")
    t0 = time.time()
    try:
        games = load_games([season])
        logger.info(f"[SCHEMA] Games: {len(games)} rows")
        drives = load_drives([season])
        logger.info(f"[SCHEMA] Drives: {len(drives)} rows")
        result = _sanitize({
            "season": season,
            "games": len(games),
            "drives": len(drives),
            "game_columns": list(games.columns),
            "drive_columns": list(drives.columns),
            "drive_result_distribution": (
                drives["drive_result"].value_counts().to_dict()
                if "drive_result" in drives.columns else {}
            ),
            "games_with_total_line": int(games["total_line"].notna().sum()) if "total_line" in games.columns else 0,
            "elapsed_seconds": round(time.time() - t0, 1),
        })
        logger.info(f"[SCHEMA] Done in {result['elapsed_seconds']}s")
        return result
    except Exception as e:
        logger.error(f"[SCHEMA] Failed: {e}", exc_info=True)
        return JSONResponse(status_code=503, content={"error": str(e)})


@app.post("/train")
def train(
    train_start: int = Query(default=2016),
    train_end: int = Query(default=2022),
):
    logger.info(f"[TRAIN] ========================================")
    logger.info(f"[TRAIN] train_start={train_start}, train_end={train_end}")
    t0 = time.time()
    try:
        seasons = list(range(train_start, train_end + 1))

        t1 = time.time()
        logger.info(f"[TRAIN] Loading games for {seasons}...")
        games = load_games(seasons)
        logger.info(f"[TRAIN] Games: {len(games)} rows in {time.time()-t1:.1f}s")

        t2 = time.time()
        logger.info(f"[TRAIN] Loading drives...")
        drives = load_drives(seasons)
        logger.info(f"[TRAIN] Drives: {len(drives)} rows in {time.time()-t2:.1f}s")

        t3 = time.time()
        logger.info(f"[TRAIN] Engineering features...")
        featured = engineer_features(games, drives)
        logger.info(f"[TRAIN] Features: {len(featured)} rows in {time.time()-t3:.1f}s")

        logger.info(f"[TRAIN] Training model...")
        model, metrics = train_model(featured)

        elapsed = time.time() - t0
        logger.info(f"[TRAIN] Complete in {elapsed:.1f}s — AUC={metrics.get('auc_roc')}, "
                    f"Accuracy={metrics.get('accuracy')}")
        return _sanitize({"status": "trained", "elapsed_seconds": round(elapsed, 1), "metrics": metrics})
    except Exception as e:
        logger.error(f"[TRAIN] FAILED after {time.time()-t0:.1f}s: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/backtest")
def backtest(
    train_start: int = Query(default=2016),
    train_end: int = Query(default=2022),
    test_year: int = Query(default=2023),
):
    """
    Full train + backtest. Model is NEVER trained on test_year data.
    """
    logger.info(f"[BACKTEST] ========================================")
    logger.info(f"[BACKTEST] train={train_start}-{train_end}, test={test_year}")
    t0 = time.time()
    try:
        train_seasons = list(range(train_start, train_end + 1))

        logger.info(f"[BACKTEST] Step 1/4: Loading train data {train_seasons}...")
        train_games = load_games(train_seasons)
        train_drives = load_drives(train_seasons)
        logger.info(f"[BACKTEST] Train: {len(train_games)} games, {len(train_drives)} drives")

        logger.info(f"[BACKTEST] Step 2/4: Engineering train features...")
        train_featured = engineer_features(train_games, train_drives)
        logger.info(f"[BACKTEST] Train features: {len(train_featured)} rows")

        logger.info(f"[BACKTEST] Step 3/4: Training model on {len(train_featured)} rows...")
        model, metrics = train_model(train_featured)
        logger.info(f"[BACKTEST] Model trained — AUC={metrics.get('auc_roc')}")

        logger.info(f"[BACKTEST] Step 4/4: Loading test season {test_year}...")
        test_games = load_games([test_year])
        all_drives = load_drives(train_seasons + [test_year])
        full_featured = engineer_features(
            pd.concat([train_games, test_games], ignore_index=True),
            all_drives,
        )
        logger.info(f"[BACKTEST] Running backtest on {test_year}...")
        results = run_backtest(full_featured, train_start=train_start, train_end=train_end, test_year=test_year)
        results["train_metrics"] = metrics

        elapsed = time.time() - t0
        logger.info(f"[BACKTEST] Complete in {elapsed:.1f}s — "
                    f"accuracy={results.get('overall_accuracy')}, "
                    f"above_breakeven={results.get('above_breakeven')}")
        return _sanitize(results)
    except Exception as e:
        logger.error(f"[BACKTEST] FAILED after {time.time()-t0:.1f}s: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/backtest-async")
def backtest_async(
    train_start: int = Query(default=2016),
    train_end: int = Query(default=2022),
    test_year: int = Query(default=2023),
):
    """Background train + backtest. Monitor via CloudWatch logs."""
    job_id = str(uuid.uuid4())[:8]

    def _run():
        t0 = time.time()
        logger.info(f"[ASYNC-BACKTEST {job_id}] ========================================")
        logger.info(f"[ASYNC-BACKTEST {job_id}] train={train_start}-{train_end}, test={test_year}")
        try:
            train_seasons = list(range(train_start, train_end + 1))

            logger.info(f"[ASYNC-BACKTEST {job_id}] Loading train games...")
            train_games = load_games(train_seasons)
            logger.info(f"[ASYNC-BACKTEST {job_id}] {len(train_games)} games")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Loading train drives...")
            train_drives = load_drives(train_seasons)
            logger.info(f"[ASYNC-BACKTEST {job_id}] {len(train_drives)} drives")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Engineering train features...")
            train_featured = engineer_features(train_games, train_drives)
            logger.info(f"[ASYNC-BACKTEST {job_id}] {len(train_featured)} feature rows")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Training model...")
            model, metrics = train_model(train_featured)
            logger.info(f"[ASYNC-BACKTEST {job_id}] Model trained — AUC={metrics.get('auc_roc')}")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Loading test season {test_year}...")
            test_games = load_games([test_year])
            all_drives = load_drives(train_seasons + [test_year])
            full_featured = engineer_features(
                pd.concat([train_games, test_games], ignore_index=True),
                all_drives,
            )
            logger.info(f"[ASYNC-BACKTEST {job_id}] Running backtest...")
            results = run_backtest(full_featured, train_start=train_start, train_end=train_end, test_year=test_year)

            elapsed = time.time() - t0
            logger.info(f"[ASYNC-BACKTEST {job_id}] ========================================")
            logger.info(f"[ASYNC-BACKTEST {job_id}] COMPLETE in {elapsed:.1f}s")
            logger.info(f"[ASYNC-BACKTEST {job_id}] accuracy={results.get('overall_accuracy')}, "
                        f"above_breakeven={results.get('above_breakeven')}")
            for label, data in results.get("confidence_tiers", {}).items():
                logger.info(f"[ASYNC-BACKTEST {job_id}] Tier {label}: "
                            f"{data.get('wins')}/{data.get('total')} "
                            f"acc={data.get('accuracy',0)*100:.1f}% "
                            f"ROI={data.get('roi',0):+.1%}")
            logger.info(f"[ASYNC-BACKTEST {job_id}] ========================================")
        except Exception as e:
            logger.error(f"[ASYNC-BACKTEST {job_id}] FAILED after {time.time()-t0:.1f}s: {e}", exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name=f"backtest-{job_id}")
    thread.start()
    logger.info(f"[ASYNC-BACKTEST {job_id}] Thread started")
    return {
        "status": "started",
        "job_id": job_id,
        "train_start": train_start,
        "train_end": train_end,
        "test_year": test_year,
        "message": f"Job {job_id} running. Monitor CloudWatch /ecs/nfl-predictor for progress.",
    }


@app.post("/tune-async")
def tune_async(
    train_start: int = Query(default=2016),
    train_end: int = Query(default=2022),
    n_iter: int = Query(default=50),
):
    """Hyperparameter tuning in background thread."""
    job_id = str(uuid.uuid4())[:8]

    def _run():
        t0 = time.time()
        logger.info(f"[ASYNC-TUNE {job_id}] Starting: train={train_start}-{train_end}, n_iter={n_iter}")
        try:
            seasons = list(range(train_start, train_end + 1))
            games = load_games(seasons)
            drives = load_drives(seasons)
            featured = engineer_features(games, drives)
            logger.info(f"[ASYNC-TUNE {job_id}] {len(featured)} rows — starting {n_iter} iterations...")
            results = tune_hyperparameters(featured, n_iter=n_iter)
            elapsed = time.time() - t0
            logger.info(f"[ASYNC-TUNE {job_id}] Complete in {elapsed:.1f}s — "
                        f"best={results['best']['mean_accuracy']:.4f} "
                        f"improvement={results['best']['improvement']:+.4f}")
        except Exception as e:
            logger.error(f"[ASYNC-TUNE {job_id}] FAILED: {e}", exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name=f"tune-{job_id}")
    thread.start()
    logger.info(f"[ASYNC-TUNE {job_id}] Thread started")
    return {"status": "started", "job_id": job_id, "n_iter": n_iter}


@app.post("/shap-async")
def shap_async(
    train_start: int = Query(default=2016),
    train_end: int = Query(default=2022),
):
    """SHAP feature analysis in background thread."""
    job_id = str(uuid.uuid4())[:8]

    def _run():
        t0 = time.time()
        logger.info(f"[ASYNC-SHAP {job_id}] Starting: seasons={train_start}-{train_end}")
        try:
            seasons = list(range(train_start, train_end + 1))
            games = load_games(seasons)
            drives = load_drives(seasons)
            featured = engineer_features(games, drives)
            logger.info(f"[ASYNC-SHAP {job_id}] {len(featured)} rows — running SHAP...")
            run_shap_analysis(featured)
            elapsed = time.time() - t0
            logger.info(f"[ASYNC-SHAP {job_id}] Complete in {elapsed:.1f}s — results written to S3")
        except Exception as e:
            logger.error(f"[ASYNC-SHAP {job_id}] FAILED: {e}", exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name=f"shap-{job_id}")
    thread.start()
    logger.info(f"[ASYNC-SHAP {job_id}] Thread started")
    return {"status": "started", "job_id": job_id}


@app.get("/predict/week/{week}")
def predict_week(
    week: int,
    season: int = Query(default=CURRENT_SEASON),
    min_confidence: float = Query(default=0.0),
):
    """All game predictions for a given week, sorted by confidence."""
    logger.info(f"[PREDICT] week={week}, season={season}, min_confidence={min_confidence}")
    t0 = time.time()
    try:
        model, feature_cols = load_model()
        if model is None:
            logger.warning("[PREDICT] No model available")
            return JSONResponse(status_code=404, content={"error": "No model. Run /train first."})
        logger.info(f"[PREDICT] Model loaded ({len(feature_cols)} features)")

        games = load_games([season])
        week_games = games[games["week"] == week]
        logger.info(f"[PREDICT] Week {week}: {len(week_games)} games")

        if week_games.empty:
            logger.warning(f"[PREDICT] No games found for week={week} season={season}")
            return {"week": week, "season": season, "predictions": []}

        drives = load_drives([season])
        featured = engineer_features(week_games, drives)

        missing_cols = [f for f in feature_cols if f not in featured.columns]
        if missing_cols:
            logger.warning(f"[PREDICT] {len(missing_cols)} missing features filled with 0: {missing_cols}")
            for col in missing_cols:
                featured[col] = 0.0

        available = [f for f in feature_cols if f in featured.columns]
        X = featured[available]
        logger.info(f"[PREDICT] Predicting {len(X)} rows with {len(available)} features...")

        y_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        confidences = np.maximum(y_proba, 1 - y_proba)

        featured = featured.copy()
        featured["probability"] = y_proba
        featured["confidence"] = confidences
        featured["prediction"] = y_pred

        filtered = featured[featured["confidence"] >= min_confidence].sort_values(
            "confidence", ascending=False
        )

        results = []
        for _, row in filtered.iterrows():
            results.append(_sanitize({
                "team":          row.get("team"),
                "opponent":      row.get("opponent"),
                "is_home":       bool(row.get("is_home", 0)),
                "prediction":    int(row["prediction"]),
                "probability":   round(float(row["probability"]), 4),
                "confidence":    round(float(row["confidence"]), 4),
                "dual_optimal":  bool(row.get("p2s_p2sz_dual_optimal", 0) == 1.0),
                "p2s_ratio":     round(float(row.get("p2s_ratio") or 0), 3),
                "p2sz_sp_ratio": round(float(row.get("p2sz_sp_ratio") or 0), 3),
            }))

        elapsed = time.time() - t0
        above_55 = sum(1 for r in results if r["confidence"] >= 0.55)
        dual_opt = sum(1 for r in results if r["dual_optimal"])
        logger.info(f"[PREDICT] {len(results)} predictions in {elapsed:.1f}s — "
                    f"{above_55} at 55%+ confidence, {dual_opt} dual optimal")

        return {
            "week": week,
            "season": season,
            "target": TARGET_COLUMN,
            "n_games": len(results),
            "high_confidence": above_55,
            "dual_optimal": dual_opt,
            "elapsed_seconds": round(elapsed, 1),
            "predictions": results,
        }
    except Exception as e:
        logger.error(f"[PREDICT] FAILED: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})