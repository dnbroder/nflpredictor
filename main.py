"""
NFLPredictor API — main.py
Entrypoint for uvicorn (main:app).
"""

import os
import resource
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
# Memory monitoring — prevents silent OOM kills
# ---------------------------------------------------------------------------
def _get_memory_mb() -> float:
    """Get current process RSS memory in MB."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _log_memory(label: str = "") -> float:
    """Log memory and return MB value."""
    mem = _get_memory_mb()
    prefix = f"[MEMORY] {label}: " if label else "[MEMORY] "
    if mem >= 7000:  # 7GB of 8GB = critical
        logger.error(f"{prefix}{mem:.1f}MB RSS — CRITICAL (OOM imminent)")
    elif mem >= 5500:  # ~70% of 8GB
        logger.warning(f"{prefix}{mem:.1f}MB RSS — WARNING")
    else:
        logger.info(f"{prefix}{mem:.1f}MB RSS")
    return mem

# ---------------------------------------------------------------------------
# Imports — wrapped individually so any failure is immediately visible in logs
# ---------------------------------------------------------------------------
logger.info("[BOOT] Importing nfl_predictor modules...")

try:
    from nfl_predictor.nfl_data import (
        load_games, load_drives, load_closing_lines,
        load_epa_by_game, load_qb_stats, load_plays,
        load_players, CURRENT_SEASON,
    )
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
    _log_memory("startup_begin")
    logger.info("=" * 60)
    logger.info(f"[STARTUP] NFLPredictor API v{VERSION}")
    logger.info(f"[STARTUP] Environment: {ENVIRONMENT}")
    logger.info(f"[STARTUP] S3 bucket:   {S3_BUCKET}")
    logger.info(f"[STARTUP] Target:      {TARGET_COLUMN}")
    logger.info(f"[STARTUP] Features:    {len(FEATURE_COLUMNS)}")
    logger.info(f"[STARTUP] Season:      {CURRENT_SEASON}")
    logger.info(f"[STARTUP] Memory:      8192MB allocated")
    logger.info("=" * 60)
    try:
        result = load_model()
        model = result[0] if result else None
        features = result[1] if result and len(result) > 1 else None
        calibrator = result[2] if result and len(result) > 2 else None
        if model is not None:
            cal_status = "with calibrator" if calibrator else "no calibrator"
            logger.info(f"[STARTUP] Model loaded OK — {len(features) if features else '?'} features, {cal_status}")
        else:
            logger.warning("[STARTUP] No model found. Call POST /train before predicting.")
    except Exception as e:
        logger.warning(f"[STARTUP] Model load failed (non-fatal): {e}")
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
    mem = _get_memory_mb()
    return {
        "status": "ok",
        "version": VERSION,
        "environment": ENVIRONMENT,
        "memory_mb": round(mem, 1),
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


@app.post("/train-async")
def train_async(
    train_start: int = Query(default=2020),
    train_end: int = Query(default=2022),
    pic_history_start: int = Query(default=2016),
):
    """
    Train model in background thread. Returns immediately.

    pic_history_start: earliest season for PIC cycle history (default 2016).
    Player cycle baselines are computed from pic_history_start→train_end,
    but the feature matrix only includes train_start→train_end rows.
    This solves the cold start problem — players have deep cycle history
    even when training on fewer seasons.
    """
    job_id = str(uuid.uuid4())[:8]

    def _run():
        t0 = time.time()
        logger.info(f"[ASYNC-TRAIN {job_id}] ========================================")
        logger.info(f"[ASYNC-TRAIN {job_id}] train_start={train_start}, train_end={train_end}, pic_history_start={pic_history_start}")
        try:
            seasons = list(range(train_start, train_end + 1))
            pic_seasons = list(range(pic_history_start, train_end + 1))

            logger.info(f"[ASYNC-TRAIN {job_id}] Loading games for {seasons}...")
            games = load_games(seasons)
            logger.info(f"[ASYNC-TRAIN {job_id}] Games: {len(games)} rows")

            logger.info(f"[ASYNC-TRAIN {job_id}] Loading drives...")
            drives = load_drives(seasons)
            logger.info(f"[ASYNC-TRAIN {job_id}] Drives: {len(drives)} rows")

            logger.info(f"[ASYNC-TRAIN {job_id}] Loading closing lines...")
            closing_lines = load_closing_lines(seasons)
            logger.info(f"[ASYNC-TRAIN {job_id}] Closing lines: {len(closing_lines)} games")

            logger.info(f"[ASYNC-TRAIN {job_id}] Loading EPA by game...")
            epa = load_epa_by_game(seasons)
            logger.info(f"[ASYNC-TRAIN {job_id}] EPA: {len(epa)} team-game rows")

            logger.info(f"[ASYNC-TRAIN {job_id}] Loading QB stats...")
            qb = load_qb_stats(seasons)
            logger.info(f"[ASYNC-TRAIN {job_id}] QB stats: {len(qb)} team-game rows")

            # PBP for PIC uses extended history (pic_history_start) to solve cold start.
            # Players need 8+ drives of history — using full history means Week 1 of
            # 2020 has cycle baselines from 2016-2019 rather than starting from scratch.
            logger.info(f"[ASYNC-TRAIN {job_id}] Loading PBP for Layer 3 PIC (history: {pic_seasons[0]}-{pic_seasons[-1]})...")
            pbp = load_plays(pic_seasons, cache=True)
            logger.info(f"[ASYNC-TRAIN {job_id}] PBP: {len(pbp)} plays ({len(pic_seasons)} seasons of history)")

            logger.info(f"[ASYNC-TRAIN {job_id}] Loading player identity...")
            players = load_players(cache=True)
            logger.info(f"[ASYNC-TRAIN {job_id}] Players: {len(players)} records")

            logger.info(f"[ASYNC-TRAIN {job_id}] Engineering features...")
            featured = engineer_features(
                games, drives,
                pbp_df=pbp,
                players_df=players,
                closing_lines_df=closing_lines,
                epa_df=epa,
                qb_df=qb,
            )
            logger.info(f"[ASYNC-TRAIN {job_id}] Features: {len(featured)} rows")

            # Diagnostic: verify closing_total survived the full pipeline before hitting model
            if "closing_total" in featured.columns:
                ct_valid = featured["closing_total"].notna().sum()
                ct_sample = featured["closing_total"].dropna().head(3).tolist()
                logger.info(f"[ASYNC-TRAIN {job_id}] closing_total handoff: {ct_valid}/{len(featured)} valid, sample={ct_sample}")
            else:
                logger.error(f"[ASYNC-TRAIN {job_id}] closing_total MISSING from featured df — column not present")

            logger.info(f"[ASYNC-TRAIN {job_id}] Training model...")
            model, metrics = train_model(featured)

            elapsed = time.time() - t0
            logger.info(f"[ASYNC-TRAIN {job_id}] ========================================")
            logger.info(f"[ASYNC-TRAIN {job_id}] COMPLETE in {elapsed:.1f}s — "
                        f"AUC={metrics.get('auc_roc')}, Accuracy={metrics.get('accuracy')}")
            logger.info(f"[ASYNC-TRAIN {job_id}] ========================================")
        except Exception as e:
            logger.error(f"[ASYNC-TRAIN {job_id}] FAILED: {e}", exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name=f"train-{job_id}")
    thread.start()
    logger.info(f"[ASYNC-TRAIN {job_id}] Thread started")
    return {
        "status": "started",
        "job_id": job_id,
        "train_start": train_start,
        "train_end": train_end,
        "message": f"Job {job_id} running. Monitor CloudWatch /ecs/nfl-predictor for progress.",
    }


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

        logger.info(f"[BACKTEST] Step 1/4: Loading all data for {train_seasons} + [{test_year}]...")
        all_seasons = train_seasons + [test_year]
        all_games = load_games(all_seasons)
        all_drives = load_drives(all_seasons)
        pbp = load_plays(all_seasons, cache=True)
        players = load_players(cache=True)
        logger.info(f"[BACKTEST] Data: {len(all_games)} games, {len(all_drives)} drives, "
                    f"{len(pbp)} plays, {len(players)} players")

        logger.info(f"[BACKTEST] Step 2/4: Engineering features (L1+L2+L3)...")
        full_featured = engineer_features(
            all_games, all_drives,
            pbp_df=pbp, players_df=players,
        )
        logger.info(f"[BACKTEST] Features: {len(full_featured)} rows")

        logger.info(f"[BACKTEST] Step 3/4: Running walk-forward backtest...")
        results = run_backtest(full_featured, train_seasons=train_end - train_start + 1)
        results["layers"] = "L1+L2+L3"

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
        _log_memory(f"backtest_{job_id}_start")
        try:
            train_seasons = list(range(train_start, train_end + 1))
            all_seasons = train_seasons + [test_year]

            logger.info(f"[ASYNC-BACKTEST {job_id}] Loading games...")
            all_games = load_games(all_seasons)
            logger.info(f"[ASYNC-BACKTEST {job_id}] {len(all_games)} games")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Loading drives...")
            all_drives = load_drives(all_seasons)
            logger.info(f"[ASYNC-BACKTEST {job_id}] {len(all_drives)} drives")
            _log_memory(f"backtest_{job_id}_after_drives")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Loading PBP for Layer 3 PIC...")
            pbp = load_plays(all_seasons, cache=True)
            logger.info(f"[ASYNC-BACKTEST {job_id}] {len(pbp)} plays")
            _log_memory(f"backtest_{job_id}_after_pbp")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Loading player identity...")
            players = load_players(cache=True)
            logger.info(f"[ASYNC-BACKTEST {job_id}] {len(players)} players")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Engineering features (L1+L2+L3)...")
            full_featured = engineer_features(
                all_games, all_drives,
                pbp_df=pbp, players_df=players,
            )
            logger.info(f"[ASYNC-BACKTEST {job_id}] {len(full_featured)} feature rows")
            _log_memory(f"backtest_{job_id}_after_features")

            # Free PBP data — no longer needed after feature engineering
            del pbp
            _log_memory(f"backtest_{job_id}_pbp_freed")

            logger.info(f"[ASYNC-BACKTEST {job_id}] Running walk-forward backtest...")
            results = run_backtest(full_featured, train_seasons=train_end - train_start + 1)

            elapsed = time.time() - t0
            mem = _log_memory(f"backtest_{job_id}_complete")
            summary = results.get("summary", {})
            logger.info(f"[ASYNC-BACKTEST {job_id}] ========================================")
            logger.info(f"[ASYNC-BACKTEST {job_id}] COMPLETE in {elapsed:.1f}s | peak_mem={mem:.0f}MB")
            logger.info(f"[ASYNC-BACKTEST {job_id}] Win rate: {summary.get('win_rate', 0):.4f} "
                        f"(break-even: 0.5238)")
            logger.info(f"[ASYNC-BACKTEST {job_id}] ROI: {summary.get('overall_roi_pct', 0):+.2f}%")
            logger.info(f"[ASYNC-BACKTEST {job_id}] Total bets: {summary.get('total_bets', 0)}")
            for tier, data in summary.get("confidence_tiers", {}).items():
                if data.get("n", 0) >= 10:
                    logger.info(f"[ASYNC-BACKTEST {job_id}]   P>={tier}: "
                                f"win_rate={data['win_rate']:.4f}, "
                                f"ROI={data['roi_pct']:+.1f}%, n={data['n']}")
            logger.info(f"[ASYNC-BACKTEST {job_id}] ========================================")
        except Exception as e:
            _log_memory(f"backtest_{job_id}_failed")
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
        model, feature_cols, calibrator = load_model()
        if model is None:
            logger.warning("[PREDICT] No model available")
            return JSONResponse(status_code=404, content={"error": "No model. Run /train first."})
        cal_status = "with calibrator" if calibrator else "raw probabilities"
        logger.info(f"[PREDICT] Model loaded ({len(feature_cols)} features, {cal_status})")

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

        raw_proba = model.predict_proba(X)[:, 1]
        if calibrator is not None:
            y_proba = calibrator.predict(raw_proba)
            logger.info(f"[PREDICT] Calibrated — raw mean={raw_proba.mean():.3f}, "
                       f"calibrated mean={y_proba.mean():.3f}")
        else:
            y_proba = raw_proba
        y_pred = (y_proba >= 0.5).astype(int)
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