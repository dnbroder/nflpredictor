"""
NFL Predictor — Operations Lambda

Handles automated operational tasks triggered by EventBridge:
  1. RETRAIN   — Preseason model retrain on latest 6 seasons
  2. REFRESH   — Weekly data cache refresh (games, drives, PBP, closing lines)
  3. ACCURACY  — Post-game accuracy check, logs results to S3

Environment variables:
  ECS_CLUSTER:    nfl-predictor-cluster
  ECS_SERVICE:    nfl-predictor-service
  S3_BUCKET:      nfl-predictor-bucket
  NFL_SEASON:     2026
  CONTAINER_PORT: 8000
"""
import json
import logging
import os
from datetime import datetime

import boto3
import urllib3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=10, read=900))
s3 = boto3.client("s3")
ecs = boto3.client("ecs")
ec2 = boto3.client("ec2")

CLUSTER = os.environ.get("ECS_CLUSTER", "nfl-predictor-cluster")
SERVICE = os.environ.get("ECS_SERVICE", "nfl-predictor-service")
S3_BUCKET = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
NFL_SEASON = int(os.environ.get("NFL_SEASON", "2026"))
CONTAINER_PORT = int(os.environ.get("CONTAINER_PORT", "8000"))

# Production model config: 6-season rolling window, PIC history from 2016
TRAIN_WINDOW = 6
PIC_HISTORY_START = 2016


def _get_ecs_public_ip() -> str:
    """Resolve the current ECS task's public IP."""
    tasks = ecs.list_tasks(cluster=CLUSTER, serviceName=SERVICE)
    if not tasks.get("taskArns"):
        raise RuntimeError(f"No running tasks in {CLUSTER}/{SERVICE}")

    task_arn = tasks["taskArns"][0]
    task_detail = ecs.describe_tasks(cluster=CLUSTER, tasks=[task_arn])
    task = task_detail["tasks"][0]

    for attachment in task.get("attachments", []):
        for detail in attachment.get("details", []):
            if detail.get("name") == "networkInterfaceId":
                eni_id = detail["value"]
                eni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
                return eni["NetworkInterfaces"][0]["Association"]["PublicIp"]

    raise RuntimeError("Could not resolve ECS task public IP")


def _call_ecs(method: str, path: str, timeout: int = 900) -> dict:
    """Call an ECS endpoint and return parsed JSON response."""
    public_ip = _get_ecs_public_ip()
    url = f"http://{public_ip}:{CONTAINER_PORT}{path}"
    logger.info(f"Calling: {method} {url}")

    response = http.request(method, url, timeout=timeout)
    if response.status != 200:
        error_body = response.data.decode()
        logger.error(f"ECS call failed: {response.status} — {error_body}")
        raise RuntimeError(f"ECS returned {response.status}: {error_body}")

    return json.loads(response.data.decode())


# =========================================================================
# Action: RETRAIN
# =========================================================================
def _handle_retrain(event: dict) -> dict:
    """Retrain model on latest 6 seasons.
    
    Triggered annually in preseason (August 15).
    Uses train-async so the Lambda returns quickly while ECS processes.
    
    Training config:
      - train_start = NFL_SEASON - TRAIN_WINDOW (e.g., 2020 for 2026 season)
      - train_end = NFL_SEASON - 1 (e.g., 2025 — last completed season)
      - pic_history_start = 2016 (deep player cycle history)
    """
    season = event.get("season", NFL_SEASON)
    train_end = season - 1
    train_start = train_end - TRAIN_WINDOW + 1

    logger.info(f"RETRAIN: seasons {train_start}-{train_end}, PIC history from {PIC_HISTORY_START}")

    path = (f"/train-async"
            f"?train_start={train_start}"
            f"&train_end={train_end}"
            f"&pic_history_start={PIC_HISTORY_START}")

    result = _call_ecs("POST", path)
    logger.info(f"RETRAIN submitted: {json.dumps(result)}")

    # Log retrain event to S3
    retrain_log = {
        "action": "retrain",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "season": season,
        "train_start": train_start,
        "train_end": train_end,
        "pic_history_start": PIC_HISTORY_START,
        "ecs_response": result,
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"logs/retrain_{season}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        Body=json.dumps(retrain_log, indent=2),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Retrain submitted for seasons {train_start}-{train_end}",
            "train_start": train_start,
            "train_end": train_end,
        }),
    }


# =========================================================================
# Action: REFRESH
# =========================================================================
def _handle_refresh(event: dict) -> dict:
    """Refresh data caches for the current season.
    
    Triggered weekly (Wednesday morning) to ensure the latest game data,
    drive data, PBP, closing lines, EPA, and QB stats are cached in S3
    before the Saturday prediction run.
    
    Calls the /train endpoint with current season only — this forces
    the data pipeline to pull fresh data from nflverse and cache it.
    We don't actually retrain; we just warm the caches.
    """
    season = event.get("season", NFL_SEASON)
    logger.info(f"REFRESH: warming data caches for season {season}")

    # Call the schema endpoint to trigger a data load for the current season
    # This loads games, drives, PBP, closing lines, EPA, QB stats
    path = f"/schema?season={season}"
    result = _call_ecs("GET", path, timeout=300)
    logger.info(f"REFRESH schema response: {result.get('n_games', 0)} games")

    # Log refresh event
    refresh_log = {
        "action": "refresh",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "season": season,
        "games_found": result.get("n_games", 0),
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"logs/refresh_{season}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        Body=json.dumps(refresh_log, indent=2),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Data refresh complete for season {season}",
            "games": result.get("n_games", 0),
        }),
    }


# =========================================================================
# Action: ACCURACY
# =========================================================================
def _handle_accuracy(event: dict) -> dict:
    """Check prediction accuracy after games complete.
    
    Triggered weekly (Tuesday morning) to compare last week's predictions
    against actual results. Writes accuracy report to S3.
    
    Reads the predictions JSON from S3, then checks actual game scores
    via the ECS /schema endpoint to determine if over/under was correct.
    """
    season = event.get("season", NFL_SEASON)
    week = event.get("week")

    if week is None:
        logger.error("ACCURACY: week parameter required")
        return {"statusCode": 400, "body": "week parameter required"}

    logger.info(f"ACCURACY: checking season={season}, week={week}")

    # Load predictions from S3
    pred_key = f"predictions/nfl_{season}_week{week:02d}.json"
    try:
        pred_obj = s3.get_object(Bucket=S3_BUCKET, Key=pred_key)
        predictions = json.loads(pred_obj["Body"].read().decode())
    except s3.exceptions.NoSuchKey:
        logger.warning(f"ACCURACY: no predictions found at {pred_key}")
        return {"statusCode": 404, "body": f"No predictions for week {week}"}
    except Exception as e:
        logger.error(f"ACCURACY: failed to read predictions: {e}")
        return {"statusCode": 500, "body": str(e)}

    # Load actual game results from ECS — the schema endpoint returns
    # game data including actual scores for completed games
    path = f"/schema?season={season}"
    schema = _call_ecs("GET", path, timeout=300)

    # Build lookup of actual game totals from schema data
    # Schema returns games with home_score + away_score for completed games
    game_scores = {}
    for game in schema.get("games", []):
        gid = game.get("game_id", "")
        home_score = game.get("home_score")
        away_score = game.get("away_score")
        if home_score is not None and away_score is not None:
            game_scores[gid] = {
                "actual_total": home_score + away_score,
                "home_score": home_score,
                "away_score": away_score,
            }

    picks = predictions.get("predictions", [])
    if not picks:
        logger.info("ACCURACY: no picks to evaluate")
        return {"statusCode": 200, "body": "No picks to evaluate"}

    # Resolve each prediction against actual results
    wins = 0
    losses = 0
    unresolved = 0
    for pick in picks:
        game_id = pick.get("game_id", "")
        if game_id not in game_scores:
            pick["actual_result"] = None
            unresolved += 1
            continue

        actual_total = game_scores[game_id]["actual_total"]
        closing_total = pick.get("closing_total", 0)
        direction = pick.get("direction", "")

        if actual_total == closing_total:
            # Push — no result
            pick["actual_result"] = None
            pick["actual_total"] = actual_total
            pick["result_detail"] = "PUSH"
            unresolved += 1
        elif direction == "OVER":
            pick["actual_result"] = actual_total > closing_total
            pick["actual_total"] = actual_total
            pick["result_detail"] = f"{'WIN' if pick['actual_result'] else 'LOSS'}: {actual_total} vs {closing_total}"
        elif direction == "UNDER":
            pick["actual_result"] = actual_total < closing_total
            pick["actual_total"] = actual_total
            pick["result_detail"] = f"{'WIN' if pick['actual_result'] else 'LOSS'}: {actual_total} vs {closing_total}"
        else:
            pick["actual_result"] = None
            unresolved += 1

        if pick.get("actual_result") is True:
            wins += 1
        elif pick.get("actual_result") is False:
            losses += 1

    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else 0
    roi = round((wins * (100 / 110) - losses) / total * 100, 1) if total > 0 else 0

    logger.info(f"ACCURACY: week {week} — {wins}W-{losses}L ({win_rate}%), ROI {roi}%, {unresolved} unresolved")

    accuracy_report = {
        "action": "accuracy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "season": season,
        "week": week,
        "n_predictions": len(picks),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "roi": roi,
        "unresolved": unresolved,
        "status": "resolved" if unresolved == 0 else "partial",
        "predictions": picks,
    }

    report_key = f"accuracy/nfl_{season}_week{week:02d}.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=report_key,
        Body=json.dumps(accuracy_report, indent=2),
        ContentType="application/json",
    )
    logger.info(f"ACCURACY: wrote report to s3://{S3_BUCKET}/{report_key}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Accuracy report for week {week}",
            "n_predictions": len(picks),
            "s3_key": report_key,
        }),
    }


# =========================================================================
# Handler — routes by action type
# =========================================================================
def handler(event, context):
    """Lambda entry point. Routes based on 'action' field in event.
    
    EventBridge events:
      {"action": "retrain"}                    — annual preseason retrain
      {"action": "retrain", "season": 2027}    — explicit season
      {"action": "refresh"}                    — weekly data cache refresh
      {"action": "accuracy", "week": 5}        — post-game accuracy check
    """
    logger.info(f"Event: {json.dumps(event)}")

    action = event.get("action", "").lower()

    if action == "retrain":
        return _handle_retrain(event)
    elif action == "refresh":
        return _handle_refresh(event)
    elif action == "accuracy":
        return _handle_accuracy(event)
    else:
        logger.error(f"Unknown action: {action}")
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": f"Unknown action: {action}",
                "valid_actions": ["retrain", "refresh", "accuracy"],
            }),
        }