"""
NFL Predictor — Weekly Predictions Lambda

Triggered by EventBridge schedule. Calls the ECS /predict/week/{week} endpoint,
writes structured JSON to S3 for website API consumption.

Environment variables:
  ECS_SERVICE_URL: http://<ecs-public-ip>:8000  (or ALB DNS)
  S3_BUCKET: nfl-predictor-bucket
  NFL_SEASON: 2026
"""
import json
import logging
import os
from datetime import datetime, timedelta

import boto3
import urllib3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=10, read=600))
s3 = boto3.client("s3")
ecs = boto3.client("ecs")
ec2 = boto3.client("ec2")

CLUSTER = os.environ.get("ECS_CLUSTER", "nfl-predictor-cluster")
SERVICE = os.environ.get("ECS_SERVICE", "nfl-predictor-service")
S3_BUCKET = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
NFL_SEASON = int(os.environ.get("NFL_SEASON", "2026"))
CONTAINER_PORT = int(os.environ.get("CONTAINER_PORT", "8000"))


def _get_ecs_public_ip() -> str:
    """Resolve the current ECS task's public IP."""
    tasks = ecs.list_tasks(cluster=CLUSTER, serviceName=SERVICE)
    if not tasks.get("taskArns"):
        raise RuntimeError(f"No running tasks in {CLUSTER}/{SERVICE}")

    task_arn = tasks["taskArns"][0]
    task_detail = ecs.describe_tasks(cluster=CLUSTER, tasks=[task_arn])
    task = task_detail["tasks"][0]

    # Find the ENI attachment
    for attachment in task.get("attachments", []):
        for detail in attachment.get("details", []):
            if detail.get("name") == "networkInterfaceId":
                eni_id = detail["value"]
                eni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
                return eni["NetworkInterfaces"][0]["Association"]["PublicIp"]

    raise RuntimeError("Could not resolve ECS task public IP")


def _get_current_nfl_week() -> int:
    """Estimate current NFL week based on date.
    
    NFL season typically starts first Thursday after Labor Day.
    This is approximate — production should use the nflverse schedule API.
    """
    # For now, use a simple mapping or pass via event
    # Default to week 1 if we can't determine
    return 1


def handler(event, context):
    """Lambda handler — triggered by EventBridge.
    
    Event can include:
      {"week": 5, "season": 2026}  — explicit week
      {"slate": "TNF"}             — Thursday Night Football (upcoming TNF game)
      {"slate": "SUNDAY"}          — Full Sunday + Monday slate
      {}                           — auto-detect current week
    """
    logger.info(f"Event: {json.dumps(event)}")

    season = event.get("season", NFL_SEASON)
    week = event.get("week")
    slate = event.get("slate", "ALL")
    min_confidence = event.get("min_confidence", 0.52)

    if week is None:
        week = _get_current_nfl_week()
        logger.info(f"Auto-detected week: {week}")

    logger.info(f"Generating predictions: season={season}, week={week}, slate={slate}")

    # Resolve ECS endpoint
    try:
        public_ip = _get_ecs_public_ip()
        base_url = f"http://{public_ip}:{CONTAINER_PORT}"
        logger.info(f"ECS endpoint: {base_url}")
    except Exception as e:
        logger.error(f"Failed to resolve ECS IP: {e}")
        return {"statusCode": 500, "body": f"ECS resolution failed: {e}"}

    # Call predict endpoint
    predict_url = f"{base_url}/predict/week/{week}?season={season}&min_confidence={min_confidence}"
    logger.info(f"Calling: {predict_url}")

    try:
        response = http.request("GET", predict_url)
        if response.status != 200:
            logger.error(f"Predict failed: {response.status} — {response.data.decode()}")
            return {"statusCode": response.status, "body": response.data.decode()}

        predictions = json.loads(response.data.decode())
        logger.info(f"Got {predictions.get('n_predictions', 0)} predictions")
    except Exception as e:
        logger.error(f"HTTP call failed: {e}")
        return {"statusCode": 500, "body": str(e)}

    # Enrich with metadata
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": predictions.get("model_version", "unknown"),
        "season": season,
        "week": week,
        "slate": slate,
        "n_predictions": predictions.get("n_predictions", 0),
        "conviction_picks": predictions.get("conviction_picks", 0),
        "high_picks": predictions.get("high_picks", 0),
        "predictions": predictions.get("predictions", []),
    }

    # Write to S3
    s3_key = f"predictions/nfl_{season}_week{week:02d}.json"
    s3_key_latest = "predictions/nfl_latest.json"

    try:
        body = json.dumps(output, indent=2)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=body,
            ContentType="application/json",
        )
        # Also write as "latest" for easy website consumption
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key_latest,
            Body=body,
            ContentType="application/json",
        )
        logger.info(f"Wrote predictions to s3://{S3_BUCKET}/{s3_key}")
        logger.info(f"Wrote latest to s3://{S3_BUCKET}/{s3_key_latest}")
    except Exception as e:
        logger.error(f"S3 write failed: {e}")
        return {"statusCode": 500, "body": f"S3 write failed: {e}"}

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Week {week} predictions generated",
            "s3_key": s3_key,
            "n_predictions": output["n_predictions"],
            "conviction_picks": output["conviction_picks"],
        }),
    }