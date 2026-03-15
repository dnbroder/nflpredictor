#!/usr/bin/env python3
"""
Patch an ECS task definition JSON before deployment.
Ensures required secrets and environment variables are present
without overwriting any existing values.

Usage:
    python3 nfl_predictor/patch_task_definition.py task-definition.json
"""

import json
import sys

REQUIRED_SECRETS = [
    {
        "name": "ODDS_API_KEY",
        # Full ARN including Secrets Manager suffix — must match exactly
        "valueFrom": "arn:aws:secretsmanager:us-east-1:033896355668:secret:odds-api-key-iBgMJ0",
    },
]

REQUIRED_ENV = [
    {"name": "LOG_LEVEL", "value": "INFO"},
]

# Health check with generous startPeriod — container imports heavy ML libs on boot
HEALTH_CHECK = {
    "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
    "interval": 30,
    "timeout": 10,
    "retries": 3,
    "startPeriod": 120,
}


def patch(path: str) -> None:
    with open(path) as f:
        td = json.load(f)

    cd = td["containerDefinitions"][0]

    # Patch secrets — replace if name matches (handles ARN suffix changes)
    secrets = cd.get("secrets", [])
    existing = {s["name"]: i for i, s in enumerate(secrets)}
    added_secrets = []
    updated_secrets = []

    for secret in REQUIRED_SECRETS:
        if secret["name"] in existing:
            idx = existing[secret["name"]]
            if secrets[idx]["valueFrom"] != secret["valueFrom"]:
                secrets[idx] = secret
                updated_secrets.append(secret["name"])
        else:
            secrets.append(secret)
            added_secrets.append(secret["name"])

    cd["secrets"] = secrets

    # Patch environment — replace if name matches
    env = cd.get("environment", [])
    existing_env = {e["name"]: i for i, e in enumerate(env)}
    added_env = []
    updated_env = []

    for var in REQUIRED_ENV:
        if var["name"] in existing_env:
            idx = existing_env[var["name"]]
            if env[idx]["value"] != var["value"]:
                env[idx] = var
                updated_env.append(var["name"])
        else:
            env.append(var)
            added_env.append(var["name"])

    cd["environment"] = env

    # Patch health check — always enforce our settings
    existing_hc = cd.get("healthCheck", {})
    if existing_hc != HEALTH_CHECK:
        cd["healthCheck"] = HEALTH_CHECK
        print(f"Updated healthCheck: startPeriod={HEALTH_CHECK['startPeriod']}s, timeout={HEALTH_CHECK['timeout']}s")

    with open(path, "w") as f:
        json.dump(td, f, indent=2)

    if added_secrets:
        print(f"Added secrets:    {', '.join(added_secrets)}")
    if updated_secrets:
        print(f"Updated secrets:  {', '.join(updated_secrets)}")
    if added_env:
        print(f"Added env vars:   {', '.join(added_env)}")
    if updated_env:
        print(f"Updated env vars: {', '.join(updated_env)}")
    if not any([added_secrets, updated_secrets, added_env, updated_env]):
        print("No changes needed — all secrets and env vars already present and correct")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "task-definition.json"
    patch(path)