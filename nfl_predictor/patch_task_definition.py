#!/usr/bin/env python3
"""
Patch an ECS task definition JSON before deployment.

Ensures required secrets and environment variables are present
without overwriting any existing values.

Usage:
    python3 scripts/patch_task_definition.py task-definition.json
"""

import json
import sys

REQUIRED_SECRETS = [
    {
        "name": "ODDS_API_KEY",
        "valueFrom": "arn:aws:secretsmanager:us-east-1:033896355668:secret:odds-api-key",
    },
]

REQUIRED_ENV = [
    {"name": "LOG_LEVEL", "value": "INFO"},
]


def patch(path: str) -> None:
    with open(path) as f:
        td = json.load(f)

    cd = td["containerDefinitions"][0]

    # Patch secrets
    secrets = cd.get("secrets", [])
    existing = {s["name"] for s in secrets}
    added_secrets = []
    for secret in REQUIRED_SECRETS:
        if secret["name"] not in existing:
            secrets.append(secret)
            added_secrets.append(secret["name"])
    cd["secrets"] = secrets

    # Patch environment
    env = cd.get("environment", [])
    existing_env = {e["name"] for e in env}
    added_env = []
    for var in REQUIRED_ENV:
        if var["name"] not in existing_env:
            env.append(var)
            added_env.append(var["name"])
    cd["environment"] = env

    with open(path, "w") as f:
        json.dump(td, f, indent=2)

    if added_secrets:
        print(f"Added secrets: {', '.join(added_secrets)}")
    if added_env:
        print(f"Added env vars: {', '.join(added_env)}")
    if not added_secrets and not added_env:
        print("No changes needed — all secrets and env vars already present")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "task-definition.json"
    patch(path)