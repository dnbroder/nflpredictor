"""
update_readme_results.py — Auto-update README.md with live NFL season results.

Reads weekly prediction and accuracy data from S3, computes running stats,
and rewrites the anchor-delimited sections in README.md.

Anchor tags:
  <!-- NFL:RUNNING_RECORD_START --> ... <!-- NFL:RUNNING_RECORD_END -->
  <!-- NFL:WEEKLY_LOG_START --> ... <!-- NFL:WEEKLY_LOG_END -->

Run locally:
    python3 scripts/update_readme_results.py

Triggered by GitHub Actions (update-results.yml) every Tuesday after accuracy Lambda.
"""

import json
import os
import re
from datetime import datetime

import boto3

S3_BUCKET = os.environ.get("S3_BUCKET", "nfl-predictor-bucket")
README_PATH = "README.md"
SEASON = int(os.environ.get("NFL_SEASON", "2026"))


def _load_s3_json(s3, key: str) -> dict | None:
    """Load a JSON file from S3, return None if not found."""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode())
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"[UPDATE] Warning: failed to load s3://{S3_BUCKET}/{key}: {e}")
        return None


def _list_s3_keys(s3, prefix: str) -> list[str]:
    """List all keys under a prefix."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _load_all_weeks(s3) -> list[dict]:
    """Load all weekly prediction files and accuracy reports."""
    weeks = []

    # Find all prediction files
    pred_keys = _list_s3_keys(s3, f"predictions/nfl_{SEASON}_week")
    for key in sorted(pred_keys):
        # Extract week number from key: predictions/nfl_2026_week05.json
        match = re.search(r"week(\d+)\.json$", key)
        if not match:
            continue
        week_num = int(match.group(1))
        pred_data = _load_s3_json(s3, key)
        if not pred_data:
            continue

        # Try to load corresponding accuracy report
        acc_key = f"accuracy/nfl_{SEASON}_week{week_num:02d}.json"
        acc_data = _load_s3_json(s3, acc_key)

        weeks.append({
            "week": week_num,
            "predictions": pred_data,
            "accuracy": acc_data,
        })

    return weeks


def _compute_record(weeks: list[dict]) -> dict:
    """Compute running season record from weekly data."""
    total_picks = 0
    total_wins = 0
    conviction_picks = 0
    conviction_wins = 0
    high_picks = 0
    high_wins = 0

    for w in weeks:
        acc = w.get("accuracy")
        if not acc or acc.get("status") == "awaiting_actuals":
            continue

        preds = acc.get("predictions", [])
        for p in preds:
            actual = p.get("actual_result")
            if actual is None:
                continue

            total_picks += 1
            if actual:
                total_wins += 1

            tier = p.get("tier", "")
            if tier == "CONVICTION":
                conviction_picks += 1
                if actual:
                    conviction_wins += 1
            elif tier == "HIGH":
                high_picks += 1
                if actual:
                    high_wins += 1

    win_rate = round(total_wins / total_picks * 100, 1) if total_picks > 0 else 0
    roi = round((total_wins * (100 / 110) - (total_picks - total_wins)) / total_picks * 100, 1) if total_picks > 0 else 0

    return {
        "total_picks": total_picks,
        "total_wins": total_wins,
        "win_rate": win_rate,
        "roi": roi,
        "conviction_picks": conviction_picks,
        "conviction_wins": conviction_wins,
        "high_picks": high_picks,
        "high_wins": high_wins,
        "weeks_predicted": len([w for w in weeks if w.get("predictions", {}).get("n_predictions", 0) > 0]),
    }


def _build_running_record(record: dict) -> str:
    """Build the running record markdown table."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M ET")

    if record["total_picks"] == 0:
        return """| Metric | Value | Updated |
|--------|-------|---------|
| **Overall Record** | — | Awaiting season start |
| **Win Rate** | — | — |
| **ROI** | — | — |
| **CONVICTION Record** | — | — |
| **HIGH Record** | — | — |
| **Weeks Predicted** | 0 | — |
| **Season Start** | September 2026 | — |"""

    overall = f"{record['total_wins']}-{record['total_picks'] - record['total_wins']}"
    conv = f"{record['conviction_wins']}-{record['conviction_picks'] - record['conviction_wins']}" if record['conviction_picks'] > 0 else "—"
    high = f"{record['high_wins']}-{record['high_picks'] - record['high_wins']}" if record['high_picks'] > 0 else "—"
    roi_str = f"+{record['roi']}%" if record['roi'] > 0 else f"{record['roi']}%"

    return f"""| Metric | Value | Updated |
|--------|-------|---------|
| **Overall Record** | {overall} | {now} |
| **Win Rate** | {record['win_rate']}% | {now} |
| **ROI** | {roi_str} | {now} |
| **CONVICTION Record** | {conv} | {now} |
| **HIGH Record** | {high} | {now} |
| **Weeks Predicted** | {record['weeks_predicted']} | {now} |
| **Season Start** | September {SEASON} | — |"""


def _build_weekly_log(weeks: list[dict]) -> str:
    """Build the weekly log markdown table."""
    if not weeks:
        return """| Week | Picks | Record | Win % | ROI | Best Pick | Result |
|------|-------|--------|-------|-----|-----------|--------|
| — | — | — | — | — | Awaiting season start | — |"""

    rows = []
    rows.append("| Week | Picks | Record | Win % | ROI | Best Pick | Result |")
    rows.append("|------|-------|--------|-------|-----|-----------|--------|")

    for w in sorted(weeks, key=lambda x: x["week"], reverse=True):
        week_num = w["week"]
        preds = w.get("predictions", {})
        acc = w.get("accuracy")
        n_preds = preds.get("n_predictions", 0)

        # Find the best pick (highest confidence)
        picks = preds.get("predictions", [])
        best_pick = picks[0] if picks else {}
        best_desc = ""
        if best_pick:
            direction = best_pick.get("direction", "?")
            home = best_pick.get("home_team", "?")
            away = best_pick.get("away_team", "?")
            best_desc = f"{direction} {away}@{home}"

        if acc and acc.get("status") != "awaiting_actuals":
            # Resolved week
            resolved_preds = acc.get("predictions", [])
            wins = sum(1 for p in resolved_preds if p.get("actual_result") is True)
            losses = sum(1 for p in resolved_preds if p.get("actual_result") is False)
            total = wins + losses
            record = f"{wins}-{losses}"
            win_pct = f"{round(wins / total * 100, 1)}%" if total > 0 else "—"
            roi = round((wins * (100 / 110) - losses) / total * 100, 1) if total > 0 else 0
            roi_str = f"+{roi}%" if roi > 0 else f"{roi}%"
            # Best pick result
            best_result = "✅" if best_pick.get("actual_result") else "❌" if best_pick.get("actual_result") is False else "⏳"
            rows.append(f"| {week_num} | {n_preds} | {record} | {win_pct} | {roi_str} | {best_desc} | {best_result} |")
        else:
            # Pending week
            rows.append(f"| {week_num} | {n_preds} | — | — | — | {best_desc} | ⏳ |")

    return "\n".join(rows)


def _replace_anchor_region(content: str, start_tag: str, end_tag: str, new_body: str) -> str:
    """Replace content between HTML comment anchors."""
    pattern = re.compile(
        re.escape(start_tag) + r"\n.*?\n" + re.escape(end_tag),
        re.DOTALL,
    )
    replacement = f"{start_tag}\n{new_body}\n{end_tag}"
    new_content, count = pattern.subn(replacement, content)
    if count == 0:
        print(f"[UPDATE] Warning: anchor {start_tag} not found in README.md")
    return new_content


def main():
    print(f"[UPDATE] Loading NFL season {SEASON} results from s3://{S3_BUCKET}/")

    s3 = boto3.client("s3")

    # Load all weekly data
    weeks = _load_all_weeks(s3)
    print(f"[UPDATE] Found {len(weeks)} week(s) of predictions")

    # Compute running record
    record = _compute_record(weeks)
    print(f"[UPDATE] Record: {record['total_wins']}-{record['total_picks'] - record['total_wins']} "
          f"({record['win_rate']}%, ROI {record['roi']}%)")

    # Read current README
    if not os.path.exists(README_PATH):
        print(f"[UPDATE] {README_PATH} not found — nothing to update")
        return

    with open(README_PATH, "r") as f:
        content = f.read()

    # Build new sections
    running_record = _build_running_record(record)
    weekly_log = _build_weekly_log(weeks)

    # Replace anchor regions
    content = _replace_anchor_region(
        content,
        "<!-- NFL:RUNNING_RECORD_START -->",
        "<!-- NFL:RUNNING_RECORD_END -->",
        running_record,
    )
    content = _replace_anchor_region(
        content,
        "<!-- NFL:WEEKLY_LOG_START -->",
        "<!-- NFL:WEEKLY_LOG_END -->",
        weekly_log,
    )

    # Write updated README
    with open(README_PATH, "w") as f:
        f.write(content)

    print(f"[UPDATE] README.md updated successfully")


if __name__ == "__main__":
    main()