# NFL Over/Under Predictor

**Patent Pending** | Provisional Patent Filed February 13, 2026 (USPTO)

A logistic regression model with Platt scaling calibration that predicts NFL game over/unders against closing lines. Built on the AABBH (Average At Bats Between Hits) cycle methodology adapted to NFL scoring patterns via three-layer feature engineering: Points-to-Scoring cycles (P2S), Points-to-Scoring Zone cycles (P2SZ), and Player Impact Curves (PIC).

### Validation Results

**Walk-Forward Validation (6-season rolling window, test 2022–2024):**

| Config | Win Rate | ROI | Bets | Break-Even |
|--------|----------|-----|------|------------|
| **Raw (no calibration)** | 55.6% | +6.1% | 1,170 | 52.38% |
| **Calibrated (Platt scaling)** | **57.7%** | **+10.1%** | **690** | 52.38% |

**Per-Window (Calibrated):**

| Window | Train | Test | ROI | Bets |
|--------|-------|------|-----|------|
| 1 | 2016–2021 | 2022 | **+22.9%** | 132 |
| 2 | 2017–2022 | 2023 | **+24.6%** | 265 |
| 3 | 2018–2023 | 2024 | -8.8% | 293 |
| **Aggregate** | | | **+10.1%** | **690** |

Calibration cut volume 41% (1,170 → 690 bets) and nearly doubled ROI (+6.1% → +10.1%) by making probability outputs honest — the model only bets when genuinely confident.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Infrastructure Deployment](#2-infrastructure-deployment)
3. [System Architecture](#3-system-architecture)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Configuration](#5-model-configuration)
6. [API Endpoints](#6-api-endpoints)
7. [Customer-Facing API](#7-customer-facing-api)
8. [Monitoring & Operations](#8-monitoring--operations)
9. [Roadmap](#9-roadmap)
10. [Live Season Results](#10-live-season-results)

---

## 1. Prerequisites

### CI/CD Role

Uses the shared `mlb-cicd-role` IAM role across all sport predictors. The role must include permissions for CloudFormation, ECS, ECR, Lambda, EventBridge, API Gateway, S3, IAM, and EC2. See the MLB repo's README for the full role definition.

The OIDC trust policy must include:
```
repo:{YourGitHubUsername}/nflpredictor:ref:refs/heads/*
```

### Required Tools

- AWS CLI v2
- Docker
- Python 3.11+
- Node.js (for GitHub Actions)

---

## 2. Infrastructure Deployment

### CI/CD Pipeline Flow

```
VS Code → git push main → GitHub Actions → CloudFormation → ECR → ECS Fargate → Lambda
```

On every push to `main`:

1. GitHub Actions authenticates via `mlb-cicd-role` (OIDC federation)
2. **CloudFormation stack update** — EventBridge rules, Lambda functions, API Gateway, IAM roles
3. Docker image build and push to ECR
4. ECS task definition update and Fargate rolling deployment
5. Lambda code deploy — `predict-weekly` and `operations` functions
6. Health check verification

### CloudFormation Stack — Single-Stack Architecture

| Resource | Purpose |
|----------|---------|
| **ECS Cluster + Service** | Fargate cluster running the FastAPI prediction API |
| **ECS Task Definition** | Container config: 2 vCPU, 8GB RAM, port 8000 |
| **S3 Bucket** | Models, predictions, accuracy reports, data cache |
| **IAM Roles** | Task execution, task role, Lambda roles, API Gateway S3 role |
| **CloudWatch Log Group** | Container and Lambda logs |
| **EventBridge Rules (5)** | Season-scoped automated schedules |
| **Lambda Functions (2)** | Weekly predictions + operations (retrain/refresh/accuracy) |
| **API Gateway** | Public REST API for dandoesdfs.com |
| **Security Group** | Port 8000 ingress |

### Automated Schedules

All in-season rules use month-scoped cron expressions (`9-12,1-2`) so they only fire during the NFL season (September through February). No manual enable/disable needed.

| Schedule | Rule | Lambda | Purpose |
|----------|------|--------|---------|
| Tuesday 7am ET (Sep–Feb) | `accuracy-check` | operations | Post-game accuracy check |
| Tuesday 10am ET (Sep–Feb) | `tnf-predictions` | predict-weekly | Thursday Night Football picks |
| Wednesday 6am ET (Sep–Feb) | `data-refresh` | operations | Warm S3 caches with latest data |
| Saturday 8am ET (Sep–Feb) | `sunday-slate-predictions` | predict-weekly | Full Sunday/Monday slate |
| August 15, 6am ET (annual) | `preseason-retrain` | operations | Retrain on latest 6 seasons |

### S3 Output Structure

```
s3://nfl-predictor-bucket/
├── predictions/
│   ├── nfl_2026_week01.json        ← Weekly picks archive
│   ├── nfl_latest.json             ← Most recent picks (API reads this)
├── accuracy/
│   ├── nfl_2026_week01.json        ← Weekly accuracy reports
├── logs/
│   ├── retrain_2026_*.json         ← Retrain audit trail
│   ├── refresh_2026_*.json         ← Data refresh audit trail
├── models/
│   └── nfl_scoring_predictor.pkl   ← Trained model (calibrated)
├── data/
│   ├── schedules/                  ← Cached game data
│   ├── drives/                     ← Cached drive data
│   ├── pbp/                        ← Cached play-by-play data
│   ├── closing_lines/              ← Cached closing line data (SBR + Odds API)
│   ├── epa/                        ← Cached EPA data
│   └── qb_stats/                   ← Cached QB stats
```

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEVELOPER WORKFLOW                           │
│  VS Code ──► GitHub (main) ──► GitHub Actions                       │
│              │                  ├── CloudFormation (infra)           │
│              │                  ├── ECR (Docker image)               │
│              │                  ├── ECS Fargate (rolling deploy)     │
│              │                  └── Lambda (code deploy)             │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ECS FARGATE CLUSTER                          │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │                   FastAPI Application                     │      │
│  │                                                           │      │
│  │  /predict/week/{week} ── Full 6-layer feature pipeline    │      │
│  │       ├── Layer 1: P2S (Points-to-Scoring cycles)         │      │
│  │       ├── Layer 2: P2SZ (Scoring Zone cycles)             │      │
│  │       ├── Layer 3: PIC (Player Impact Curves)             │      │
│  │       ├── EPA (Expected Points Added)                     │      │
│  │       ├── QB Stats (career/season quality)                │      │
│  │       └── Closing Lines (SBR + Odds API)                  │      │
│  │       ▼                                                   │      │
│  │  Logistic Regression (C=0.034) → Platt Calibration        │      │
│  │       ▼                                                   │      │
│  │  Calibrated over/under probabilities → Tier assignment    │      │
│  │                                                           │      │
│  │  /train-async, /backtest-async, /tune-async, /shap-async  │      │
│  └───────────────────────────────────────────────────────────┘      │
└──────────────────────────│──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS SERVICES                                │
│  EventBridge (5 rules) ──► Lambda (predict-weekly, operations)      │
│  S3 ──► predictions/nfl_latest.json ──► API Gateway ──► website     │
│  CloudWatch Logs ──► /ecs/nfl-predictor                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Map

```
main.py (FastAPI entrypoint)
├── nfl_predictor/
│   ├── nfl_data.py              ← nflverse data pulls, SBR/Odds API closing lines, S3 caching
│   ├── feature_engineering.py   ← 63-feature pipeline (32 used), P2S + P2SZ + PIC + EPA + QB
│   ├── model.py                 ← Logistic regression, Platt calibration, prediction
│   └── backtester.py            ← Walk-forward validation, confidence tier analysis
├── lambda/
│   ├── predict_weekly/
│   │   └── handler.py           ← Weekly prediction trigger (EventBridge → ECS → S3)
│   └── operations/
│       └── handler.py           ← Retrain, data refresh, accuracy check
└── infrastructure.yaml          ← CloudFormation (ECS, Lambda, EventBridge, API Gateway)
```

---

## 4. Feature Engineering

The model uses **32 features** selected from a 63-feature pipeline. Features span six engineering layers. All rolling features use temporal ordering to prevent data leakage.

### Layer 1: P2S — Points-to-Scoring Cycles (8 features)

| Feature | Description |
|---------|-------------|
| `p2s_ratio` | Current scoring cycle position (drives since last scoring drive / baseline) |
| `p2s_optimal_zone` | 1 if team is in the scoring-likely zone of its cycle |
| `p2s_conversion_rate_5g` | 5-game rolling scoring drive conversion rate |
| `p2s_def_ratio` | Opponent's defensive cycle position |
| `p2s_def_optimal_zone` | 1 if opponent defense is in its vulnerable zone |
| `p2s_p2sz_dual_optimal` | Both P2S and P2SZ in optimal zone simultaneously |
| `p2s_p2sz_signal_strength` | Combined signal magnitude |
| `matchup_dual_optimal` | Offense optimal + defense degraded interaction |

### Layer 2: P2SZ — Points-to-Scoring Zone Cycles (3 features)

| Feature | Description |
|---------|-------------|
| `p2sz_rz_ratio` | Red zone scoring cycle position |
| `p2sz_rz_optimal_zone` | 1 if team is in red zone scoring-likely zone |
| `p2sz_sp_baseline` | Scoring play baseline rate |

### Layer 3: PIC — Player Impact Curves (8 features)

| Feature | Description |
|---------|-------------|
| `pic_qb_in_zone` | QB in scoring cycle optimal zone |
| `pic_qb_career_rate` | QB career scoring drive rate |
| `pic_qb_is_backup` | Backup QB flag (scoring suppressor) |
| `pic_skill_sync_rate` | Skill position players cycle synchronization |
| `pic_team_alignment` | Full offensive unit cycle alignment |
| `pic_present_quality` | Quality of present skill players (snap-weighted) |
| `pic_avg_player_epa` | EPA-weighted player quality score |
| `pic_cycle_confirmed` | PIC cycle confirmed by multiple signals |

### EPA Features (2)

| Feature | Description |
|---------|-------------|
| `epa_offense_avg` | 5-game rolling offensive EPA per play |
| `epa_defense_avg` | 5-game rolling defensive EPA per play |

### Market Features (4)

| Feature | Description |
|---------|-------------|
| `opening_total` | Vegas opening total line |
| `opening_spread` | Vegas opening spread |
| `total_movement` | Closing total minus opening total (sharp money signal) |
| `spread_movement` | Closing spread minus opening spread |

### Context Features (7)

| Feature | Description |
|---------|-------------|
| `is_home` | Home field advantage |
| `is_dome` | Dome game (weather-independent) |
| `is_cold_game` | Temperature below 32°F |
| `temp` | Game temperature |
| `wind_mph` | Wind speed |
| `scoring_avg_5g` | 5-game rolling scoring average |
| `scoring_allowed_avg_5g` | 5-game rolling points allowed |

### Data Coverage

| Source | Seasons | Records | Coverage |
|--------|---------|---------|----------|
| nflverse games | 2016–2024 | 2,476 | 100% |
| nflverse drives | 2016–2024 | 50,458 | 100% |
| nflverse PBP | 2016–2024 | 435,483 | 100% |
| SBR closing lines | 2016–2019 | 1,068/1,068 | 100% |
| Odds API closing lines | 2020–2024 | 1,408 | 99.5% |
| Combined closing lines | 2016–2024 | 2,476/2,488 | 99.5% |

---

## 5. Model Configuration

### Production Model (v2.0)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | Logistic Regression (L2) | Interpretable, fast, robust to NFL's limited sample sizes |
| C (regularization) | 0.034 | 50-value 5-fold CV grid search (best AUC=0.5403) |
| Features | 32 / 63 available | Pruned 6 features with near-zero or negative contribution |
| Calibration | Platt scaling (sigmoid, 3-fold CV) | Compresses overconfident probabilities to honest values |
| Training window | 6 seasons (rolling) | Walk-forward validated; adapts to game evolution |
| Break-even | 52.38% | Standard -110 vig |

### Confidence Tiers

| Tier | Threshold | Walk-Forward Win Rate | Walk-Forward ROI | Bets (3 seasons) |
|------|-----------|----------------------|------------------|-------------------|
| CONVICTION | P ≥ 55% | 53.8% | +2.8% | 117 |
| HIGH | P ≥ 53% | 56.6% | +8.0% | 599 |
| STANDARD | P ≥ 52% | 57.7% | +10.1% | 690 |
| LEAN | P < 52% | — | — | DFS signal only |

### Decisions Log

| Decision | Outcome | Date |
|----------|---------|------|
| Stage 1 rule filter | **KILLED** — consistently negative lift on test sets | Apr 2026 |
| Feature pruning (38 → 32) | Dropped 6 low-signal features, no accuracy loss | Apr 2026 |
| C parameter tuning (0.1 → 0.034) | +0.3% AUC improvement | Apr 2026 |
| Platt scaling calibration | **+4.0% ROI improvement** (6.1% → 10.1%), -41% volume | Apr 2026 |
| Recency-weighted training | **REVERTED** — hurt aggregate ROI (10.1% → 7.4%) | Apr 2026 |
| 6-season rolling window | Outperforms 8-season; drops stale data | Apr 2026 |

---

## 6. API Endpoints

### Internal Endpoints

| Method | Endpoint | Description | Runtime |
|--------|----------|-------------|---------|
| `GET` | `/health` | Health check | Instant |
| `GET` | `/schema` | Season data summary | 5–10s |
| `GET` | `/predict/week/{week}` | Production predictions for a week | 10–12 min |
| `POST` | `/train-async` | Train model (background) | 10–15 min |
| `POST` | `/backtest-async` | Walk-forward backtest (background) | 10–12 min |
| `POST` | `/tune-async` | Hyperparameter tuning (background) | 15–30 min |
| `POST` | `/shap-async` | SHAP feature importance (background) | 5–10 min |

### Prediction Output Format

```json
{
  "week": 5,
  "season": 2026,
  "model_version": "v2.0_calibrated",
  "n_predictions": 8,
  "conviction_picks": 2,
  "predictions": [
    {
      "game_id": "2026_05_KC_BUF",
      "home_team": "BUF",
      "away_team": "KC",
      "direction": "UNDER",
      "over_probability": 0.4312,
      "confidence": 0.5688,
      "tier": "CONVICTION",
      "opening_total": 47.5,
      "closing_total": 47.0,
      "is_dome": false,
      "wind_mph": 12,
      "temp": 42
    }
  ]
}
```

---

## 7. Customer-Facing API

### Public Endpoint

```
GET https://{api-id}.execute-api.us-east-1.amazonaws.com/v1/nfl/picks
```

Returns the latest weekly predictions JSON from S3. CORS enabled for `dandoesdfs.com`. Served via API Gateway → S3 direct integration (no compute cost per request).

### Multi-Sport API (planned)

The API Gateway is designed to serve all sport models under one unified endpoint:

| Endpoint | Sport | Status |
|----------|-------|--------|
| `/v1/nfl/picks` | NFL Over/Under | **Live** |
| `/v1/baseball/picks/daily` | MLB Hit Predictions | Live (separate stack) |
| `/v1/golf/picks/tournament` | Golf Placement | Live (separate stack) |
| `/v1/basketball/picks/daily` | NBA Player Points | Planned |

---

## 8. Monitoring & Operations

### Log Prefixes

| Prefix | Module | Purpose |
|--------|--------|---------|
| `[PREDICT]` | main.py | Prediction pipeline trace |
| `[MODEL]` | model.py | Training, calibration, evaluation |
| `[FEAT]` | feature_engineering.py | Feature pipeline progress |
| `[PIC]` | feature_engineering.py | Layer 3 Player Impact Curves |
| `[BT]` | backtester.py | Walk-forward validation |
| `[DATA]` | nfl_data.py | Data loading, caching, coverage |
| `[ASYNC-BACKTEST]` | main.py | Background backtest jobs |
| `[ASYNC-TRAIN]` | main.py | Background training jobs |

### Key Operational Commands

```bash
# Get ECS public IP
aws ecs list-tasks --cluster nfl-predictor-cluster --region us-east-1 \
  --query 'taskArns[0]' --output text | xargs -I{} \
  aws ecs describe-tasks --cluster nfl-predictor-cluster --tasks {} \
  --region us-east-1 --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text | xargs -I{} \
  aws ec2 describe-network-interfaces --network-interface-ids {} \
  --region us-east-1 --query 'NetworkInterfaces[0].Association.PublicIp' --output text

# Tail logs
aws logs tail /ecs/nfl-predictor --follow --since 5m --region us-east-1

# Run walk-forward backtest (6-season rolling, test 2022-2024)
curl -s -X POST "http://${IP}:8000/backtest-async?train_start=2016&train_end=2021&test_year=2024"

# Train production model
curl -s -X POST "http://${IP}:8000/train-async?train_start=2019&train_end=2024"

# Get week predictions
curl -s "http://${IP}:8000/predict/week/5?season=2026"
```

---

## 9. Roadmap

### Completed

- [x] Three-layer feature engineering (P2S + P2SZ + PIC)
- [x] SBR + Odds API closing line integration (99.5% coverage)
- [x] C parameter tuning via grid search
- [x] Feature pruning (38 → 32)
- [x] Stage 1 rule filter evaluation and removal
- [x] Platt scaling calibration (+4% ROI improvement)
- [x] Walk-forward validation (3-season, 1,170 OOS bets)
- [x] Production predict endpoint with full pipeline
- [x] EventBridge automation (season-scoped, fully hands-free)
- [x] API Gateway for dandoesdfs.com
- [x] CI/CD pipeline (CFN + ECR + ECS + Lambda)

### Parked

- [ ] Opponent-adjusted cycle features (offense optimal × defense degraded interaction) — deferred due to dual_optimal signal instability
- [ ] Recency-weighted training — reverted, hurt aggregate ROI
- [ ] CLV filtering — production-time confirmation signal, not a backtest change

### Future

- [ ] Under bet analysis — model shows stronger under signal; asymmetric thresholds may improve ROI
- [ ] Season-start auto-retrain with extended training window validation
- [ ] Live closing line comparison for CLV confirmation
- [ ] Weekly accuracy auto-update in README (GitHub Actions bot)

---

## 10. Live Season Results

> Results update automatically via the accuracy check Lambda and GitHub Actions bot.
> Data source: `s3://nfl-predictor-bucket/predictions/nfl_latest.json`

### 2026 Season — Running Record

<!-- NFL:RUNNING_RECORD_START -->
| Metric | Value | Updated |
|--------|-------|---------|
| **Overall Record** | — | Awaiting season start |
| **Win Rate** | — | — |
| **ROI** | — | — |
| **CONVICTION Record** | — | — |
| **HIGH Record** | — | — |
| **Weeks Predicted** | 0 | — |
| **Season Start** | September 2026 | — |
<!-- NFL:RUNNING_RECORD_END -->

### Weekly Log

<!-- NFL:WEEKLY_LOG_START -->
| Week | Picks | Record | Win % | ROI | Best Pick | Result |
|------|-------|--------|-------|-----|-----------|--------|
| — | — | — | — | — | Awaiting season start | — |
<!-- NFL:WEEKLY_LOG_END -->

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1.0 | Apr 2026 | Initial codebase — nflverse data pipeline, drive-level P2S/P2SZ features |
| v0.2.0 | Apr 2026 | Layer 3 PIC (Player Impact Curves), EPA integration, QB stats |
| v0.3.0 | Apr 2026 | SBR closing lines (2016–2019), Odds API (2020–2024), 99.5% coverage |
| v1.0.0 | Apr 2026 | Two-stage model (rule filter + logistic regression), 14 bug audit fix |
| v1.1.0 | Apr 2026 | C=0.034 tuning, 38→32 feature pruning, Stage 1 filter killed |
| v2.0.0 | Apr 2026 | **Production release** — Platt calibration (+10.1% ROI), walk-forward validated, EventBridge automation, API Gateway, CI/CD with CFN |

---

## AWS Infrastructure

| Component | Resource | Region |
|-----------|----------|--------|
| ECS Cluster | `nfl-predictor-cluster` | us-east-1 |
| S3 Bucket | `nfl-predictor-bucket` | us-east-1 |
| ECR Repository | `nfl-predictor` | us-east-1 |
| CloudWatch Logs | `/ecs/nfl-predictor` | us-east-1 |
| Lambda (predictions) | `nfl-predictor-predict-weekly` | us-east-1 |
| Lambda (operations) | `nfl-predictor-operations` | us-east-1 |
| API Gateway | `nfl-predictor-api` | us-east-1 |
| Account | 033896355668 | — |