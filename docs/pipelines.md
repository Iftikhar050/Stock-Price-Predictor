# Data & Model Pipelines — How Things Actually Get Refreshed

This exists because the answer to "how does data get updated?" was previously scattered across three separate, independently-scheduled entry points with no single doc tying them together. None of them currently run on an OS-level scheduler (no cron / Task Scheduler / systemd unit found in this repo) — all three are started by hand.

## 1. Data ingestion — `scripts/setup_and_sync.py`

**What it does**: creates/verifies the DB schema, loads the active ticker list, then calls `run_full_data_pipeline()` (`src/psx_predictor/pipelines/orchestrator.py`), which runs in order:

1. `macro_pipeline.py` — SBP EasyData rates, IMF DataMapper, global commodities/indices (yfinance), PSX official indices (KMI-30/KSE-30/All-Share/sector indices).
2. `institutional_flows_pipeline.py` — NCCPL FIPI/LIPI flows (manual snapshot files only — see `src/psx_predictor/data/fetch_nccpl_flows.py`), PSX market/sector index sync.
3. `fundamentals_pipeline.py` — EOD price sync (yfinance, PSX DPS fallback), fundamentals (yfinance, PSX DPS, manual files), dividends.
4. `news_sentiment_pipeline.py` — Alpha Vantage macro indicators/news, Google News + RSS + archive scraping, PSX PUCARS corporate disclosures.
5. Supplementary best-effort steps (market breadth, circuit breakers, Google Trends search interest, SBP tier-2 series, Pakistan real-economy activity, political/calendar flags) — each wrapped in its own try/except and treated as optional; a failure here doesn't abort the run.
6. `build_features(ticker)` per active ticker — regenerates every `data/processed/{TICKER}_master.csv`, the single file both the API and model training read.
7. `run_quality_checks_all_tickers()` (`src/psx_predictor/data/quality_gate.py`) — runs after step 6, prints one aggregated "which column is flagged, in how many tickers" summary instead of requiring someone to scroll through ~107 separate per-ticker log blocks.

**Run it**: `python scripts/setup_and_sync.py`

**Trigger**: manual only. No scheduler wraps this today.

## 2. Model retraining, evaluation & promotion — `scripts/run_pipeline.py`

**What it does** (`execute_full_pipeline()`): trains RF → Ridge → XGBoost → LSTM (each a separate subprocess), then runs the full walk-forward evaluation (`src/psx_predictor/models/walk_forward.py`) for all three sklearn-family models across every active ticker, registers each run (`models/registry/`), and calls `promotion.select_best_model_overall()` — if a run clears the promotion gate, its model artifact is copied into `models/production/` and the live API is told to hot-reload (`POST /api/reload_models`, requires `ADMIN_API_KEY`).

Note: this script's own data-sync and feature-rebuild steps (1–2 in its internal numbering) are commented out — **it assumes `setup_and_sync.py` already ran and `data/processed/*_master.csv` is current.** Running this without a recent data sync retrains against stale features.

**Run it**:
- One-off: `python scripts/run_pipeline.py --run-now`
- As a daemon: `python scripts/run_pipeline.py` (schedules itself for 17:00 daily — 5:00 PM, chosen because PSX closes at 3:30 PM)

**Trigger**: manual only unless someone leaves the daemon process running. Not wired to `setup_and_sync.py` or `orchestrator.py` — they are two separate scripts a person has to remember to run in the right order (sync, then retrain).

## 3. News sentiment — `src/psx_predictor/news/scheduler.py`

A third, separate daemon. `start_scheduler()` schedules `NewsAggregator().run_pipeline()` for 11:15 UTC daily (16:15 PKT, just after market close) and loops on `schedule.run_pending()`.

**Run it**: `python -m src.psx_predictor.news.scheduler`

**Relationship to the others**: redundant with (not additive to) step 4 of `setup_and_sync.py`'s pipeline above — both ultimately call the same `NewsAggregator`. If both this daemon and `setup_and_sync.py` run in the same day, news gets fetched twice; harmless (dedup happens at insert time) but wasteful.

## Practical guidance until this is automated

Until Phase 6's scheduling question is decided, refresh the data manually in this order:

```
python scripts/setup_and_sync.py     # data ingestion + feature rebuild + quality summary
python scripts/run_pipeline.py --run-now   # retrain + walk-forward eval + promotion (only if you want a new model)
```

`news/scheduler.py` and `run_pipeline.py`'s daemon mode are optional standalone processes — only run one of them long-term if you actually want unattended scheduling, and be aware they'll double up with `setup_and_sync.py`'s own news step / require a fresh sync respectively.
