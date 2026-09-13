# AI Stock Predictor 🚀

A modern, full-stack AI application designed to predict stock prices for the Pakistan Stock Exchange (PSX) using ensemble machine learning techniques. It features a beautifully designed React dashboard and a robust FastAPI backend.

## 🌟 Key Features

* **Growing Market Coverage:** Actively tracks 100+ PSX-listed equities today, with a phased staged-ingestion pipeline (`scripts/run_staged_ingestion.py`) built to expand toward the full ~750-symbol PSX universe in reviewable batches — new tickers are ingested into `data/staging/phase_N/` and quality-gated before ever being promoted into the production dataset. See [Data Pipeline & Ticker Universe](#-data-pipeline--ticker-universe) below.
* **Ensemble AI Predictions:** The production walk-forward pipeline trains and evaluates four models per ticker — Random Forest, Ridge Regression, XGBoost, and LightGBM — on a shared engineered feature set, and promotes whichever configuration beats the naive persistence baseline.
* **Consensus Target Range:** Automatically calculates an ensemble range and assigns an **AI Confidence Score** based on the agreement variance between the models.
* **Live Market Performers & Pricing:** An auto-updating dashboard ranking the top active stocks, advancers, and decliners. Automatically polls live prices during market hours and displays a "Market Closed" indicator after hours.
* **Macro, Institutional Flow & Fundamentals Context:** Ingests SBP policy rates/KIBOR/T-bills, IMF indicators, global commodities/indices, NCCPL institutional flows, and per-company quarterly fundamentals (official PSX DPS filings + Yahoo Finance, with manually-parsed PDF filings taking priority where available) alongside price/volume history.
* **NLP News Sentiment Analysis:** Aggregates financial news and PSX corporate announcements (PUCARS) per ticker, scores sentiment (VADER + Alpha Vantage NLP), and engineers decayed sentiment features into the ML dataset.
* **Corporate Dividend Engine:** Scrapes historical cash payouts via Yahoo Finance to engineer `dividend_yield` and `days_since_dividend` features, drastically improving AI accuracy around ex-dividend dates.
* **Automated Pipeline Orchestration:** A zero-touch cron-ready architecture (`scripts/run_pipeline.py`) continuously orchestrates EOD OHLCV scraping, macro/institutional-flow/fundamentals/news sync, dynamic ML feature engineering, and model retraining.
* **Model Experimentation Notebooks:** `notebooks/PSO_pipeline.ipynb` and `notebooks/MEBL_pipeline.ipynb` walk through a full EDA-to-evaluation comparison across 9 model families (linear, kernel/SVM, tree ensembles, and deep learning), including overfitting/underfitting diagnostics and hyperparameter tuning — written up in `reports/PSX_Model_Training_Report.pdf`.

## 🛠️ Technology Stack

* **Frontend:** React, Vite, Tailwind CSS, Recharts (for dynamic and interactive financial charts).
* **Backend:** FastAPI, Python, SQLAlchemy.
* **Database:** PostgreSQL (for storing historical OHLCV data, macro/institutional/fundamentals data, news, and dividends).
* **Machine Learning:** Scikit-Learn (Random Forest, Ridge, SVR), XGBoost, LightGBM, PyTorch (for the LSTM/MLP experiments in `notebooks/`).
* **NLP & Scraping:** BeautifulSoup4, VADER Sentiment Intensity Analyzer, cloudscraper, requests, pdfplumber (for manually-sourced PDF filings).

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Node.js & npm
* PostgreSQL

### 1. Database & Environment Setup
Ensure PostgreSQL is running and you've created a database for the project. Copy `.env.example` to `.env` and fill in your credentials:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=postgres
DB_PASSWORD=your_secure_password
```
Two more keys are optional but recommended — each fetcher skips gracefully (never fabricates data) if its key is unset:
```env
SBP_API_KEY=your_sbp_easydata_api_key        # State Bank of Pakistan EasyData - policy rate, KIBOR, T-bills, CPI, reserves
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key  # News & sentiment feed
```

### 2. Backend Setup
Navigate to the project root and install the Python dependencies:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn src.psx_predictor.api.main:app --reload
```

The database scraping, feature engineering, and AI model training is entirely orchestrated through a single automated pipeline. In a new terminal, run:
```bash
python scripts/run_pipeline.py --run-now
```
*(This will fetch the latest market data, build all technical features, train the model ensemble, and automatically hot-reload the backend server you started above).*

### 3. Frontend Setup
Open a new terminal, navigate to the `frontend` folder, and install the dependencies:
```bash
cd frontend
npm install
```

Start the Vite development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`.

## 🔄 Updating Data

To fully sync the latest market data (EOD + macro + institutional flows + fundamentals + news + dividends), calculate features, and retrain the machine learning models for the currently-active ticker set, trigger the pipeline orchestrator:
```bash
python scripts/run_pipeline.py --run-now
```

Alternatively, to automate this process so the AI trains on new market data every day without human intervention, run:
```bash
python scripts/run_pipeline.py
```

## 🌐 Data Pipeline & Ticker Universe

Expanding beyond the currently-active ticker set is handled as a **separate, reviewable process** so unaudited data never contaminates the production dataset in `data/processed/`:

1. `scripts/fetch_psx_listed_companies.py` — scrapes the full PSX-listed symbol list.
2. `scripts/build_phase_manifest.py` — splits the remaining universe into fixed-size batches, recorded in `data/phase_manifest.json`.
3. `scripts/run_staged_ingestion.py phase_N` — runs the full ingestion pipeline for one batch, writing master CSVs to `data/staging/phase_N/` instead of `data/processed/`.
4. `scripts/run_phase_quality_check.py phase_N` — runs the data-quality gate against a phase's staged output and writes a summary report for review.
5. `scripts/audit_master_csv_freshness.py` — flags any existing dataset that predates a data-fix commit and needs regenerating.

A phase is only copied into `data/processed/` (and its tickers marked active) after a human reviews its quality report — nothing is promoted automatically.

## 📜 License
This project is open-source and available under the MIT License.
