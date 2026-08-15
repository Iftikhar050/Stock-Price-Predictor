# AI Stock Predictor 🚀

A modern, full-stack AI application designed to predict stock prices for the Pakistan Stock Exchange (PSX) using ensemble machine learning techniques. It features a beautifully designed React dashboard and a robust FastAPI backend.

## 🌟 Key Features

* **Full-Market Coverage:** Designed to dynamically ingest, track, and predict over 30 active KSE-100 equities seamlessly. The system builds and updates the universe directly from PSX Metadata.
* **Global AI Predictions:** Utilizes four different ML models (Random Forest, Linear Regression, XGBoost, and LSTM Deep Learning) trained on a massive concatenated dataset across all companies to learn generalized market trends.
* **Consensus Target Range:** Automatically calculates an ensemble range and assigns an **AI Confidence Score** based on the agreement variance between the models.
* **Live Market Performers & Pricing:** An auto-updating dashboard ranking the top active stocks, advancers, and decliners. Automatically polls live prices during market hours and displays a "Market Closed" indicator after hours.
* **NLP News Sentiment Analysis:** Scrapes real-time financial news for each ticker from Google News, performs NLP sentiment analysis using VADER, and incorporates a 3-day sentiment decay into the ML dataset.
* **Corporate Dividend Engine:** Scrapes historical cash payouts via Yahoo Finance to engineer `dividend_yield` and `days_since_dividend` features, drastically improving AI accuracy around ex-dividend dates.
* **Automated Pipeline Orchestration:** A zero-touch cron-ready architecture (`run_pipeline.py`) continuously orchestrates EOD OHLCV scraping (via Yahoo Finance), NLP News aggregation, Dividend tracking, dynamic ML Feature Engineering, and model retraining.

## 🛠️ Technology Stack

* **Frontend:** React, Vite, Tailwind CSS, Recharts (for dynamic and interactive financial charts).
* **Backend:** FastAPI, Python, SQLAlchemy.
* **Database:** PostgreSQL (for storing historical OHLCV data, news, and dividends).
* **Machine Learning:** PyTorch (for LSTM), Scikit-Learn (Random Forest, Linear Regression), XGBoost.
* **NLP & Scraping:** BeautifulSoup4, VADER Sentiment Intensity Analyzer, requests.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Node.js & npm
* PostgreSQL

### 1. Database Setup
Ensure PostgreSQL is running. Create a new database for the project. Rename `.env.example` to `.env` and update your PostgreSQL credentials:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/your_database
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
python run_pipeline.py --run-now
```
*(This will fetch the latest market data, build all technical features, train all 4 ML models, and automatically hot-reload the backend server you started above).*

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

To fully sync the latest market data (EOD + News + Dividends), calculate features, and retrain the machine learning models, simply trigger the pipeline orchestrator:
```bash
python run_pipeline.py --run-now
```

Alternatively, to automate this process so that the AI trains on the new market data every single day without human intervention, run:
```bash
python run_pipeline.py
```

## 📜 License
This project is open-source and available under the MIT License.
