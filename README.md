# AI Stock Predictor 🚀

A modern, full-stack AI application designed to predict stock prices for the Pakistan Stock Exchange (PSX) using ensemble machine learning techniques. It features a beautifully designed React dashboard and a robust FastAPI backend.

## 🌟 Key Features

* **Multi-Company Coverage:** Fully integrated support for tracking and predicting major PSX players including **PSO, FFC, NBP, MEBL, OGDC, and LUCK**.
* **Global AI Predictions:** Utilizes four different ML models (Random Forest, Linear Regression, XGBoost, and LSTM Deep Learning) trained on a massive concatenated dataset across all companies to learn generalized market trends.
* **Consensus Target Range:** Automatically calculates an ensemble range and assigns an **AI Confidence Score** based on the agreement variance between the models.
* **Live Market Performers & Pricing:** An auto-updating dashboard ranking the top active stocks, advancers, and decliners. Automatically polls live prices during market hours and displays a "Market Closed" indicator after hours.
* **NLP News Sentiment Analysis:** Scrapes real-time financial news for each ticker from Google News, performs NLP sentiment analysis using VADER, and incorporates a 3-day sentiment decay into the ML dataset.
* **Corporate Dividend Engine:** Scrapes historical cash payouts via Yahoo Finance to engineer `dividend_yield` and `days_since_dividend` features, drastically improving AI accuracy around ex-dividend dates.
* **Automated Data Scraping:** Includes a robust scraper pipeline to fetch the latest End-Of-Day (EOD) OHLCV data, News Articles, and Dividend Payouts directly into PostgreSQL.

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

Initialize the database tables and scrape the latest historical data, news, and dividends:
```bash
python setup_and_sync.py
```

Build the technical indicators, merge sentiment, merge dividends, and prepare the dataset for the AI models:
```bash
python src/psx_predictor/data/build_features.py
```

Train the ML models from scratch on the newly generated dataset:
```bash
python retrain_models_now.py
```

Start the FastAPI server:
```bash
uvicorn src.psx_predictor.api.main:app --reload
```

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

To manually sync the latest market data (EOD + News + Dividends) without rebuilding the database tables:
```bash
python sync_data_now.py
```
After syncing, you can trigger `build_features.py` and `retrain_models_now.py` to update the AI.

## 📜 License
This project is open-source and available under the MIT License.
