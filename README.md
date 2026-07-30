# AI Stock Predictor 🚀

A modern, full-stack AI application designed to predict stock prices for the Pakistan Stock Exchange (PSX) using ensemble machine learning techniques. It features a beautifully designed React dashboard and a robust FastAPI backend.

## 🌟 Key Features

* **Real-time Live Price Monitoring:** Automatically polls live prices during market hours.
* **Ensemble AI Predictions:** Utilizes four different ML models (Random Forest, Linear Regression, XGBoost, and LSTM Deep Learning) to forecast stock prices.
* **Consensus Target Range:** Automatically calculates an ensemble range and assigns an **AI Confidence Score** based on the agreement variance between the models.
* **Automated Data Scraping:** Includes a robust scraper pipeline to fetch the latest End-Of-Day (EOD) OHLCV data directly from the PSX Data Portal.
* **Comprehensive Company Profiles:** Scrapes and displays vital company information, auditor details, business descriptions, and key personnel.

## 🛠️ Technology Stack

* **Frontend:** React, Vite, Tailwind CSS, Recharts (for dynamic and interactive financial charts).
* **Backend:** FastAPI, Python, SQLAlchemy.
* **Database:** PostgreSQL (for storing historical OHLCV data).
* **Machine Learning:** PyTorch (for LSTM), Scikit-Learn (Random Forest, Linear Regression), XGBoost.

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
source venv/Scripts/activate  # On Windows
pip install -r requirements.txt
```

Initialize the database tables and scrape the latest historical data:
```bash
python setup_and_sync.py
```

Build the technical indicators and prepare the dataset for the AI models:
```bash
python src/psx_predictor/data/build_features.py
```

*(Optional)* Train the ML models:
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

To manually sync the latest market data and recalculate technical indicators without retraining the models, simply run:
```bash
python sync_data_now.py
```

## 📜 License
This project is open-source and available under the MIT License.
