import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.psx_predictor.db.connection import engine
from sqlalchemy import text
from src.psx_predictor.models.utils import choose_global_cutoff

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "figures")

def get_ticker_sectors():
    query = text("SELECT ticker, sector FROM stock_metadata")
    with engine.connect() as conn:
        res = conn.execute(query).fetchall()
    return {row[0]: row[1] for row in res}

TICKER_SECTORS = get_ticker_sectors()

def get_original_features(ticker):
    """Reconstructs the original baseline features from the DB for fair comparison."""
    query = text("SELECT * FROM stock_eod_data WHERE ticker = :ticker ORDER BY date ASC")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker.upper()})
    
    if df.empty:
        return df
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    for window in [7, 21, 50]:
        sma = df['close'].rolling(window=window).mean()
        df[f'sma_{window}_dist'] = (df['close'] / sma) - 1.0
        
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['daily_return'] = df['close'].pct_change()
    for lag in [1, 2, 3]:
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        
    bb_middle = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    bb_upper = bb_middle + (rolling_std * 2)
    bb_lower = bb_middle - (rolling_std * 2)
    df['bb_middle_dist'] = (df['close'] / bb_middle) - 1.0
    df['bb_upper_dist'] = (df['close'] / bb_upper) - 1.0
    df['bb_lower_dist'] = (df['close'] / bb_lower) - 1.0
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_v = typical_price * df['volume']
    vwap = tp_v.rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    df['vwap_14_dist'] = (df['close'] / vwap) - 1.0
    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    query_sent = text("SELECT date, sentiment_score FROM stock_news_sentiment WHERE ticker = :ticker ORDER BY date ASC")
    with engine.connect() as conn:
        sentiment_df = pd.read_sql(query_sent, conn, params={"ticker": ticker.upper()})
    
    if not sentiment_df.empty:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        df = pd.merge(df, sentiment_df, on='date', how='left')
        df['sent_lag_1'] = df['sentiment_score'].shift(1)
        df['sent_lag_2'] = df['sentiment_score'].shift(2)
        df['sent_lag_3'] = df['sentiment_score'].shift(3)
        def apply_decay(row):
            if pd.notna(row['sentiment_score']): return row['sentiment_score']
            if pd.notna(row['sent_lag_1']): return row['sent_lag_1'] * 0.5
            if pd.notna(row['sent_lag_2']): return row['sent_lag_2'] * 0.25
            if pd.notna(row['sent_lag_3']): return row['sent_lag_3'] * 0.125
            return 0.0
        df['sentiment_score'] = df.apply(apply_decay, axis=1)
        df.drop(columns=['sent_lag_1', 'sent_lag_2', 'sent_lag_3'], inplace=True)
    else:
        df['sentiment_score'] = 0.0
        
    query_div = text("SELECT ex_dividend_date as date, dividend_amount FROM stock_dividends WHERE ticker = :ticker ORDER BY date ASC")
    with engine.connect() as conn:
        div_df = pd.read_sql(query_div, conn, params={"ticker": ticker.upper()})
        
    if not div_df.empty:
        div_df['date'] = pd.to_datetime(div_df['date'])
        df = pd.merge(df, div_df, on='date', how='left')
        df['dividend_amount'] = df['dividend_amount'].fillna(0)
        div_dates = df.loc[df['dividend_amount'] > 0, 'date']
        df['last_div_date'] = pd.Series(index=df.index, dtype='datetime64[ns]')
        df.loc[div_dates.index, 'last_div_date'] = div_dates
        df['last_div_date'] = df['last_div_date'].ffill()
        df['days_since_dividend'] = (df['date'] - df['last_div_date']).dt.days
        df['days_since_dividend'] = df['days_since_dividend'].fillna(9999)
        
        df['last_div_amount'] = df['dividend_amount'].replace(0, np.nan).ffill().fillna(0).astype(float)
        df['dividend_yield'] = (df['last_div_amount'] / df['close']).astype(float)
        
        df['next_div_date'] = pd.Series(index=df.index, dtype='datetime64[ns]')
        df.loc[div_dates.index, 'next_div_date'] = div_dates
        df['next_div_date'] = df['next_div_date'].bfill()
        df['days_to_next_dividend'] = (df['next_div_date'] - df['date']).dt.days
        df['is_ex_dividend_week'] = ((df['days_to_next_dividend'] >= 0) & (df['days_to_next_dividend'] <= 7)).astype(int)
        df.drop(columns=['dividend_amount', 'last_div_date', 'last_div_amount', 'next_div_date', 'days_to_next_dividend'], inplace=True)
    else:
        df['days_since_dividend'] = 9999
        df['dividend_yield'] = 0.0
        df['is_ex_dividend_week'] = 0
        
    df['daily_spread'] = (df['high'] - df['low']) / df['close']
    high_low_diff = df['high'] - df['low']
    df['close_pos'] = np.where(high_low_diff == 0, 0.5, (df['close'] - df['low']) / high_low_diff)
    
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    cols_to_drop = ['open', 'high', 'low', 'volume']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    df['target_return_t1'] = (df['close'].shift(-1) - df['close']) / df['close']
    df.dropna(subset=['target_return_t1'], inplace=True)
    return df

def run_ablation():
    cutoff_str, valid_tickers = choose_global_cutoff(test_trading_days=250, min_train_trading_days=500)
    cutoff_date = pd.to_datetime(cutoff_str)
    
    configs = {
        "Baseline (Original)": [],
        "Baseline - Close - Raw OBV": [],
        "Full Patched": [],
        "Naive Persistence": []
    }
    
    for ticker in valid_tickers:
        # 1. Baseline
        df_base = get_original_features(ticker)
        if not df_base.empty:
            exclude_base = ['date', 'created_at', 'target_return_t1']
            feat_base = [c for c in df_base.columns if c not in exclude_base]
            configs["Baseline (Original)"].append((df_base[feat_base], df_base['target_return_t1'], df_base['close'], ticker, df_base['date']))
            
            exclude_minus = exclude_base + ['close', 'obv']
            feat_minus = [c for c in df_base.columns if c not in exclude_minus]
            configs["Baseline - Close - Raw OBV"].append((df_base[feat_minus], df_base['target_return_t1'], df_base['close'], ticker, df_base['date']))
            
        # 3. Full Patched & Naive Persistence
        file_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
        if os.path.exists(file_path):
            df_full = pd.read_csv(file_path)
            if 'target_return_t1' not in df_full.columns:
                df_full['target_return_t1'] = (df_full['close'].shift(-1) - df_full['close']) / df_full['close']
                df_full.dropna(subset=['target_return_t1'], inplace=True)
                
            df_full['sector'] = df_full['ticker'].map(TICKER_SECTORS)
            exclude_full = ['date', 'created_at', 'target_return_t1', 'close']
            feat_full = [c for c in df_full.columns if c not in exclude_full]
            
            df_full['date'] = pd.to_datetime(df_full['date'])
            configs["Full Patched"].append((df_full[feat_full], df_full['target_return_t1'], df_full['close'], ticker, df_full['date']))
            configs["Naive Persistence"].append((df_full[feat_full], df_full['target_return_t1'], df_full['close'], ticker, df_full['date']))
            
    results = []
    ticker_results = []
    final_model = None
    final_X_all = None
    
    for name, data_list in configs.items():
        if not data_list:
            continue
            
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []
        close_test_list = []
        ticker_test_list = []
        
        for X, y, close, ticker_name, dates in data_list:
            train_mask = dates <= cutoff_date
            test_mask = dates > cutoff_date
            
            X_train_list.append(X[train_mask])
            X_test_list.append(X[test_mask])
            y_train_list.append(y[train_mask])
            y_test_list.append(y[test_mask])
            close_test_list.append(close[test_mask])
            ticker_test_list.extend([ticker_name] * test_mask.sum())
            
        X_train = pd.concat(X_train_list, ignore_index=True)
        X_test = pd.concat(X_test_list, ignore_index=True)
        
        # Cast categorical columns after concat
        X_train['ticker'] = X_train['ticker'].astype('category')
        X_test['ticker'] = pd.Categorical(X_test['ticker'], categories=X_train['ticker'].cat.categories)
        if 'sector' in X_train.columns:
            X_train['sector'] = X_train['sector'].astype('category')
            X_test['sector'] = pd.Categorical(X_test['sector'], categories=X_train['sector'].cat.categories)
            
        y_train = pd.concat(y_train_list, ignore_index=True)
        y_test = pd.concat(y_test_list, ignore_index=True)
        close_test_all = pd.concat(close_test_list, ignore_index=True)
        
        if name == "Naive Persistence":
            predictions_return = np.zeros_like(y_test)
        else:
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, objective='reg:squarederror', enable_categorical=True)
            model.fit(X_train, y_train)
            predictions_return = model.predict(X_test)
        
        predicted_prices = close_test_all * (1 + predictions_return)
        actual_prices = close_test_all * (1 + y_test)
        
        actual_dir = np.sign(y_test)
        pred_dir = np.sign(predictions_return)
        dir_acc = (actual_dir == pred_dir).mean() * 100
        
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
        
        results.append({
            "Config": name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Directional Accuracy (%)": dir_acc
        })
        
        df_eval = pd.DataFrame({
            'ticker': ticker_test_list,
            'actual_price': actual_prices.values,
            'predicted_price': predicted_prices.values,
            'actual_return': y_test.values,
            'predicted_return': predictions_return
        })
        
        for t in valid_tickers:
            df_t = df_eval[df_eval['ticker'] == t]
            if not df_t.empty:
                t_mae = mean_absolute_error(df_t['actual_price'], df_t['predicted_price'])
                t_rmse = np.sqrt(mean_squared_error(df_t['actual_price'], df_t['predicted_price']))
                t_mape = mean_absolute_percentage_error(df_t['actual_price'], df_t['predicted_price']) * 100
                t_dir = (np.sign(df_t['actual_return']) == np.sign(df_t['predicted_return'])).mean() * 100
                ticker_results.append({
                    "Config": name,
                    "Ticker": t,
                    "MAE": t_mae,
                    "RMSE": t_rmse,
                    "MAPE": t_mape,
                    "Directional Accuracy (%)": t_dir
                })
        
        if name == "Full Patched":
            final_model = model
            final_X_all = X_train
            
    print("\n--- Global Ablation Study Results ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n--- Per-Ticker Ablation Study Results ---")
    ticker_df = pd.DataFrame(ticker_results)
    print(ticker_df.to_string(index=False))
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary_csv = os.path.join(REPORTS_DIR, "ablation_summary.csv")
    ticker_csv = os.path.join(REPORTS_DIR, "ablation_by_ticker.csv")
    results_df.to_csv(summary_csv, index=False)
    ticker_df.to_csv(ticker_csv, index=False)
    print(f"\nSaved CSV reports to {REPORTS_DIR}")
    
    patched_global = results_df[results_df['Config'] == 'Full Patched'].iloc[0]
    naive_global = results_df[results_df['Config'] == 'Naive Persistence'].iloc[0]
    
    diff_dir_acc = patched_global['Directional Accuracy (%)'] - naive_global['Directional Accuracy (%)']
    diff_mape = patched_global['MAPE'] - naive_global['MAPE']
    
    print("\n### Conclusion")
    print(f"**Globally**, Full Patched beats Naive Persistence in Directional Accuracy by {diff_dir_acc:+.2f} percentage points.")
    print(f"For MAPE, the difference is {diff_mape:+.3f}% (negative is better).")
    
    print("\n**Per-Ticker Highlights (Full Patched vs Naive):**")
    for t in valid_tickers:
        patched_filter = ticker_df[(ticker_df['Config'] == 'Full Patched') & (ticker_df['Ticker'] == t)]
        naive_filter = ticker_df[(ticker_df['Config'] == 'Naive Persistence') & (ticker_df['Ticker'] == t)]
        if not patched_filter.empty and not naive_filter.empty:
            t_patched = patched_filter.iloc[0]
            t_naive = naive_filter.iloc[0]
            t_diff_dir = t_patched['Directional Accuracy (%)'] - t_naive['Directional Accuracy (%)']
            t_diff_mape = t_patched['MAPE'] - t_naive['MAPE']
            print(f" - {t}: DirAcc {t_diff_dir:+.2f} pts, MAPE {t_diff_mape:+.3f}%")
    
    if final_model is not None:
        importance = final_model.feature_importances_
        features = final_X_all.columns
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        print("\n--- Top 15 Feature Importances (Full Patched) ---")
        print(feat_imp.head(15))
        
        plt.figure(figsize=(12, 8))
        top_15 = feat_imp.head(15)
        plt.barh(top_15['Feature'][::-1], top_15['Importance'][::-1] * 100, color='mediumseagreen')
        plt.xlabel('Importance (%)')
        plt.title('Top 15 XGBoost Feature Importances (Full Patched)')
        plt.tight_layout()
        os.makedirs(REPORTS_DIR, exist_ok=True)
        plot_path = os.path.join(REPORTS_DIR, "ablation_feature_importance.png")
        plt.savefig(plot_path)
        print(f"\nSaved feature importance chart to {plot_path}")

if __name__ == '__main__':
    run_ablation()
