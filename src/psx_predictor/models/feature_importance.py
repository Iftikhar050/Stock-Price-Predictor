import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
from src.psx_predictor.db.repository import get_active_tickers
TICKERS = get_active_tickers()

def analyze_feature_importance():
    X_list, y_list = [], []
    
    for ticker in TICKERS:
        file_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
        if not os.path.exists(file_path):
            print(f"Skipping {ticker}, feature file not found.")
            continue
            
        df = pd.read_csv(file_path)
        df['target_return_t1'] = (df['close'].shift(-1) - df['close']) / df['close']
        df.dropna(subset=['target_return_t1'], inplace=True)
        
        exclude_cols = ['ticker', 'date', 'created_at', 'target_return_t1', 'close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X_list.append(df[feature_cols])
        y_list.append(df['target_return_t1'])
        
    if not X_list:
        print("No data available. Please run build_features.py first.")
        return
        
    X_all = pd.concat(X_list, ignore_index=True)
    y_all = pd.concat(y_list, ignore_index=True)
    
    print(f"Training XGBoost on {len(X_all)} samples to analyze feature importance...")
    
    model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_all, y_all)
    
    importance = model.feature_importances_
    features = X_all.columns
    
    # Sort features by importance
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\n--- Feature Importances ---")
    for idx, row in feat_imp.iterrows():
        print(f"{idx+1}. {row['Feature']:<25} : {row['Importance']*100:.2f}%")
        
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1] * 100, color='skyblue')
    plt.xlabel('Importance (%)')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plot_path = os.path.join(ROOT_DIR, "reports", "figures", "feature_importance.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"\nSaved feature importance chart to {plot_path}")

if __name__ == '__main__':
    analyze_feature_importance()
