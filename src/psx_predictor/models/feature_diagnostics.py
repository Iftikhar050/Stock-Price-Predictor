import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "figures")
TICKERS = ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']

def run_diagnostics():
    print("Loading data for diagnostics...")
    X_list = []
    
    for ticker in TICKERS:
        file_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        if 'target_return_t1' not in df.columns:
            df['target_return_t1'] = (df['close'].shift(-1) - df['close']) / df['close']
            df.dropna(subset=['target_return_t1'], inplace=True)
            
        exclude_cols = ['ticker', 'date', 'created_at', 'target_return_t1', 'close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X_list.append(df[feature_cols])
        
    if not X_list:
        print("No data available.")
        return
        
    X_all = pd.concat(X_list, ignore_index=True)
    
    # 1. Correlation Matrix
    print(f"\nComputing correlation matrix for {len(X_all.columns)} features...")
    corr_matrix = X_all.corr()
    
    # Plot Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plot_path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved correlation heatmap to {plot_path}")
    
    # Flag High Correlations
    print("\n--- Highly Correlated Feature Pairs (|corr| > 0.85) ---")
    corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", key=abs, ascending=False)
    # Remove self correlations
    corr_pairs = corr_pairs[corr_pairs != 1.0]
    
    seen = set()
    for pair, val in corr_pairs.items():
        if abs(val) > 0.85:
            # Sort the tuple so (A,B) and (B,A) are considered the same
            sorted_pair = tuple(sorted(pair))
            if sorted_pair not in seen:
                print(f"{sorted_pair[0]:<25} and {sorted_pair[1]:<25} : {val:.3f}")
                seen.add(sorted_pair)
                
    # 2. VIF Calculation
    print("\n--- Variance Inflation Factor (VIF) ---")
    print("Calculating VIF (this may take a moment)...")
    # Drop NaNs or infinite values if any exist
    X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_all.dropna(inplace=True)
    
    # Subsample if dataset is too large to speed up VIF
    if len(X_all) > 10000:
        X_sample = X_all.sample(n=10000, random_state=42)
    else:
        X_sample = X_all
        
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_sample.columns
    
    # Calculate VIF
    # Note: VIF can be problematic if there's perfect multicollinearity.
    try:
        vif_data["VIF"] = [variance_inflation_factor(X_sample.values, i) for i in range(len(X_sample.columns))]
        vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
        print(vif_data.head(20))
        print("\nNote: VIF > 10 indicates high multicollinearity.")
    except np.linalg.LinAlgError:
        print("Could not calculate VIF due to perfect multicollinearity in the data.")

if __name__ == '__main__':
    run_diagnostics()
