import pandas as pd

def verify(ticker):
    print(f"--- Verifying {ticker} ---")
    df = pd.read_csv(f"data/processed/{ticker}_master.csv")
    
    # 1. SBP columns
    print("SBP Policy Rate non-zero count:", df[df['sbp_policy_rate'] != 0.0]['sbp_policy_rate'].count())
    print("SBP Policy Rate missing flag mean:", df['sbp_policy_rate_is_missing'].mean() if 'sbp_policy_rate_is_missing' in df.columns else "Not Found")
    
    # 2. Pytrends
    search_col = f"search_trend_{ticker.lower()}"
    print(f"{search_col} valid count:", df[search_col].notna().sum() if search_col in df.columns else "Not Found")
    
    # 3. Redundant columns
    print("interest_bearing_debt in columns:", 'interest_bearing_debt' in df.columns)
    print("imf_primary_balance_pct_gdp in columns:", 'imf_primary_balance_pct_gdp' in df.columns)
    
    # 4. Political sentiment
    print("political_news_sentiment_3d non-zero count:", df[df['political_news_sentiment_3d'] != 0.0]['political_news_sentiment_3d'].count() if 'political_news_sentiment_3d' in df.columns else "Not Found")
    
    # 5. NaN event
    print("NaN_event in columns:", 'NaN_event' in df.columns)
    print("nan_event in columns:", 'nan_event' in df.columns)

if __name__ == "__main__":
    verify("PSO")
    verify("MEBL")
