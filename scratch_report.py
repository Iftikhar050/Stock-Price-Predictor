import pandas as pd
import glob
from pathlib import Path

cols_touched = [
    # KIBOR
    "kibor_1w", "kibor_1m", "kibor_3m", "kibor_6m", "kibor_1y",
    # Bulletin
    "private_sector_credit_growth", "banking_deposits_growth", 
    "sbp_omo_net_outstanding", "t_bill_cutoff_3m", "t_bill_cutoff_6m",
    "forward_usd_pkr_3m", "reer_index", "external_debt_total_usd_bn",
    "m2_money_supply", "sbp_reserves", "monthly_remittances",
    # NCCPL FIPI/LIPI
    "lipi_individuals_net", "lipi_companies_net", "lipi_banks_net", "lipi_nbfc_net",
    "lipi_mutual_funds_net", "lipi_insurance_net", "fipi_foreign_corporate_net",
    "fipi_foreign_individual_net", "fipi_overseas_pakistani_net", "lipi_broker_net",
    # Fundamentals
    "revenue", "net_income", "eps", "total_assets", "total_debt", "ebitda", "gross_profit"
]

def generate_report(ticker):
    print(f"\n{'='*50}\n REPORT FOR {ticker}\n{'='*50}")
    master_file = f"data/processed/{ticker}_master.csv"
    if not os.path.exists(master_file):
        print(f"File {master_file} not found.")
        return

    df = pd.read_csv(master_file, low_memory=False)
    
    report_data = []
    
    for col in cols_touched:
        if col not in df.columns:
            print(f"Column missing from dataset: {col}")
            continue
            
        total = len(df)
        null_count = df[col].isna().sum()
        null_pct = (null_count / total) * 100
        unique_count = df[col].nunique()
        
        # Determine source (we assume manual/live/missing based on is_missing flags if they exist)
        # SBP EasyData puts {col}_is_missing columns in.
        missing_flag_col = f"{col}_is_missing"
        
        source = "Unknown"
        if missing_flag_col in df.columns:
            is_missing_count = df[missing_flag_col].sum()
            if is_missing_count == total:
                source = "Missing (Live & Manual Failed)"
            elif is_missing_count > 0:
                source = f"Mixed (Live/Manual/Missing)"
            else:
                source = "Live/Manual"
        else:
            if null_count == total:
                source = "Missing (All NaNs)"
            elif null_count > 0:
                source = "Mixed (Partial NaNs)"
            else:
                source = "Populated"
                
        report_data.append({
            "Column": col,
            "Null %": f"{null_pct:.1f}%",
            "Unique Vals": unique_count,
            "Source Status": source
        })
        
    report_df = pd.DataFrame(report_data)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    generate_report("PSO")
    generate_report("MEBL")
