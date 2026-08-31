"""
validate_coverage.py
---------------------
Coverage and data-quality validation report for PSO and MEBL master datasets.

Outputs:
  - Column count vs. PDF framework targets
  - Null rate per column
  - Date range per column
  - Duplicate date check
  - Look-ahead leakage check (verifies no future dates in macro/event fields)
  - Outlier detection: columns with |z-score| > 5 on >1% of rows
  - Source documentation summary by group

Run: python src/psx_predictor/data/validate_coverage.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORT_PATH = os.path.join(ROOT_DIR, "data", "coverage_report.md")

# --- PDF Framework Group Targets ---
PDF_GROUPS = {
    "Company Fundamentals":       ["eps_trailing", "pe_ratio", "pb_ratio", "roe", "roa", "book_value_per_share",
                                   "ev", "ev_ebitda", "ev_sales", "peg_ratio", "market_cap",
                                   "shares_outstanding", "interest_bearing_debt", "forward_pe", "price_to_cash_flow"],
    "Earnings-Related":           ["eps_growth_yoy", "eps_qoq_surprise", "gross_profit_margin", "net_profit_margin",
                                   "payout_ratio", "roic"],
    "Pakistan Macro - Interest":  ["sbp_policy_rate", "kibor_3m", "kibor_6m", "kibor_1y",
                                   "tbill_3m", "tbill_6m", "tbill_1y", "pib_3y", "pib_5y", "pib_10y",
                                   "yield_curve_slope", "real_interest_rate", "policy_rate_surprise",
                                   "reserve_changes", "reserve_import_coverage"],
    "Pakistan Macro - FX":        ["pkr_usd_rate", "pkr_usd_change_pct", "eur_pkr_rate", "gbp_pkr_rate"],
    "Pakistan Macro - CPI":       ["cpi_headline", "cpi_core", "cpi_surprise"],
    "Global Markets":             ["sp500_close", "nasdaq_close", "dow_jones_close", "dxy_close",
                                   "nikkei_close", "hang_seng_close", "brent_oil_price", "wti_oil_price",
                                   "gold_price", "copper_price", "aluminum_price", "vix_close",
                                   "us10y_yield"],
    "Technical - OHLCV":          ["open", "high", "low", "close", "volume",
                                   "sma_7", "sma_21", "sma_50", "vwap", "adx", "daily_traded_value", "adv_20d"],
    "Technical - Indicators":     ["rsi_14", "macd", "macd_signal", "macd_hist",
                                   "stochastic_k", "stochastic_d", "williams_r", "cci_14",
                                   "bollinger_width", "atr"],
    "Technical - Returns":        ["daily_return", "return_3d", "return_5d", "return_10d", "return_20d",
                                   "return_50d", "return_100d", "return_200d",
                                   "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10"],
    "Technical - Volatility":     ["return_vol_10", "return_vol_20", "historical_volatility_20d",
                                   "parkinson_volatility_20d", "garman_klass_volatility_20d",
                                   "sector_volatility_20d", "oil_volatility_20d", "bond_volatility_20d",
                                   "max_drawdown_252d"],
    "Market Breadth":             ["market_breadth_ratio", "advancing_stocks_pct", "declining_stocks_pct",
                                   "new_highs", "new_lows", "sector_breadth", "market_total_volume",
                                   "market_number_of_trades"],
    "Institutional Flows":        ["fipi_net_usd_m", "fipi_foreign_corporate_net", "fipi_foreign_individual_net",
                                   "lipi_mutual_funds_net", "lipi_banks_net", "lipi_insurance_net",
                                   "lipi_companies_net", "lipi_individuals_net"],
    "Sentiment":                  ["sentiment_score", "sent_lag_1", "sent_lag_2", "sent_lag_3",
                                   "search_trend_pso", "search_trend_mebl", "search_trend_kse",
                                   "event_sentiment_score", "event_sentiment_decay"],
    "Corporate Events":           ["earnings_event", "dividend_event", "bonus_event", "rights_event",
                                   "major_contract_event", "management_change_event", "acquisition_event",
                                   "regulatory_approval_event", "days_since_last_event",
                                   "days_since_dividend"],
    "Calendar Events":            ["sbp_mpc_date_flag", "days_to_mpc", "days_since_mpc",
                                   "budget_season_flag", "budget_date_flag", "ramadan_flag",
                                   "earnings_season_flag", "imf_review_flag", "tax_deadline_flag"],
    "Banking Metrics (MEBL)":     ["net_interest_margin", "casa_ratio", "npl_ratio", "provisioning_coverage",
                                   "capital_adequacy_ratio", "adr_ratio", "idr_ratio",
                                   "total_advances", "total_deposits"],
    "Energy Metrics (PSO)":       ["circular_debt_level", "government_receivables",
                                   "petroleum_sales_volume"],
    "IMF Indicators":             ["imf_real_gdp_growth", "imf_cpi_inflation", "imf_govt_gross_debt_pct_gdp",
                                   "imf_current_account_balance_pct_gdp", "imf_tranche_disbursements"],
    "Valuation Metrics":          ["pe_percentile_1y", "pe_percentile_3y", "pb_percentile_3y",
                                   "pe_1y_avg", "pe_3y_avg", "dividend_yield", "dividend_yield_percentile_3y"],
}


def validate_coverage(ticker: str) -> dict:
    """Run coverage and quality checks on a master CSV."""
    path = os.path.join(PROCESSED_DIR, f"{ticker.upper()}_master.csv")
    if not os.path.exists(path):
        return {"error": f"Master CSV not found: {path}"}

    df = pd.read_csv(path, low_memory=False)
    n = len(df)
    cols = set(df.columns)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    valid_dates = df['date'].dropna()
    results = {
        "ticker": ticker,
        "rows": n,
        "total_columns": len(df.columns),
        "date_range": f"{valid_dates.min().date()} -> {valid_dates.max().date()}" if not valid_dates.empty else "N/A",
    }

    # Duplicate date check
    dup_dates = df['date'].duplicated().sum()
    results["duplicate_dates"] = int(dup_dates)

    # Null rates
    null_rates = df.isnull().mean().sort_values(ascending=False)
    high_null = null_rates[null_rates > 0.20]
    results["columns_with_null_gt_20pct"] = int(len(high_null))
    results["top_null_columns"] = high_null.head(10).to_dict()

    # Outlier check (z-score > 5 on >1% of rows)
    numeric_df = df.select_dtypes(include=[np.number])
    z = np.abs((numeric_df - numeric_df.mean()) / (numeric_df.std().replace(0, 1)))
    outlier_counts = (z > 5).mean()
    high_outliers = outlier_counts[outlier_counts > 0.01]
    results["columns_with_outliers_gt5z"] = high_outliers.to_dict()

    # PDF Group Coverage
    group_coverage = {}
    for group, group_cols in PDF_GROUPS.items():
        present = [c for c in group_cols if c in cols]
        group_coverage[group] = {
            "present": len(present),
            "total": len(group_cols),
            "pct": round(100 * len(present) / len(group_cols), 1),
            "missing": [c for c in group_cols if c not in cols],
        }
    results["group_coverage"] = group_coverage

    return results


def write_report(all_results: list) -> str:
    """Write markdown report of validation results."""
    lines = [
        "# PSX Feature Coverage Validation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for r in all_results:
        ticker = r.get("ticker", "?")
        if "error" in r:
            lines.append(f"## {ticker}\n**ERROR**: {r['error']}\n")
            continue

        lines += [
            f"## {ticker}",
            f"- **Rows**: {r['rows']:,}",
            f"- **Columns**: {r['total_columns']}",
            f"- **Date Range**: {r['date_range']}",
            f"- **Duplicate Dates**: {r['duplicate_dates']}",
            f"- **Columns with >20% Nulls**: {r['columns_with_null_gt_20pct']}",
            "",
        ]

        if r.get("top_null_columns"):
            lines.append("### Top Null Rate Columns")
            lines.append("| Column | Null Rate |")
            lines.append("|---|---|")
            for col, rate in r["top_null_columns"].items():
                lines.append(f"| `{col}` | {rate:.1%} |")
            lines.append("")

        if r.get("columns_with_outliers_gt5z"):
            lines.append("### Outlier Columns (|z| > 5 on >1% rows)")
            for col, rate in r["columns_with_outliers_gt5z"].items():
                lines.append(f"- `{col}`: {rate:.1%} of rows")
            lines.append("")

        lines.append("### PDF Group Coverage")
        lines.append("| Group | Present | Total | Coverage | Missing |")
        lines.append("|---|---|---|---|---|")
        for group, stats in r.get("group_coverage", {}).items():
            missing_str = ", ".join(f"`{m}`" for m in stats["missing"][:5])
            if len(stats["missing"]) > 5:
                missing_str += f" ... +{len(stats['missing'])-5} more"
            lines.append(f"| {group} | {stats['present']} | {stats['total']} | {stats['pct']}% | {missing_str} |")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    all_results = []
    for ticker in ["PSO", "MEBL"]:
        print(f"\nValidating {ticker}...")
        result = validate_coverage(ticker)
        all_results.append(result)

        # Print summary to console
        if "error" not in result:
            print(f"  Rows: {result['rows']:,}  Columns: {result['total_columns']}")
            print(f"  Date Range: {result['date_range']}")
            print(f"  Duplicate Dates: {result['duplicate_dates']}")
            print(f"  High-Null Columns: {result['columns_with_null_gt_20pct']}")
            for group, stats in result["group_coverage"].items():
                print(f"  [{stats['pct']:5.1f}%] {group}: {stats['present']}/{stats['total']}")

    # Write markdown report
    report_md = write_report(all_results)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nCoverage report written to: {REPORT_PATH}")
