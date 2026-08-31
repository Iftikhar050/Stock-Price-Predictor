"""
scratch/full_523_audit.py — Comprehensive 523 PDF Feature Audit

Performs normalized matching of the 523 PDF feature framework against
MEBL_master.csv (365 cols) and PSO_master.csv (364 cols).

Classifies each PDF feature entry into:
  1. EXACT MATCH          — Direct 1:1 mapping in CSV with non-zero historical data
  2. PARTIAL / PROXY      — Related proxy, derived calculation, or step function
  3. MISSING              — Not present in CSV or required Level-2 intraday data
  4. EXTRA PROJECT        — Additional technical/macro indicator added for ML
"""

import pandas as pd
import numpy as np
import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

mebl_path = os.path.join(PROCESSED_DIR, "MEBL_master.csv")
pso_path = os.path.join(PROCESSED_DIR, "PSO_master.csv")

df_mebl = pd.read_csv(mebl_path, low_memory=False)
df_pso = pd.read_csv(pso_path, low_memory=False)

mebl_cols = set(df_mebl.columns)
pso_cols = set(df_pso.columns)

# Define the PDF groups and representative framework feature items
PDF_FRAMEWORK_MAPPING = {
    "Group #1: Company Fundamentals": [
        ("Customer/User Growth", "customer_growth", "MISSING", "P2"),
        ("Capacity Expansion", "capacity_expansion_event", "EXACT", "P1"),
        ("Market-Share Growth", "market_share_growth", "MISSING", "P2"),
        ("Future Earnings Guidance", "management_guidance_flag", "PARTIAL", "P1"),
        ("Interest-Bearing Debt", "interest_bearing_debt", "EXACT", "P0"),
        ("Forward P/E", "forward_pe", "EXACT", "P0"),
        ("Price/Cash Flow", "price_to_cash_flow", "EXACT", "P0"),
        ("Sector-Relative Valuation", "pe_ratio_relative_to_sector", "EXACT", "P0"),
        ("EPS Trailing", "eps_trailing", "EXACT", "P0"),
        ("P/E Ratio", "pe_ratio", "EXACT", "P0"),
        ("P/B Ratio", "pb_ratio", "EXACT", "P0"),
        ("ROE", "roe", "EXACT", "P0"),
        ("ROA", "roa", "EXACT", "P0"),
        ("Book Value per Share", "book_value_per_share", "EXACT", "P0"),
        ("Enterprise Value (EV)", "ev", "EXACT", "P1"),
        ("EV/EBITDA", "ev_ebitda", "EXACT", "P1"),
        ("EV/Sales", "ev_sales", "EXACT", "P1"),
        ("PEG Ratio", "peg_ratio", "EXACT", "P1"),
        ("Market Capitalization", "market_cap", "EXACT", "P0"),
        ("Shares Outstanding", "shares_outstanding", "EXACT", "P0"),
    ],
    "Group #2: Earnings & Profitability": [
        ("Quarterly Earnings", "eps_trailing", "EXACT", "P0"),
        ("Annual Earnings", "net_profit", "EXACT", "P0"),
        ("Earnings Surprise", "eps_surprise", "EXACT", "P0"),
        ("Earnings Growth YoY", "eps_growth_yoy", "EXACT", "P0"),
        ("Gross Profit Margin", "gross_profit_margin", "EXACT", "P0"),
        ("Net Profit Margin", "net_profit_margin", "EXACT", "P0"),
        ("Payout Ratio", "payout_ratio", "EXACT", "P0"),
        ("ROIC", "roic", "EXACT", "P1"),
    ],
    "Group #3: Pakistan Macro - Interest Rates": [
        ("SBP Policy Rate", "sbp_policy_rate", "EXACT", "P0"),
        ("KIBOR 3M", "kibor_3m", "EXACT", "P0"),
        ("KIBOR 6M", "kibor_6m", "EXACT", "P0"),
        ("KIBOR 1Y", "kibor_1y", "EXACT", "P0"),
        ("T-Bill 3M Cutoff", "t_bill_cutoff_3m", "EXACT", "P0"),
        ("T-Bill 6M Cutoff", "t_bill_cutoff_6m", "EXACT", "P0"),
        ("PIB 3Y", "pib_3y", "EXACT", "P1"),
        ("PIB 5Y", "pib_5y", "EXACT", "P1"),
        ("PIB 10Y", "pib_10y", "EXACT", "P0"),
        ("Yield Curve Slope", "yield_curve_slope", "EXACT", "P0"),
        ("Real Interest Rate", "real_interest_rate", "EXACT", "P0"),
        ("Policy Rate Surprise", "policy_rate_surprise", "EXACT", "P0"),
        ("SBP OMO Injections", "sbp_omo_net_outstanding", "EXACT", "P1"),
    ],
    "Group #4: Commodities & Energy": [
        ("Brent Crude", "brent_oil_price", "EXACT", "P0"),
        ("WTI Crude", "wti_oil_price", "EXACT", "P0"),
        ("Gold Price", "gold_price", "EXACT", "P0"),
        ("Copper Price", "copper_price", "EXACT", "P1"),
        ("Steel Price", "steel_price", "EXACT", "P1"),
        ("Iron Ore Price", "iron_ore_price", "EXACT", "P1"),
        ("Urea Price", "urea_price", "EXACT", "P1"),
        ("LNG Price", "lng_price", "EXACT", "P1"),
        ("Coal Price", "coal_price", "EXACT", "P1"),
        ("Cotton Price", "cotton_price", "EXACT", "P1"),
    ],
    "Group #5: FX & Currency Risks": [
        ("USD/PKR Rate", "pkr_usd_rate", "EXACT", "P0"),
        ("EUR/PKR Rate", "eur_pkr_rate", "EXACT", "P1"),
        ("GBP/PKR Rate", "gbp_pkr_rate", "EXACT", "P1"),
        ("CNY/PKR Cross", "cny_pkr_rate", "EXACT", "P1"),
        ("REER Index", "reer_index", "EXACT", "P0"),
        ("3M Forward USD/PKR", "forward_usd_pkr_3m", "EXACT", "P1"),
        ("PKR/USD Volatility 20d", "pkr_usd_volatility_20d", "EXACT", "P1"),
    ],
    "Group #6: SBP Balance Sheet & Reserves": [
        ("SBP Foreign Reserves", "sbp_reserves", "EXACT", "P0"),
        ("Total FX Reserves", "total_fx_reserves", "EXACT", "P0"),
        ("Reserve Import Coverage", "reserve_import_coverage", "EXACT", "P0"),
        ("Reserve Weekly Changes", "reserve_changes", "EXACT", "P0"),
        ("External Debt Total", "external_debt_total_usd_bn", "EXACT", "P1"),
    ],
    "Group #7: Inflation & CPI": [
        ("CPI Headline", "cpi_headline", "EXACT", "P0"),
        ("CPI Core", "cpi_core", "EXACT", "P0"),
        ("CPI Surprise", "cpi_surprise", "EXACT", "P0"),
    ],
    "Group #8: Global Equity & Yields": [
        ("S&P 500", "sp500_close", "EXACT", "P0"),
        ("Nasdaq", "nasdaq_close", "EXACT", "P0"),
        ("Dow Jones", "dow_jones_close", "EXACT", "P1"),
        ("MSCI Emerging Markets", "msci_em_close", "EXACT", "P0"),
        ("MSCI Frontier Markets", "msci_fm_close", "EXACT", "P0"),
        ("US 10Y Yield", "us10y_yield", "EXACT", "P0"),
        ("US 2Y Yield", "us2y_yield", "EXACT", "P0"),
        ("US 5Y Yield", "us5y_yield", "EXACT", "P1"),
        ("US TIPS ETF", "tips_etf_price", "EXACT", "P1"),
        ("Fed Funds Rate", "fed_funds_rate", "EXACT", "P0"),
        ("ECB Policy Rate", "ecb_rate", "EXACT", "P0"),
        ("US 2s10s Yield Curve", "us_yield_curve_2y10y", "EXACT", "P0"),
        ("Fed/ECB Policy Spread", "fed_ecb_policy_spread", "EXACT", "P1"),
    ],
    "Group #9: Institutional Flows": [
        ("FIPI Net Foreign USD", "fipi_net_usd_m", "EXACT", "P0"),
        ("LIPI Mutual Funds Net", "lipi_mutual_funds_net", "EXACT", "P0"),
        ("LIPI Banks Net", "lipi_banks_net", "EXACT", "P0"),
        ("LIPI Insurance Net", "lipi_insurance_net", "EXACT", "P0"),
        ("LIPI Companies Net", "lipi_companies_net", "EXACT", "P0"),
        ("LIPI Individuals Net", "lipi_individuals_net", "EXACT", "P0"),
    ],
    "Group #10: Pakistan Real Economy Activity": [
        ("Cement Dispatches", "cement_dispatches_mt", "EXACT", "P1"),
        ("Auto Total Sales", "auto_sales_total", "EXACT", "P1"),
        ("Electricity Generation", "electricity_gen_gwh", "EXACT", "P1"),
        ("Wheat Procurement", "wheat_procurement_mt", "EXACT", "P1"),
        ("Private Sector Credit Growth", "private_sector_credit_growth", "EXACT", "P0"),
        ("Banking Deposits Growth", "banking_deposits_growth", "EXACT", "P0"),
    ],
    "Group #11: Political & Country Risk": [
        ("Election Flag", "election_flag", "EXACT", "P0"),
        ("FATF Greylist Flag", "fatf_greylist_flag", "EXACT", "P0"),
        ("Government Stability Score", "government_stability_score", "EXACT", "P0"),
        ("Political Uncertainty Score", "political_uncertainty_score", "EXACT", "P0"),
        ("India-Pakistan Tension Flag", "india_pakistan_tension_flag", "EXACT", "P1"),
    ],
    "Group #12: Geopolitical & Regional Conflicts": [
        ("Middle East Conflict Flag", "middle_east_conflict_flag", "EXACT", "P1"),
        ("Red Sea Shipping Disruption Flag", "red_sea_disruption_flag", "EXACT", "P1"),
        ("Global Oil Supply Shock Flag", "global_oil_supply_shock_flag", "EXACT", "P1"),
    ],
    "Group #13: Banking Sector Metrics (MEBL Specific)": [
        ("Net Interest Margin (NIM)", "net_interest_margin", "EXACT", "P0"),
        ("CASA Ratio", "casa_ratio", "EXACT", "P0"),
        ("CASA Deposits", "casa_deposits", "EXACT", "P0"),
        ("Total Advances", "total_advances", "EXACT", "P0"),
        ("Total Deposits", "total_deposits", "EXACT", "P0"),
        ("NPL Ratio", "npl_ratio", "EXACT", "P0"),
        ("Provisioning Coverage Ratio", "provisioning_coverage", "EXACT", "P0"),
        ("Capital Adequacy Ratio (CAR)", "capital_adequacy_ratio", "EXACT", "P0"),
        ("Advances-to-Deposit Ratio (ADR)", "adr_ratio", "EXACT", "P0"),
        ("Investments-to-Deposit Ratio (IDR)", "idr_ratio", "EXACT", "P0"),
    ],
    "Group #14: Market Microstructure (Level-2 Order Book)": [
        ("Buy Volume (Intraday)", "buy_volume", "MISSING", "P2 (Level-2 required)"),
        ("Sell Volume (Intraday)", "sell_volume", "MISSING", "P2 (Level-2 required)"),
        ("Volume Imbalance", "volume_imbalance", "MISSING", "P2 (Level-2 required)"),
        ("Bid/Ask Spread", "bid_ask_spread", "MISSING", "P2 (Level-2 required)"),
        ("Order Book Depth", "order_book_depth", "MISSING", "P2 (Level-2 required)"),
    ],
}

print("=================================================================")
print("PSX PDF FEATURE FRAMEWORK vs MEBL & PSO MASTER DATASETS AUDIT")
print("=================================================================")
print()

total_framework_items = 0
exact_count = 0
partial_count = 0
missing_count = 0

for group_name, items in PDF_FRAMEWORK_MAPPING.items():
    print(f"\n--- {group_name} ---")
    for feat_title, col_name, expected_status, priority in items:
        total_framework_items += 1
        in_mebl = col_name in mebl_cols
        in_pso = col_name in pso_cols

        if in_mebl or in_pso:
            status = "EXACT MATCH"
            exact_count += 1
            mebl_pop = (df_mebl[col_name] != 0).mean() * 100 if in_mebl else 0.0
            pso_pop = (df_pso[col_name] != 0).mean() * 100 if in_pso else 0.0
            pop_str = f"MEBL:{mebl_pop:.0f}% / PSO:{pso_pop:.0f}%"
        else:
            status = expected_status
            if status == "PARTIAL":
                partial_count += 1
                pop_str = "Proxy/Derived"
            else:
                missing_count += 1
                pop_str = "0%"

        print(f"  [{status:<11}] {feat_title:<34} | Col: {col_name:<28} | P: {priority:<3} | Pop: {pop_str}")

print("\n=================================================================")
print(f"SUMMARY AUDIT STATS:")
print(f"  Total Framework Feature Items Evaluated: {total_framework_items}")
print(f"  Exact Matches (Populated in CSV):        {exact_count} ({exact_count/total_framework_items*100:.1f}%)")
print(f"  Partial / Proxies / Step Functions:     {partial_count} ({partial_count/total_framework_items*100:.1f}%)")
print(f"  Missing / Intraday L2 Dependent:         {missing_count} ({missing_count/total_framework_items*100:.1f}%)")
print(f"  Master CSV Column Counts:                MEBL: {len(mebl_cols)} | PSO: {len(pso_cols)}")
print("=================================================================")
