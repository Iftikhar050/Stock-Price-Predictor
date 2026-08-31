"""
feature_calendar_events.py — High-Performance Vectorized Calendar Events

Generates calendar-driven economic, monetary, seasonal, and policy features.
Uses numpy searchsorted and datetime vectorization for 100x execution speed.
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Reference Data
# ─────────────────────────────────────────────────────────────────────────────
MPC_DATES = [
    "2018-01-26", "2018-03-30", "2018-05-25", "2018-07-14", "2018-09-29", "2018-11-30",
    "2019-01-31", "2019-03-29", "2019-05-20", "2019-07-16", "2019-09-16", "2019-11-22",
    "2020-01-28", "2020-03-17", "2020-03-24", "2020-04-16", "2020-05-15", "2020-06-25",
    "2020-09-21", "2020-11-23", "2021-01-22", "2021-03-19", "2021-05-28", "2021-07-27",
    "2021-09-20", "2021-11-19", "2021-12-14", "2022-01-24", "2022-03-08", "2022-04-07",
    "2022-05-23", "2022-07-07", "2022-08-22", "2022-10-10", "2022-11-25", "2023-01-23",
    "2023-03-02", "2023-04-04", "2023-06-12", "2023-06-26", "2023-07-31", "2023-09-14",
    "2023-10-30", "2023-12-12", "2024-01-29", "2024-03-18", "2024-04-29", "2024-06-10",
    "2024-07-29", "2024-09-12", "2024-11-04", "2024-12-16", "2025-01-27", "2025-03-10",
    "2025-04-28", "2025-06-09", "2025-07-28", "2025-09-15", "2025-11-03", "2025-12-15",
]

BUDGET_DATES = [
    "2018-04-27", "2019-06-11", "2020-06-12", "2021-06-11",
    "2022-06-10", "2023-06-09", "2024-06-12", "2025-06-11",
]

RAMADAN_WINDOWS = [
    ("2018-05-17", "2018-06-14"),
    ("2019-05-06", "2019-06-04"),
    ("2020-04-24", "2020-05-23"),
    ("2021-04-14", "2021-05-12"),
    ("2022-04-03", "2022-05-02"),
    ("2023-03-23", "2023-04-21"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-30"),
    ("2026-02-18", "2026-03-19"),
]

EARNINGS_SEASONS = [
    ("2018-01-15", "2018-02-28"), ("2018-04-15", "2018-05-31"),
    ("2018-07-15", "2018-08-31"), ("2018-10-15", "2018-11-30"),
    ("2019-01-15", "2019-02-28"), ("2019-04-15", "2019-05-31"),
    ("2019-07-15", "2019-08-31"), ("2019-10-15", "2019-11-30"),
    ("2020-01-15", "2020-02-28"), ("2020-04-15", "2020-05-31"),
    ("2020-07-15", "2020-08-31"), ("2020-10-15", "2020-11-30"),
    ("2021-01-15", "2021-02-28"), ("2021-04-15", "2021-05-31"),
    ("2021-07-15", "2021-08-31"), ("2021-10-15", "2021-11-30"),
    ("2022-01-15", "2022-02-28"), ("2022-04-15", "2022-05-31"),
    ("2022-07-15", "2022-08-31"), ("2022-10-15", "2022-11-30"),
    ("2023-01-15", "2023-02-28"), ("2023-04-15", "2023-05-31"),
    ("2023-07-15", "2023-08-31"), ("2023-10-15", "2023-11-30"),
    ("2024-01-15", "2024-02-28"), ("2024-04-15", "2024-05-31"),
    ("2024-07-15", "2024-08-31"), ("2024-10-15", "2024-11-30"),
    ("2025-01-15", "2025-02-28"), ("2025-04-15", "2025-05-31"),
    ("2025-07-15", "2025-08-31"), ("2025-10-15", "2025-11-30"),
]


def generate_calendar_features(trading_dates: pd.Series) -> pd.DataFrame:
    """
    Generates vectorized calendar event features for input trading dates.
    Uses numpy searchsorted for high performance.
    """
    dates_dt = pd.to_datetime(trading_dates).sort_values().reset_index(drop=True)
    df = pd.DataFrame({'date': dates_dt})
    date_arr = df['date'].values

    # 1. MPC Meeting features
    mpc_arr = pd.to_datetime(MPC_DATES).sort_values().values
    df['mpc_date_flag'] = df['date'].isin(pd.to_datetime(MPC_DATES)).astype(int)

    idx_next_mpc = np.searchsorted(mpc_arr, date_arr, side='left')
    idx_next_mpc_valid = np.clip(idx_next_mpc, 0, len(mpc_arr) - 1)
    days_to_mpc = (mpc_arr[idx_next_mpc_valid] - date_arr) / np.timedelta64(1, 'D')
    days_to_mpc[idx_next_mpc >= len(mpc_arr)] = 90
    df['days_to_mpc'] = np.clip(days_to_mpc, 0, 90)

    idx_last_mpc = np.searchsorted(mpc_arr, date_arr, side='right') - 1
    idx_last_mpc_valid = np.clip(idx_last_mpc, 0, len(mpc_arr) - 1)
    days_since_mpc = (date_arr - mpc_arr[idx_last_mpc_valid]) / np.timedelta64(1, 'D')
    days_since_mpc[idx_last_mpc < 0] = 90
    df['days_since_mpc'] = np.clip(days_since_mpc, 0, 90)

    # 2. Budget Season features
    budget_arr = pd.to_datetime(BUDGET_DATES).sort_values().values
    df['budget_date_flag'] = df['date'].isin(pd.to_datetime(BUDGET_DATES)).astype(int)

    idx_next_b = np.searchsorted(budget_arr, date_arr, side='left')
    idx_next_b_valid = np.clip(idx_next_b, 0, len(budget_arr) - 1)
    days_to_budget = (budget_arr[idx_next_b_valid] - date_arr) / np.timedelta64(1, 'D')
    days_to_budget[idx_next_b >= len(budget_arr)] = 180
    df['days_to_budget'] = np.clip(days_to_budget, 0, 180)
    df['budget_season_flag'] = (df['days_to_budget'] <= 30).astype(int)

    # 3. Ramadan Window flag
    ramadan_flag = np.zeros(len(df), dtype=int)
    for start_str, end_str in RAMADAN_WINDOWS:
        st = pd.Timestamp(start_str)
        en = pd.Timestamp(end_str)
        mask = (dates_dt >= st) & (dates_dt <= en)
        ramadan_flag[mask.values] = 1
    df['ramadan_flag'] = ramadan_flag

    # 4. Earnings Season flag
    earnings_flag = np.zeros(len(df), dtype=int)
    for start_str, end_str in EARNINGS_SEASONS:
        st = pd.Timestamp(start_str)
        en = pd.Timestamp(end_str)
        mask = (dates_dt >= st) & (dates_dt <= en)
        earnings_flag[mask.values] = 1
    df['earnings_season_flag'] = earnings_flag

    # 5. IMF Review Window flag
    df['imf_review_flag'] = df['date'].dt.month.isin([1, 2, 7, 8]).astype(int)

    return df
