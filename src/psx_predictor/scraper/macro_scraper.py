import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from src.psx_predictor.db.repository import upsert_macro_indicators

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)


def _build_ecb_rate_series(dates: pd.DatetimeIndex) -> pd.Series:
    """
    ECB Main Refinancing Rate step-function from official ECB decisions.
    Source: https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/
    Forward-filled to daily; no look-ahead — only past decision dates included.
    """
    ecb_milestones = {
        "2000-06-09": 4.25, "2001-05-11": 4.50, "2001-08-31": 4.25, "2001-09-18": 3.75,
        "2001-11-09": 3.25, "2002-12-06": 2.75, "2003-03-07": 2.50, "2003-06-06": 2.00,
        "2005-12-06": 2.25, "2006-03-08": 2.50, "2006-06-15": 2.75, "2006-08-09": 3.00,
        "2006-10-11": 3.25, "2006-12-13": 3.50, "2007-03-14": 3.75, "2007-06-13": 4.00,
        "2008-10-09": 3.75, "2008-11-13": 3.25, "2008-12-11": 2.50, "2009-01-15": 2.00,
        "2009-03-05": 1.50, "2009-04-02": 1.25, "2009-05-07": 1.00, "2011-04-13": 1.25,
        "2011-07-13": 1.50, "2011-11-09": 1.25, "2011-12-14": 1.00, "2012-07-11": 0.75,
        "2013-05-08": 0.50, "2013-11-13": 0.25, "2014-06-11": 0.15, "2014-09-10": 0.05,
        "2016-03-16": 0.00, "2022-07-27": 0.50, "2022-09-14": 1.25, "2022-11-02": 2.00,
        "2022-12-21": 2.50, "2023-02-08": 3.00, "2023-03-22": 3.50, "2023-05-10": 3.75,
        "2023-06-21": 4.00, "2023-09-20": 4.50, "2024-06-12": 4.25, "2024-09-18": 3.65,
        "2024-12-11": 3.15, "2025-03-12": 2.65, "2025-06-11": 2.15,
    }
    s = pd.Series(index=dates, dtype=float)
    for d_str, val in ecb_milestones.items():
        ts = pd.Timestamp(d_str)
        if ts in s.index:
            s[ts] = val
        else:
            future_idx = s.index[s.index >= ts]
            if len(future_idx) > 0:
                s[future_idx[0]] = val
    return s.ffill().bfill()


def _build_fed_rate_series(dates: pd.DatetimeIndex) -> pd.Series:
    """
    US Federal Funds Rate (upper bound) from FOMC decisions.
    Source: Federal Reserve (federalreserve.gov).
    Forward-filled to daily; no look-ahead.
    """
    fed_milestones = {
        "2000-05-16": 6.50, "2001-01-04": 6.00, "2001-01-31": 5.50, "2001-03-20": 5.00,
        "2001-04-18": 4.50, "2001-05-15": 4.00, "2001-06-27": 3.75, "2001-08-21": 3.50,
        "2001-09-17": 3.00, "2001-10-02": 2.50, "2001-11-06": 2.00, "2001-12-11": 1.75,
        "2002-11-06": 1.25, "2003-06-25": 1.00, "2004-06-30": 1.25, "2004-08-10": 1.50,
        "2004-09-21": 1.75, "2004-11-10": 2.00, "2004-12-14": 2.25, "2005-02-02": 2.50,
        "2005-03-22": 2.75, "2005-05-03": 3.00, "2005-06-30": 3.25, "2005-08-09": 3.50,
        "2005-09-20": 3.75, "2005-11-01": 4.00, "2005-12-13": 4.25, "2006-01-31": 4.50,
        "2006-03-28": 4.75, "2006-05-10": 5.00, "2006-06-29": 5.25, "2007-09-18": 4.75,
        "2007-10-31": 4.50, "2007-12-11": 4.25, "2008-01-22": 3.50, "2008-01-30": 3.00,
        "2008-03-18": 2.25, "2008-04-30": 2.00, "2008-10-08": 1.50, "2008-10-29": 1.00,
        "2008-12-16": 0.25, "2015-12-16": 0.50, "2016-12-14": 0.75, "2017-03-15": 1.00,
        "2017-06-14": 1.25, "2017-12-13": 1.50, "2018-03-21": 1.75, "2018-06-13": 2.00,
        "2018-09-26": 2.25, "2018-12-19": 2.50, "2019-07-31": 2.25, "2019-09-18": 2.00,
        "2019-10-30": 1.75, "2020-03-03": 1.25, "2020-03-15": 0.25, "2022-03-16": 0.50,
        "2022-05-04": 1.00, "2022-06-15": 1.75, "2022-07-27": 2.50, "2022-09-21": 3.25,
        "2022-11-02": 4.00, "2022-12-14": 4.50, "2023-02-01": 4.75, "2023-03-22": 5.00,
        "2023-05-03": 5.25, "2023-07-26": 5.50, "2024-09-18": 5.00, "2024-11-07": 4.75,
        "2024-12-18": 4.50, "2025-01-29": 4.25, "2025-03-19": 4.00,
    }
    s = pd.Series(index=dates, dtype=float)
    for d_str, val in fed_milestones.items():
        ts = pd.Timestamp(d_str)
        if ts in s.index:
            s[ts] = val
        else:
            future_idx = s.index[s.index >= ts]
            if len(future_idx) > 0:
                s[future_idx[0]] = val
    return s.ffill().bfill()


class MacroScraper:
    def sync_macro(self) -> bool:
        logger.info("Fetching macro indicators from Yahoo Finance (Batch)...")
        try:
            ticker_map = {
                "PKR=X": "pkr_usd_rate",
                "EURPKR=X": "eur_pkr_rate",
                "GBPPKR=X": "gbp_pkr_rate",
                "CNYUSD=X": "cny_usd_rate",
                "BZ=F": "brent_oil_price",
                "CL=F": "wti_oil_price",
                "GC=F": "gold_price",
                "HG=F": "copper_price",
                "MTF=F": "coal_price",
                "CT=F": "cotton_price",
                "NG=F": "gas_price",
                "ALI=F": "aluminum_price",
                "ZW=F": "wheat_price",
                "ZS=F": "soybean_price",
                "SLX": "steel_price",
                "VALE": "iron_ore_price",
                "WILMAR.SI": "palm_oil_price",
                "MOS": "urea_price",
                "BOIL": "lng_price",
                "^GSPC": "sp500_close",
                "^IXIC": "nasdaq_close",
                "^DJI": "dow_jones_close",
                "^N225": "nikkei_close",
                "^HSI": "hang_seng_close",
                "000001.SS": "shanghai_close",
                "^FTSE": "ftse_close",
                "^GDAXI": "dax_close",
                "^VIX": "vix_close",
                "EEM": "msci_em_close",
                "FM": "msci_fm_close",
                "DX-Y.NYB": "dxy_close",
                "^TNX": "us10y_yield",
                "^IRX": "us2y_yield",
                "^FVX": "us5y_yield",
                "SCHP": "tips_etf_price",
            }

            tickers = list(ticker_map.keys())
            raw = yf.download(tickers, period="10y", progress=False)

            if raw.empty:
                logger.warning("Batch yfinance download returned empty frame.")
                return False

            dfs = []
            for ticker, col_name in ticker_map.items():
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if 'Close' in raw and ticker in raw['Close']:
                            sub = raw['Close'][ticker].dropna().reset_index()
                        elif ticker in raw and 'Close' in raw[ticker]:
                            sub = raw[ticker]['Close'].dropna().reset_index()
                        else:
                            continue
                    else:
                        sub = raw[['Close']].dropna().reset_index()
                    date_col = 'Date' if 'Date' in sub.columns else sub.columns[0]
                    sub.columns = ['date', col_name]
                    sub['date'] = pd.to_datetime(sub['date']).dt.date
                    dfs.append(sub)
                except Exception as ex:
                    logger.debug(f"Could not extract {ticker}: {ex}")

            if not dfs:
                logger.warning("No series extracted from yfinance download.")
                return False

            macro_df = dfs[0]
            for df in dfs[1:]:
                macro_df = pd.merge(macro_df, df, on='date', how='outer')
            macro_df = macro_df.sort_values('date').reset_index(drop=True)

            # CNY/PKR derived cross rate: CNY/USD * USD/PKR
            if 'cny_usd_rate' in macro_df.columns and 'pkr_usd_rate' in macro_df.columns:
                macro_df['cny_pkr_rate'] = macro_df['cny_usd_rate'] * macro_df['pkr_usd_rate']

            # --- ECB & Fed Rates: step-function from official decision history ---
            all_dates = pd.to_datetime(macro_df['date'])
            macro_df['ecb_rate'] = _build_ecb_rate_series(all_dates).values
            macro_df['fed_funds_rate'] = _build_fed_rate_series(all_dates).values

            # --- Computed Spreads ---
            if 'us10y_yield' in macro_df.columns and 'us2y_yield' in macro_df.columns:
                macro_df['us_yield_curve_2y10y'] = macro_df['us10y_yield'] - macro_df['us2y_yield']
            if 'fed_funds_rate' in macro_df.columns and 'ecb_rate' in macro_df.columns:
                macro_df['fed_ecb_policy_spread'] = macro_df['fed_funds_rate'] - macro_df['ecb_rate']
            if 'dxy_close' in macro_df.columns:
                macro_df['dxy_volatility_20d'] = (
                    macro_df['dxy_close'].pct_change().rolling(20, min_periods=5).std().fillna(0.0)
                )
            if 'pkr_usd_rate' in macro_df.columns:
                macro_df['pkr_usd_volatility_20d'] = (
                    macro_df['pkr_usd_rate'].pct_change().rolling(20, min_periods=5).std().fillna(0.0)
                )

            # --- Global Oil Supply Shock Flag (PDF Group #12) ---
            if 'wti_oil_price' in macro_df.columns:
                wti_chg = macro_df['wti_oil_price'].pct_change(5)
                rolling_std = wti_chg.rolling(252, min_periods=60).std()
                macro_df['global_oil_supply_shock_flag'] = (
                    wti_chg.abs() > 2.5 * rolling_std
                ).astype(int).fillna(0)

            # --- Red Sea / Shipping Disruption Flag ---
            macro_df['date_dt'] = pd.to_datetime(macro_df['date'])
            macro_df['red_sea_disruption_flag'] = (
                macro_df['date_dt'] >= pd.Timestamp('2023-12-19')
            ).astype(int)
            macro_df.drop(columns=['date_dt'], inplace=True)

            success = upsert_macro_indicators(macro_df)
            logger.info(
                f"Macro batch sync complete: {len(macro_df)} rows, {len(macro_df.columns)} columns upserted."
            )
            return success

        except Exception as e:
            logger.error(f"Error fetching macro indicators: {e}")
            return False
