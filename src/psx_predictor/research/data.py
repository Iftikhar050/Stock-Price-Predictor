"""Price panel for research: loads EOD data (read-only), removes holiday rows and
impossible price prints, and builds the tradability / liquidity matrices the
backtest needs. All matrices are wide: index = trading dates, columns = tickers."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text


@dataclass
class Panel:
    close: pd.DataFrame        # clean price index (rebuilt from clean returns)
    ret: pd.DataFrame          # clean daily simple returns (artefacts set to 0)
    raw_ret: pd.DataFrame      # returns as stored, for reference
    volume: pd.DataFrame
    adv: pd.DataFrame          # 20-day median traded value (PKR, adjusted prices)
    bad: pd.DataFrame          # True where the print was an artefact
    can_buy: pd.DataFrame      # traded, not an artefact, not locked at upper circuit
    can_sell: pd.DataFrame     # traded, not an artefact, not locked at lower circuit
    listed: pd.DataFrame       # True from a ticker's first price onwards
    sector: pd.Series
    active: list               # the app's current universe (stock_metadata.is_active)
    holidays: list             # dates dropped as non-trading days


def load_eod(engine, start, end):
    q = text("SELECT ticker, date, close, volume FROM stock_eod_data "
             "WHERE date >= :s AND date <= :e")
    df = pd.read_sql(q, engine, params={"s": start, "e": end})
    df["date"] = pd.to_datetime(df["date"])
    meta = pd.read_sql(text("SELECT ticker, sector, is_active FROM stock_metadata"), engine)
    return df, meta


def circuit_band(dates, cfg):
    change = pd.Timestamp(cfg["circuit_change_date"])
    return pd.Series(np.where(dates < change, cfg["circuit_band_pre_2020"],
                              cfg["circuit_band_from_2020"]), index=dates)


def build_panel(df, meta, cfg, calendar=None):
    """calendar: the exchange's real trading days (KSE-100 dates). Rows on other
    days are holidays stored by the price source and are dropped; real trading
    days with zero volume are kept but untradeable."""
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = df.pivot(index="date", columns="ticker", values="volume").reindex_like(close)
    active = sorted(meta.loc[meta["is_active"].astype(bool), "ticker"])
    active = [t for t in active if t in close.columns]

    # 1. Holiday rows.
    if calendar is not None:
        holidays = close.index[~close.index.isin(calendar)]
    else:   # fallback: most of the liquid universe shows zero volume
        act_vol = volume[active].fillna(0)
        zero_share = (act_vol == 0).sum(axis=1) / close[active].notna().sum(axis=1).clip(lower=1)
        holidays = close.index[zero_share > cfg["data_cleaning"]["holiday_zero_volume_share"]]
    close, volume = close.drop(holidays), volume.drop(holidays)

    # 2. Impossible prints: a move no circuit breaker allows (spikes that revert,
    #    unadjusted splits). The day's return is neutralised and the name is
    #    untradeable that day, so the rebuilt series stays continuous.
    filled = close.ffill()
    raw_ret = filled.pct_change(fill_method=None)
    prev = filled.shift(1)
    # The Rs 1 leg lets genuine penny stocks move more than the % band, but on
    # dividend-adjusted prices it is far too lenient, so it is capped.
    dc = cfg["data_cleaning"]
    limit = np.maximum(dc["impossible_move"], np.minimum(1.05 / prev, dc["penny_cap"]))
    bad = (raw_ret.abs() > limit) & raw_ret.notna()
    ret = raw_ret.mask(bad, 0.0).fillna(0.0)
    listed = close.notna().cummax()
    ret = ret.where(listed, np.nan)
    clean_close = (1 + ret.fillna(0)).cumprod().where(listed)

    # 3. Tradability: needs a real print with volume, and not locked at the limit.
    traded = (volume.fillna(0) > 0) & close.notna() & ~bad
    band = circuit_band(close.index, cfg["execution"])
    # limit is the band or Rs 1, whichever is larger (adjusted prices make the
    # Rs 1 leg slightly lenient for old dates)
    lim = np.maximum(prev.mul(0).add(band, axis=0), 1.0 / prev)
    up_lock = raw_ret >= lim - 0.002
    down_lock = raw_ret <= -lim + 0.002
    can_buy = traded & ~up_lock
    can_sell = traded & ~down_lock

    tv = (filled * volume.fillna(0))
    adv = tv.rolling(20, min_periods=10).median()

    sector = meta.set_index("ticker")["sector"].reindex(close.columns).fillna("Unknown")
    return Panel(clean_close, ret, raw_ret, volume, adv, bad, can_buy, can_sell, listed,
                 sector, active, list(holidays))


def load_kse100(path):
    """KSE-100 daily closes from an Investing.com style export (Date, Price, ...)."""
    d = pd.read_csv(Path(path).expanduser(), thousands=",")
    d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y")
    return d.set_index("Date")["Price"].astype(float).sort_index()


def pit_universe(panel, top_n, min_history):
    """Point-in-time universe: each month, the top_n names by trailing 60-day
    median traded value among everything listed then (incl. later-delisted)."""
    tv60 = (panel.close.ffill() * panel.volume.fillna(0)).rolling(60, min_periods=40).median()
    hist = panel.listed.cumsum() >= min_history
    idx = panel.close.index
    month_ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().values
    ranks = tv60.loc[month_ends].where(hist.loc[month_ends]).rank(axis=1, ascending=False)
    at_month_end = (ranks <= top_n).astype(float)
    member = at_month_end.reindex(idx).ffill().fillna(0.0).astype(bool)
    # membership decided at a month end applies from the next trading day
    return member.shift(1, fill_value=False)
