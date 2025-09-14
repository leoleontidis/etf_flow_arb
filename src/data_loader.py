from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

def _read_csv(path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if parse_dates:
        for c in parse_dates:
            df[c] = pd.to_datetime(df[c])
    return df

def load_prices(prices_path: str) -> pd.DataFrame:
    """
    Returns tidy prices with MultiIndex (date, ticker) and columns: open, high, low, close, volume.
    """
    df = _read_csv(prices_path, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"])
    df.set_index(["date", "ticker"], inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def load_etf_flows(flows_path: str) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date, etf with columns: net_flow_usd, aum_usd, flow_pct.
    """
    df = _read_csv(flows_path, parse_dates=["date"])
    df = df.sort_values(["etf", "date"])
    df["flow_pct"] = df["net_flow_usd"] / df["aum_usd"].replace(0, np.nan)
    df.set_index(["date", "etf"], inplace=True)
    return df

def load_etf_holdings(holdings_path: str, max_per_etf: int = 100) -> pd.DataFrame:
    """
    Returns holdings with MultiIndex (date, etf, constituent) and a 'weight' column.
    Filters to top N per ETF per date.
    """
    df = _read_csv(holdings_path, parse_dates=["date"])
    df = df.sort_values(["date", "etf", "weight"], ascending=[True, True, False])
    if max_per_etf is not None:
        df["rank"] = df.groupby(["date", "etf"])["weight"].rank(method="first", ascending=False)
        df = df[df["rank"] <= max_per_etf].drop(columns="rank")
    df.set_index(["date", "etf", "constituent"], inplace=True)
    df = df[["weight"]].astype(float)
    return df

def align_calendar(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Outer-joins on the date dimension across all inputs, forward-filling where sensible.
    (We do NOT forward-fill prices; we just align indices and let later code handle missing.)
    """
    # Collect all unique dates
    all_dates = sorted(set().union(*[set(df.index.get_level_values(0)) for df in dfs]))
    # Reindex each df on dates; preserve other index levels
    out = []
    for df in dfs:
        levels = df.index.names
        if len(levels) == 2:  # e.g., (date, ticker) or (date, etf)
            a = (pd.MultiIndex.from_product([all_dates, df.index.get_level_values(1).unique()],
                                            names=levels))
            out.append(df.reindex(a))
        elif len(levels) == 3:  # (date, etf, constituent)
            a = (pd.MultiIndex.from_product([all_dates,
                                             df.index.get_level_values(1).unique(),
                                             df.index.get_level_values(2).unique()],
                                            names=levels))
            out.append(df.reindex(a))
        else:
            out.append(df)
    return tuple(out)
