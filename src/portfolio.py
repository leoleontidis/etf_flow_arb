from __future__ import annotations
import pandas as pd
import numpy as np

def risk_parity_weights(vol: pd.Series) -> pd.Series:
    """
    Invert vol to allocate more to stable names. Normalize to sum(|w|)=1.
    """
    v = vol.replace(0, np.nan)
    inv = 1.0 / v
    inv = inv.fillna(0.0)
    if inv.abs().sum() == 0:
        return inv
    w = inv / inv.abs().sum()
    return w

def compute_position_sizes(signals_df: pd.DataFrame,
                           equity_vol: pd.DataFrame,
                           gross_leverage: float = 1.0,
                           use_risk_parity: bool = True) -> pd.DataFrame:
    """
    signals_df index: (date, etf, constituent), column 'signal' in {-1,0,1}
    equity_vol index: (date, ticker), daily volatility proxy (e.g., rolling std of returns).
    Returns 'target_w' per (date, etf, constituent).
    """
    df = signals_df.copy()
    df = df.reset_index().rename(columns={"constituent":"ticker"})
    # join vol
    vol = equity_vol.rename(columns={"vol":"_vol"}).reset_index()
    df = df.merge(vol, on=["date", "ticker"], how="left")
    df["_vol"] = df["_vol"].fillna(df["_vol"].median())

    def _alloc(group):
        # group has rows per (etf,ticker) at a given date
        s = group["signal"]
        if s.abs().sum() == 0:
            group["target_w"] = 0.0
            return group
        if use_risk_parity:
            rp = risk_parity_weights(group["_vol"])
            w = s * rp
        else:
            # equal weight long and short
            pos = s.replace(0, np.nan)
            w = pos / pos.abs().sum()
            w = w.fillna(0.0)
        # scale to gross leverage
        w = w * gross_leverage
        group["target_w"] = w
        return group

    out = df.groupby(["date", "etf"], group_keys=False).apply(_alloc)
    out.set_index(["date", "etf", "ticker"], inplace=True)
    return out[["target_w"]]
