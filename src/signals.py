from __future__ import annotations
import pandas as pd
import numpy as np

def compute_flow_factor(flows: pd.DataFrame, z_window: int = 60) -> pd.DataFrame:
    """
    Input: flows indexed by (date, etf) with column 'flow_pct'.
    Output: same index with columns: flow_pct, flow_z.
    """
    df = flows.copy()
    df["flow_z"] = (df.groupby(level=1)["flow_pct"]
                      .transform(lambda s: (s - s.rolling(z_window, min_periods=10).mean())
                                           / (s.rolling(z_window, min_periods=10).std(ddof=1) + 1e-9)))
    return df

def propagate_flow_to_constituents(flow_df: pd.DataFrame,
                                   holdings: pd.DataFrame,
                                   min_weight: float = 0.002) -> pd.DataFrame:
    """
    Join flow_z (per ETF) onto constituents via holdings weights to create per constituent 'pressure'.
    pressure = weight * flow_z
    """
    # Align: indices (date, etf) join (date, etf, constituent)
    h = holdings.copy()
    f = flow_df[["flow_z"]].copy()
    h = h.join(f, how="left")
    h = h[h["weight"] >= min_weight].copy()
    h["pressure"] = h["weight"] * h["flow_z"]
    # Return (date, etf, constituent): columns weight, flow_z, pressure
    return h

def select_constituents(pressure_df: pd.DataFrame,
                        top_k: int = 15,
                        side: str = "both") -> pd.DataFrame:
    """
    For each (date, etf), rank constituents by 'pressure'.
    Returns a DataFrame with column 'signal' in {+1, -1, 0}.
    """
    df = pressure_df.copy()
    df["rank_desc"] = df.groupby(level=[0,1])["pressure"].rank(ascending=False, method="first")
    df["rank_asc"]  = df.groupby(level=[0,1])["pressure"].rank(ascending=True, method="first")

    df["signal"] = 0
    if side in ("both", "long_only"):
        df.loc[df["rank_desc"] <= top_k, "signal"] = 1
    if side in ("both", "short_only"):
        df.loc[df["rank_asc"]  <= top_k, "signal"] = -1

    return df[["weight", "flow_z", "pressure", "signal"]]
