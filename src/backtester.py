from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal, Tuple, Dict

def _to_ret(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    df_prices: MultiIndex (date, ticker) with 'open' and 'close'.
    Returns daily returns for open->close and close->next_open (for different trade_at choices).
    """
    px = df_prices[["open", "close"]].copy()
    px = px.sort_index()
    # Returns if we enter at next_open and exit at same-day close:
    # ret = (close_t / open_t) - 1 for each date
    px["ret_open_to_close"] = (px["close"] / px["open"]) - 1.0

    # For close-to-next-open (if you want overnight exposure)
    px_by_ticker = px.groupby(level=1)
    next_open = px_by_ticker["open"].shift(-1)
    px["ret_close_to_next_open"] = (next_open / px["close"]) - 1.0
    return px

def _rolling_vol(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ret = prices.groupby(level=1)["close"].pct_change()
    vol = ret.groupby(level=1).rolling(window).std().reset_index(level=0, drop=True)
    vol = vol.to_frame("vol")
    return vol

def backtest_flow_strategy(
    equity_prices: pd.DataFrame,     # (date,ticker) open/close/...
    etf_prices: pd.DataFrame,        # (date,ticker) for hedging ETF
    target_weights: pd.DataFrame,    # (date, etf, ticker) -> target_w
    costs_bps_equity: float = 3.0,
    costs_bps_etf: float = 1.0,
    trade_at: Literal["next_open","close"] = "next_open",
    slippage_bps: float = 1.0,
    hedge_with_parent: bool = True,
    hedge_method: Literal["notional","regression"] = "notional"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple daily backtester:
    - Enters positions at 'next_open' (default) and exits at 'same-day close' (hold_period=1) by shifting signals.
    - Applies transaction costs at entry and exit (bps).
    - Hedges with parent ETF by offsetting the net notional of constituents within each parent.
    Returns:
      trades_df: per (date,ticker) position and PnL
      daily_pnl: per date aggregated PnL
    """
    eq = equity_prices.copy()
    etf = etf_prices.copy()
    tw = target_weights.copy()

    # Step 1: choose return leg
    rets = _to_ret(eq)
    rets_etf = _to_ret(etf)

    # Step 2: align target weights to execution day (enter at next open)
    if trade_at == "next_open":
        tw_exec = tw.groupby(level=[1,2]).shift(1)  # shift by one day per (etf,ticker)
    else:
        tw_exec = tw.copy()

    # Step 3: map weights onto tickers and compute PnL
    # Join returns (open->close)
    eq_ret = rets["ret_open_to_close"].to_frame("ret").copy()
    eq_ret.index.names = ["date","ticker"]

    tw_exec = tw_exec.reset_index().set_index(["date","ticker"])
    tw_exec = tw_exec.join(eq_ret, how="left")
    tw_exec["ret"] = tw_exec["ret"].fillna(0.0)

    # Transaction costs (entry+exit)
    # If we assume in/out daily, apply 2 legs of costs.
    round_trip_bps = (costs_bps_equity + slippage_bps) * 2.0
    tw_exec["pnl"] = tw_exec["target_w"] * tw_exec["ret"] - (abs(tw_exec["target_w"]) * (round_trip_bps / 1e4))

    # Step 4: Hedge with parent ETF
    if hedge_with_parent:
        # For each (date, etf) compute net notional from constituents
        net_parent_w = tw_exec.groupby(["date","etf"])["target_w"].sum().rename("net_child_w")
        # Map parent ETF returns
        etf_ret = rets_etf["ret_open_to_close"].rename("ret")
        # Join parent returns for hedging
        net_parent = net_parent_w.reset_index().set_index(["date","etf"])
        net_parent = net_parent.join(etf_ret.reset_index().set_index(["date","ticker"]),
                                     how="left", rsuffix="_etf")
        net_parent["ret"] = net_parent["ret"].fillna(0.0)

        # Hedge PnL: short (or long) parent ETF by net child weight (notional hedge)
        parent_cost_bps = (costs_bps_etf + slippage_bps) * 2.0
        net_parent["hedge_pnl"] = (-net_parent["net_child_w"]) * net_parent["ret"] \
                                  - (abs(net_parent["net_child_w"]) * (parent_cost_bps / 1e4))
        # Aggregate per (date,etf)
        hedge_daily = net_parent["hedge_pnl"].groupby(level=0).sum()
    else:
        hedge_daily = pd.Series(0.0, index=tw_exec.index.get_level_values(0).unique())

    # Step 5: aggregate daily PnL
    pnl_daily_const = tw_exec.groupby(level=0)["pnl"].sum()
    daily_pnl = (pnl_daily_const + hedge_daily).to_frame("pnl")
    daily_pnl["equity_curve"] = (1.0 + daily_pnl["pnl"]).cumprod()

    # Trades detail (optional granularity)
    trades = tw_exec[["target_w","ret","pnl"]].copy()
    trades.reset_index(inplace=True)
    trades.rename(columns={"ticker":"constituent"}, inplace=True)

    return trades, daily_pnl
