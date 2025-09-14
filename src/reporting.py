from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _savefig(path: str, tight: bool=True):
    if tight: plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def plot_equity_curve(daily: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    daily["equity_curve"].plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date"); ax.set_ylabel("Equity (gross)")
    _savefig(os.path.join(outdir, "equity_curve.png"))

def plot_drawdown(daily: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    ret = daily["pnl"].fillna(0.0)
    curve = (1.0 + ret).cumprod()
    peak = curve.cummax()
    dd = (curve/peak) - 1.0
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    dd.plot(ax=ax)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    _savefig(os.path.join(outdir, "drawdown.png"))

def plot_rolling_sharpe(daily: pd.DataFrame, outdir: str, window: int=63):
    ensure_dir(outdir)
    r = daily["pnl"].fillna(0.0)
    roll_mu = r.rolling(window).mean() * 252
    roll_sd = r.rolling(window).std() * np.sqrt(252)
    roll_sh = roll_mu / (roll_sd + 1e-12)
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    roll_sh.plot(ax=ax)
    ax.set_title(f"Rolling Sharpe ({window}d window)")
    ax.set_xlabel("Date"); ax.set_ylabel("Sharpe")
    _savefig(os.path.join(outdir, "rolling_sharpe.png"))

def plot_daily_pnl_hist(daily: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    r = daily["pnl"].dropna().values
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.hist(r, bins=50)
    ax.set_title("Daily PnL Distribution")
    ax.set_xlabel("Daily Return"); ax.set_ylabel("Frequency")
    _savefig(os.path.join(outdir, "pnl_hist.png"))

def plot_signal_distributions(signals_const: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    if "pressure" not in signals_const.columns or "flow_z" not in signals_const.columns:
        return
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.hist(signals_const["flow_z"].dropna().values, bins=60)
    ax.set_title("Flow Z-Score Distribution")
    ax.set_xlabel("flow_z"); ax.set_ylabel("count")
    _savefig(os.path.join(outdir, "flow_z_hist.png"))

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.hist(signals_const["pressure"].dropna().values, bins=60)
    ax.set_title("Constituent Pressure Distribution")
    ax.set_xlabel("pressure"); ax.set_ylabel("count")
    _savefig(os.path.join(outdir, "pressure_hist.png"))

def plot_turnover(trades: pd.DataFrame, outdir: str):
    """
    Approx turnover: sum of abs target_w change per date across all names.
    Requires 'date','constituent','target_w'.
    """
    ensure_dir(outdir)
    if "target_w" not in trades.columns:
        return
    t = trades[["date","constituent","target_w"]].copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values(["constituent","date"])
    t["w_lag"] = t.groupby("constituent")["target_w"].shift(1)
    t["chg"] = (t["target_w"] - t["w_lag"]).abs()
    daily = t.groupby("date")["chg"].sum()
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    daily.plot(ax=ax)
    ax.set_title("Approx Portfolio Turnover (|Δw| per day)")
    ax.set_xlabel("Date"); ax.set_ylabel("Sum |Δw|")
    _savefig(os.path.join(outdir, "turnover.png"))

def top_contributors(trades: pd.DataFrame, outdir: str, k: int=20):
    """
    Aggregate PnL by constituent and save a bar plot.
    Requires 'constituent','pnl'
    """
    ensure_dir(outdir)
    if "constituent" not in trades.columns or "pnl" not in trades.columns:
        return
    agg = trades.groupby("constituent")["pnl"].sum().sort_values(ascending=False)
    top_pos = agg.head(k)
    top_neg = agg.tail(k)
    top = pd.concat([top_pos, top_neg])

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    top.plot(kind="bar", ax=ax)
    ax.set_title(f"Top +/- Contributors (total PnL), k={k}")
    ax.set_ylabel("Total PnL")
    _savefig(os.path.join(outdir, "top_contributors.png"))

def signal_forward_corr(signals_const: pd.DataFrame, prices_equity: pd.DataFrame, outdir: str):
    """
    Diagnostic: correlation between today's signal and next-day open->close return.
    Returns CSV and plot.
    """
    ensure_dir(outdir)
    if not {"date","ticker","signal"}.issubset(set(signals_const.columns)):
        # adapt name if 'constituent' used:
        if "constituent" in signals_const.columns:
            sc = signals_const.rename(columns={"constituent": "ticker"})
        else:
            return
    else:
        sc = signals_const.copy()

    # compute next-day open->close ret from equity prices
    pe = prices_equity.copy()
    pe = pe.sort_values(["ticker","date"])
    pe["ret_oc"] = (pe["close"]/pe["open"]) - 1.0
    ret = pe[["date","ticker","ret_oc"]]

    sc["date"] = pd.to_datetime(sc["date"])
    ret["date"] = pd.to_datetime(ret["date"])
    # Use prior-day signal for next-day ret_oc (align)
    sc = sc.sort_values(["ticker","date"])
    sc["signal_lag"] = sc.groupby("ticker")["signal"].shift(1)

    df = sc.merge(ret, on=["date","ticker"], how="inner").dropna(subset=["signal_lag","ret_oc"])
    corr = df["signal_lag"].corr(df["ret_oc"])

    df.to_csv(os.path.join(outdir, "signal_vs_nextday_ret.csv"), index=False)

    # small scatter plot
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.scatter(df["signal_lag"], df["ret_oc"], s=6, alpha=0.5)
    ax.set_title(f"Signal (t-1) vs Next-Day Return (corr={corr:.3f})")
    ax.set_xlabel("signal (t-1)")
    ax.set_ylabel("open->close return (t)")
    _savefig(os.path.join(outdir, "signal_forward_corr.png"))

    # Save a text summary
    with open(os.path.join(outdir, "signal_forward_corr.txt"), "w") as f:
        f.write(f"Pearson corr (signal t-1 vs next-day OC return): {corr:.6f}\n")

def plot_series(df: pd.DataFrame, x: str, y: str, title: str, outpath: str):
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    df.plot(x=x, y=y, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xlabel(x); ax.set_ylabel(y)
    _savefig(outpath)

def plot_bar(series: pd.Series, title: str, outpath: str):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    _savefig(outpath)
