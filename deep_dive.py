from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- config ----------
FIG_DIR   = os.path.join("results", "deep_dive")
VOL_WINDOW = 21      # days for realized-vol proxy
VOL_PCTL  = 0.60     # split at 60th percentile of realized vol
TREND_WIN = 200      # SPY long-term trend window

# ---------- tiny local plotting helpers ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

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

# ---------- core loaders ----------
def load_core():
    daily  = pd.read_csv("results/daily_pnl.csv", parse_dates=["date"])
    trades = pd.read_csv("results/trades.csv", parse_dates=["date"])

    sig_path = os.path.join("results", "signals_constituent.csv")
    signals_const = pd.read_csv(sig_path, parse_dates=["date"]) if os.path.exists(sig_path) else None

    pe   = pd.read_csv("data/prices_equity.csv", parse_dates=["date"])
    petf = pd.read_csv("data/prices_etf.csv",    parse_dates=["date"])
    return daily, trades, signals_const, pe, petf

def kpis_from_daily(daily: pd.DataFrame) -> dict:
    r = daily["pnl"].fillna(0.0)
    if len(r) == 0:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    ann_ret = (1 + r).prod() ** (252 / max(1, len(r))) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / (ann_vol + 1e-12)
    curve = (1 + r).cumprod()
    mdd = ((curve / curve.cummax()) - 1).min()
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(sharpe), "max_drawdown": float(mdd)}

# ---------- analyses ----------
def rolling_parent_exposure(trades: pd.DataFrame):
    """
    Approximate net parent ETF exposure by summing target weights of hedge tickers (SPY/QQQ/IWM).
    Writes CSV + PNG.
    """
    df = trades.copy()
    name_col = "ticker" if "ticker" in df.columns else ("instrument" if "instrument" in df.columns else None)
    if not name_col:
        return None
    parents = {"SPY", "QQQ", "IWM"}
    df["date"] = pd.to_datetime(df["date"])
    df["is_parent"] = df[name_col].isin(parents)
    hed = (df[df["is_parent"]]
             .groupby("date")["target_w"]
             .sum()
             .rename("net_parent_w")
             .reset_index())
    plot_series(hed, "date", "net_parent_w",
                "Net Parent Exposure (sum target_w of SPY/QQQ/IWM)",
                os.path.join(FIG_DIR, "net_parent_exposure.png"))
    hed.to_csv(os.path.join(FIG_DIR, "net_parent_exposure.csv"), index=False)
    return hed

def decile_buckets(signals_const: pd.DataFrame, pe: pd.DataFrame):
    """
    Monotonicity: decile of prior-day signal vs next-day open->close returns.
    Writes CSV + PNG + TXT.
    """
    if signals_const is None or signals_const.empty:
        return None

    sc = signals_const.copy()
    sig_col = next((c for c in ["signal", "pressure", "flow_z"] if c in sc.columns), None)
    if sig_col is None:
        return None

    tick_col = "ticker" if "ticker" in sc.columns else ("constituent" if "constituent" in sc.columns else None)
    if not tick_col:
        return None

    pe2 = pe.sort_values(["ticker","date"]).copy()
    pe2["ret_oc"] = (pe2["close"] / pe2["open"]) - 1.0
    sc["date"]  = pd.to_datetime(sc["date"])
    pe2["date"] = pd.to_datetime(pe2["date"])

    sc = sc.sort_values([tick_col,"date"])
    sc["signal_lag"] = sc.groupby(tick_col)[sig_col].shift(1)

    df = sc.rename(columns={tick_col:"ticker"})[["date","ticker","signal_lag"]].merge(
        pe2[["date","ticker","ret_oc"]], on=["date","ticker"], how="inner"
    ).dropna(subset=["signal_lag","ret_oc"])

    # daily cross-section deciles
    def decile_groups(x):
        try:
            q = pd.qcut(x["signal_lag"], 10, labels=False, duplicates="drop")
        except Exception:
            q = pd.Series([np.nan]*len(x), index=x.index)
        return x.assign(decile=q)

    df = df.groupby("date", group_keys=False).apply(decile_groups).dropna(subset=["decile"])
    by_dec = df.groupby("decile")["ret_oc"].mean()
    by_dec.to_csv(os.path.join(FIG_DIR, "decile_avg_nextday_return.csv"))
    plot_bar(by_dec, "Avg Next-Day OC Return by Signal Decile (0=lowest, 9=highest)",
             os.path.join(FIG_DIR, "decile_monotonicity.png"))

    spread = by_dec.get(9, np.nan) - by_dec.get(0, np.nan)
    with open(os.path.join(FIG_DIR, "decile_monotonicity.txt"), "w") as f:
        f.write(f"Top-bottom decile next-day OC return spread: {spread:.6f}\n")
    return by_dec

def daily_ic(signals_const: pd.DataFrame, pe: pd.DataFrame):
    """
    Daily cross-sectional IC: corr(signal_{t-1}, next-day OC return).
    Writes CSV + PNG.
    """
    if signals_const is None or signals_const.empty:
        return None

    sc = signals_const.copy()
    sig_col = next((c for c in ["signal", "pressure", "flow_z"] if c in sc.columns), None)
    if sig_col is None:
        return None

    tick_col = "ticker" if "ticker" in sc.columns else ("constituent" if "constituent" in sc.columns else None)
    if not tick_col:
        return None

    pe2 = pe.sort_values(["ticker","date"]).copy()
    pe2["ret_oc"] = (pe2["close"] / pe2["open"]) - 1.0
    sc["date"]  = pd.to_datetime(sc["date"])
    pe2["date"] = pd.to_datetime(pe2["date"])

    sc = sc.sort_values([tick_col,"date"])
    sc["signal_lag"] = sc.groupby(tick_col)[sig_col].shift(1)

    df = sc.rename(columns={tick_col:"ticker"})[["date","ticker","signal_lag"]].merge(
        pe2[["date","ticker","ret_oc"]], on=["date","ticker"], how="inner"
    ).dropna(subset=["signal_lag","ret_oc"])

    def day_corr(x):
        if len(x) < 3: return np.nan
        return x["signal_lag"].corr(x["ret_oc"])

    ic = df.groupby("date").apply(day_corr).rename("ic").reset_index()
    ic.to_csv(os.path.join(FIG_DIR, "daily_ic.csv"), index=False)
    plot_series(ic, "date", "ic",
                "Daily Cross-Sectional IC (signal t-1 vs next-day OC)",
                os.path.join(FIG_DIR, "daily_ic.png"))
    return ic

def realized_vol_from_spy(petf: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Build a VIX-like proxy from SPY realized volatility (rolling window).
    Also compute a 200d trend flag for SPY.
    """
    spy = petf[petf["ticker"]=="SPY"].sort_values("date").copy()
    spy["ret"] = (spy["close"] / spy["close"].shift(1)) - 1.0
    spy["rv"]  = spy["ret"].rolling(window, min_periods=max(5, window//2)).std() * np.sqrt(252) * 100.0
    spy["trend200"] = (spy["close"] / spy["close"].rolling(TREND_WIN, min_periods=TREND_WIN//4).mean()) - 1.0
    return spy[["date","rv","trend200"]]

def regime_splits(daily: pd.DataFrame, petf: pd.DataFrame):
    """
    Split KPIs by:
      - SPY 200d trend (up/down) using local prices
      - Realized vol proxy percentile (low/high) from SPY returns
    Writes CSV + two PNGs (Sharpe by regime).
    """
    flags = realized_vol_from_spy(petf, VOL_WINDOW)
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(flags, on="date", how="left")

    # Trend regime
    df["trend_regime"] = np.where(df["trend200"] > 0, "SPY_uptrend", "SPY_downtrend")

    # Vol regime via percentile split
    thr = np.nanpercentile(df["rv"], VOL_PCTL * 100)
    df["vol_regime"] = np.where(df["rv"] > thr, f"VOL>p{int(VOL_PCTL*100)}", f"VOL<=p{int(VOL_PCTL*100)}")

    # KPIs by regime
    out_rows = []
    for col, name in [("trend_regime","Trend"), ("vol_regime","Volatility")]:
        for reg, sub in df.groupby(col):
            k = kpis_from_daily(sub)
            out_rows.append({"split": name, "regime": reg, **k})
    out = pd.DataFrame(out_rows).sort_values(["split","regime"])
    out.to_csv(os.path.join(FIG_DIR, "regime_kpis.csv"), index=False)

    # Bar charts of Sharpe by regime
    for split_name, sub in out.groupby("split"):
        ser = sub.set_index("regime")["sharpe"]
        fn = f"sharpe_by_{split_name.lower()}.png"
        plot_bar(ser, f"Sharpe by Regime: {split_name}", os.path.join(FIG_DIR, fn))
    return out

def main():
    ensure_dir(FIG_DIR)
    daily, trades, signals_const, pe, petf = load_core()

    # 1) Hedge quality / exposure
    rolling_parent_exposure(trades)

    # 2) Decile monotonicity of signal
    decile_buckets(signals_const, pe)

    # 3) Daily IC time series
    daily_ic(signals_const, pe)

    # 4) Regime splits (purely from local SPY prices)
    regime_splits(daily, petf)

    # 5) Write a short deep dive report
    md = []
    md.append("# Deep Dive")
    md.append("")
    md.append("## Figures")
    figs = [
        "net_parent_exposure.png",
        "decile_monotonicity.png",
        "daily_ic.png",
        "sharpe_by_trend.png",
        "sharpe_by_volatility.png",
    ]
    for fn in figs:
        fp = os.path.join(FIG_DIR, fn)
        if os.path.exists(fp):
            md.append(f"![{fn}]({fp})")
    md.append("")
    md.append("## Tables")
    for csv in ["net_parent_exposure.csv","decile_avg_nextday_return.csv","daily_ic.csv","regime_kpis.csv"]:
        if os.path.exists(os.path.join(FIG_DIR, csv)):
            md.append(f"- {os.path.join('results','deep_dive', csv)}")
    with open(os.path.join("results", "DEEP_DIVE.md"), "w") as f:
        f.write("\n".join(md))

    print("Wrote deep dive figs to:", FIG_DIR)
    print("Wrote deep dive report:", os.path.join("results","DEEP_DIVE.md"))

if __name__ == "__main__":
    main()
