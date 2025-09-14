from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: use the helpers if you added them
try:
    from src.reporting import ensure_dir, plot_series, plot_bar
except Exception:
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

FIG_DIR = os.path.join("results", "deep_dive")
ensure_dir(FIG_DIR)

def load_core():
    daily = pd.read_csv("results/daily_pnl.csv", parse_dates=["date"])
    trades = pd.read_csv("results/trades.csv", parse_dates=["date"])
    # Optional: signals saved by main.py patch; otherwise skip parts that need it
    sig_path = os.path.join("results", "signals_constituent.csv")
    signals_const = pd.read_csv(sig_path, parse_dates=["date"]) if os.path.exists(sig_path) else None
    pe = pd.read_csv("data/prices_equity.csv", parse_dates=["date"])
    petf = pd.read_csv("data/prices_etf.csv", parse_dates=["date"])
    return daily, trades, signals_const, pe, petf

def kpis_from_daily(daily: pd.DataFrame) -> dict:
    r = daily["pnl"].fillna(0.0)
    ann_ret = (1 + r).prod() ** (252 / max(1, len(r))) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    eq = (1 + r).cumprod()
    mdd = ((eq / eq.cummax()) - 1).min()
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(sharpe), "max_drawdown": float(mdd)}

def rolling_parent_exposure(trades: pd.DataFrame):
    """
    Approximate net parent ETF exposure by summing target weights for the hedge instrument(s).
    Assumes trades.csv has columns: date, instrument (or ticker), target_w, pnl...
    If your file uses different column names, adjust below.
    """
    df = trades.copy()
    # Try to detect parent tickers commonly used as hedges
    parents = {"SPY", "QQQ", "IWM"}
    name_col = "ticker" if "ticker" in df.columns else ("instrument" if "instrument" in df.columns else None)
    if not name_col:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df["is_parent"] = df[name_col].isin(parents)
    hed = (df[df["is_parent"]]
             .groupby("date")["target_w"]
             .sum()
             .rename("net_parent_w")
             .reset_index())
    # Plot
    plot_series(hed, "date", "net_parent_w",
                "Net Parent Exposure (sum target_w of SPY/QQQ/IWM)", os.path.join(FIG_DIR, "net_parent_exposure.png"))
    hed.to_csv(os.path.join(FIG_DIR, "net_parent_exposure.csv"), index=False)
    return hed

def decile_buckets(signals_const: pd.DataFrame, pe: pd.DataFrame):
    """
    Build monotonicity check: decile buckets of prior-day signal vs next-day OC returns.
    Requires signals_const with columns:
      - date, constituent (or ticker), and a numeric 'signal' (or 'pressure'/'flow_z' we will pick)
    """
    if signals_const is None or signals_const.empty:
        return None

    sc = signals_const.copy()
    # Try to pick a single signal field
    sig_col = None
    for c in ["signal", "pressure", "flow_z"]:
        if c in sc.columns:
            sig_col = c; break
    if sig_col is None:
        return None

    tick_col = "ticker" if "ticker" in sc.columns else ("constituent" if "constituent" in sc.columns else None)
    if not tick_col:
        return None

    # Next-day OC returns
    pe2 = pe.sort_values(["ticker","date"]).copy()
    pe2["ret_oc"] = (pe2["close"] / pe2["open"]) - 1.0
    sc["date"] = pd.to_datetime(sc["date"])
    pe2["date"] = pd.to_datetime(pe2["date"])

    # Lag signal by 1 day per ticker
    sc = sc.sort_values([tick_col,"date"])
    sc["signal_lag"] = sc.groupby(tick_col)[sig_col].shift(1)

    df = sc.rename(columns={tick_col:"ticker"})[["date","ticker","signal_lag"]].merge(
        pe2[["date","ticker","ret_oc"]], on=["date","ticker"], how="inner"
    ).dropna(subset=["signal_lag","ret_oc"])

    # Deciles each day (cross-section)
    def decile_groups(x):
        try:
            q = pd.qcut(x["signal_lag"], 10, labels=False, duplicates="drop")
        except Exception:
            q = pd.Series([np.nan]*len(x), index=x.index)
        x = x.assign(decile=q)
        return x

    df = df.groupby("date", group_keys=False).apply(decile_groups).dropna(subset=["decile"])
    # Average next-day return per decile
    by_dec = df.groupby("decile")["ret_oc"].mean()
    by_dec.to_csv(os.path.join(FIG_DIR, "decile_avg_nextday_return.csv"))
    plot_bar(by_dec, "Avg Next-Day OC Return by Signal Decile (0=lowest, 9=highest)",
             os.path.join(FIG_DIR, "decile_monotonicity.png"))

    # Spread: top - bottom
    spread = by_dec.get(9, np.nan) - by_dec.get(0, np.nan)
    with open(os.path.join(FIG_DIR, "decile_monotonicity.txt"), "w") as f:
        f.write(f"Top-bottom decile next-day OC return spread: {spread:.6f}\n")
    return by_dec

def daily_ic(signals_const: pd.DataFrame, pe: pd.DataFrame):
    """
    Daily cross-sectional information coefficient (Pearson corr) between prior-day signal
    and next-day OC return.
    """
    if signals_const is None or signals_const.empty:
        return None

    sc = signals_const.copy()
    sig_col = None
    for c in ["signal", "pressure", "flow_z"]:
        if c in sc.columns: sig_col = c; break
    if sig_col is None:
        return None

    tick_col = "ticker" if "ticker" in sc.columns else ("constituent" if "constituent" in sc.columns else None)
    if not tick_col:
        return None

    pe2 = pe.sort_values(["ticker","date"]).copy()
    pe2["ret_oc"] = (pe2["close"] / pe2["open"]) - 1.0
    sc["date"] = pd.to_datetime(sc["date"])
    pe2["date"] = pd.to_datetime(pe2["date"])

    sc = sc.sort_values([tick_col,"date"])
    sc["signal_lag"] = sc.groupby(tick_col)[sig_col].shift(1)
    df = sc.rename(columns={tick_col:"ticker"})[["date","ticker","signal_lag"]].merge(
        pe2[["date","ticker","ret_oc"]], on=["date","ticker"], how="inner"
    ).dropna(subset=["signal_lag","ret_oc"])

    # Cross-sectional correlation per day
    def day_corr(x):
        if len(x) < 3: return np.nan
        return x["signal_lag"].corr(x["ret_oc"])

    ic = df.groupby("date").apply(day_corr).rename("ic").reset_index()
    ic.to_csv(os.path.join(FIG_DIR, "daily_ic.csv"), index=False)
    plot_series(ic, "date", "ic", "Daily Cross-Sectional IC (signal t-1 vs next-day OC)", os.path.join(FIG_DIR, "daily_ic.png"))
    return ic

def get_vix_series():
    """
    Try multiple sources for a VIX-like series, returning a DataFrame(date, vix).
    Order:
      1) yfinance ^VIX
      2) yfinance VIXY  (short-term VIX futures ETF, scaled)
      3) yfinance VXX   (ETN proxy, scaled)
      4) None (caller will fall back to realized vol proxy)
    """
    import warnings
    warnings.filterwarnings("ignore")

    def _yf_hist(tkr):
        import yfinance as yf
        try:
            d = yf.Ticker(tkr).history(period="max", interval="1d").reset_index()
            d = d.rename(columns=str.lower)
            if "date" not in d.columns and "index" in d.columns:
                d = d.rename(columns={"index": "date"})
            if "close" in d.columns and len(d):
                return d[["date", "close"]].rename(columns={"close": tkr})
        except Exception:
            return None
        return None

    # 1) True ^VIX
    vix = _yf_hist("^VIX")
    if vix is not None:
        return vix.rename(columns={"^VIX": "vix"}).rename(columns={"vix": "vix"})

    # 2) VIXY ETF as proxy (scale to look like VIX levels)
    vixy = _yf_hist("VIXY")
    if vixy is not None:
        # Rescale to rough VIX level using a rolling ratio to SPY realized vol
        vixy = vixy.rename(columns={"VIXY": "vixy"})
        return vixy.rename(columns={"vixy": "vix"})

    # 3) VXX ETN as proxy
    vxx = _yf_hist("VXX")
    if vxx is not None:
        vxx = vxx.rename(columns={"VXX": "vix"})
        return vxx

    # 4) Give up: caller will use realized-vol proxy from SPY
    return None


def regime_splits(daily: pd.DataFrame, petf: pd.DataFrame):
    """
    Split KPIs by regimes:
      - SPY 200d trend (up / down)
      - VIX (<= 20 low / > 20 high)  -- configurable threshold
    """
    # merge SPY prices
    spy = petf[petf["ticker"]=="SPY"].sort_values("date").copy()
    spy["ret"] = (spy["close"] / spy["close"].shift(1)) - 1.0
    spy["trend200"] = (spy["close"] / spy["close"].rolling(200, min_periods=50).mean()) - 1.0
    spy_flags = spy[["date","trend200"]].copy()

    # VIX
    vix = get_vix_series()
    if vix is not None:
        vix["date"] = pd.to_datetime(vix["date"])
        vix_flags = vix.copy()
    else:
        # fallback: proxy VIX with realized volatility of SPY
        vix_flags = spy[["date","ret"]].copy()
        vix_flags["vix"] = vix_flags["ret"].rolling(21, min_periods=10).std() * np.sqrt(252) * 100

    # merge with daily pnl
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    df = d.merge(spy_flags, on="date", how="left").merge(vix_flags[["date","vix"]], on="date", how="left")

    df["trend_regime"] = np.where(df["trend200"] > 0, "SPY_uptrend", "SPY_downtrend")
    vix_thr = 20.0
    df["vix_regime"] = np.where(df["vix"] > vix_thr, f"VIX>{vix_thr}", f"VIX<={vix_thr}")

    # KPIs by regime
    out_rows = []
    for col, name in [("trend_regime","Trend"), ("vix_regime","VIX")]:
        for reg, sub in df.groupby(col):
            k = kpis_from_daily(sub)
            out_rows.append({"split": name, "regime": reg, **k})
    out = pd.DataFrame(out_rows).sort_values(["split","regime"])
    out.to_csv(os.path.join(FIG_DIR, "regime_kpis.csv"), index=False)

    # small bar charts of Sharpe by regime
    for split_name, sub in out.groupby("split"):
        ser = sub.set_index("regime")["sharpe"]
        plot_bar(ser, f"Sharpe by Regime: {split_name}", os.path.join(FIG_DIR, f"sharpe_by_{split_name.lower()}.png"))

    return out

def main():
    daily, trades, signals_const, pe, petf = load_core()

    # 1) Hedge quality / exposure
    hed = rolling_parent_exposure(trades)

    # 2) Decile monotonicity of signal
    deciles = decile_buckets(signals_const, pe)

    # 3) Daily IC time series
    ic = daily_ic(signals_const, pe)

    # 4) Regime splits
    regimes = regime_splits(daily, petf)

    # 5) Write a small deep dive report
    md = []
    md.append("# Deep Dive")
    md.append("")
    md.append("## Figures")
    figs = [
        "net_parent_exposure.png", "decile_monotonicity.png",
        "daily_ic.png", "sharpe_by_trend.png", "sharpe_by_vix.png"
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
