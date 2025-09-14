#!/usr/bin/env python
"""
Builds the canonical CSVs in /data for the ETF Flow / Rebalancing framework, using free sources.

Outputs:
  data/prices_equity.csv
  data/prices_etf.csv
  data/etf_holdings.csv
  data/etf_flows.csv
"""

from __future__ import annotations
import os, io, sys, glob, re, time, requests
from datetime import date
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# --------- USER CONFIG ------------------------------------------------------

UNIVERSE_ETF  = ["SPY", "QQQ", "IWM"]
UNIVERSE_EQ   = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"]
START_DATE    = "2023-01-01"
END_DATE      = None

ISHARES_URLS: Dict[str, str] = {
    # Example:
    # "IWM": "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/?dataType=fund&fileName=IWM_holdings&fileType=csv",
}

DATA_DIR      = "data"
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PRICES_EQ_CSV = os.path.join(DATA_DIR, "prices_equity.csv")
PRICES_ETF_CSV= os.path.join(DATA_DIR, "prices_etf.csv")
HOLDINGS_CSV  = os.path.join(DATA_DIR, "etf_holdings.csv")
FLOWS_CSV     = os.path.join(DATA_DIR, "etf_flows.csv")

UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PythonRequests/2.x"}

# ----------------------------------------------------------------------------

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def info(msg: str): print(f"[make_data] {msg}")

# --------------------- PRICES (Stooq + Yahoo) -------------------------------

def stooq_symbol(t: str) -> str:
    t = t.strip().upper()
    if not t.endswith(".US"):
        t = t + ".US"
    return t.lower()

def fetch_stooq_csv(tkr: str, start: str=None, end: str=None) -> pd.DataFrame | None:
    from io import StringIO
    sym = stooq_symbol(tkr)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA_HEADERS, timeout=15)
            if r.status_code == 200 and r.text.strip():
                d = pd.read_csv(StringIO(r.text))
                d = d.rename(columns=str.lower)
                if "date" not in d.columns:
                    return None
                d["ticker"] = tkr.upper()
                d["date"] = pd.to_datetime(d["date"])
                if start: d = d[d["date"] >= pd.to_datetime(start)]
                if end:   d = d[d["date"] <= pd.to_datetime(end)]
                return d[["date","ticker","open","high","low","close","volume"]].dropna(subset=["open","close"])
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None

def build_prices(tickers: List[str], out_path: str, start: str, end: Optional[str]):
    import yfinance as yf
    rows, failed = [], []
    for t in tickers:
        df = fetch_stooq_csv(t, start, end)
        if df is None:
            try:
                d = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=False)
                if len(d):
                    d = d.reset_index().rename(columns=str.lower)
                    if "date" not in d.columns and "index" in d.columns:
                        d = d.rename(columns={"index":"date"})
                    d["ticker"] = t.upper()
                    df = d[["date","ticker","open","high","low","close","volume"]]
            except Exception:
                df = None
        if df is None or df.empty:
            failed.append(t); continue
        rows.append(df)

    if rows:
        out = (pd.concat(rows, ignore_index=True)
                .dropna(subset=["date"])
                .sort_values(["ticker","date"]))
        out.to_csv(out_path, index=False)
        info(f"Wrote: {out_path} ({len(out):,} rows)")
    else:
        pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"]).to_csv(out_path, index=False)
        info(f"Wrote: {out_path} (0 rows)")
    if failed: info(f"Failed tickers ({len(failed)}): {failed}")

# ---------------------- HOLDINGS --------------------------------------------

def _standardize_holdings_df(df: pd.DataFrame, etf_symbol: str, date_str: Optional[str]=None) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    cand_ticker = [c for c in cols if re.search(r"(ticker|symbol)", c)]
    cand_weight = [c for c in cols if "weight" in c or "%" in c]
    if not cand_ticker or not cand_weight:
        raise ValueError(f"Cannot find ticker/weight cols for {etf_symbol}: {list(df.columns)}")
    tcol, wcol = cols[cand_ticker[0]], cols[cand_weight[0]]
    out = pd.DataFrame({
        "constituent": df[tcol].astype(str).str.upper().str.strip(),
        "weight": pd.to_numeric(df[wcol].astype(str).str.replace("%","").str.replace(",",""), errors="coerce")
    }).dropna()
    if out["weight"].median() > 1.5: out["weight"] /= 100.0
    out["etf"] = etf_symbol
    out["date"] = date_str or date.today().isoformat()
    return out[["date","etf","constituent","weight"]]

def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx",".xls")): return pd.read_excel(path)
    return pd.read_csv(path)

def _get_wiki_constituents(etf: str) -> list[str]:
    from io import StringIO
    if etf.upper() == "SPY":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    elif etf.upper() == "QQQ":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    else: return []
    r = requests.get(url, headers=UA_HEADERS, timeout=20); r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    if etf.upper()=="SPY":
        df = tables[0]; tick = df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    else:
        cand = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any(c.startswith("ticker") for c in cols): cand = t; break
        if cand is None: cand = tables[0]
        tcol = [c for c in cand.columns if str(c).lower().startswith("ticker")][0]
        tick = cand[tcol].astype(str).str.upper().str.replace(".", "-", regex=False)
    return tick.tolist()

def build_holdings_from_px_volume(prices_equity_csv: str, etfs: list[str], out_path: str,
                                  max_constituents: int = 100, window: int = 40):
    """
    Fully OFFLINE fallback:
      - load data/prices_equity.csv
      - compute rolling dollar-volume per ticker: dv_ma = MA_n(close*volume)
      - per date, weight_i = dv_ma_i / sum_j dv_ma_j
      - write same weights for each ETF (unless you customize per-ETF membership)
    This creates real cross-sectional variation even with a fixed universe.
    """
    import pandas as pd
    info("Auto-building holdings from price×volume (offline fallback)")
    pe = pd.read_csv(prices_equity_csv, parse_dates=["date"])
    if pe.empty:
        info("[px-vol holdings] equity prices empty; writing empty file")
        pd.DataFrame(columns=["date","etf","constituent","weight"]).to_csv(out_path, index=False)
        return

    pe = pe.sort_values(["ticker","date"]).copy()
    pe["dollar_volume"] = pe["close"] * pe["volume"]
    # rolling mean by ticker
    pe["dv_ma"] = pe.groupby("ticker")["dollar_volume"].transform(
        lambda s: s.rolling(window, min_periods=max(5, window//3)).mean()
    )

    # For stability: drop very early dates where dv_ma missing
    pe = pe.dropna(subset=["dv_ma"])
    if pe.empty:
        info("[px-vol holdings] dv_ma all NaN (short history) — writing empty file")
        pd.DataFrame(columns=["date","etf","constituent","weight"]).to_csv(out_path, index=False)
        return

    # per-date weights
    def weights_one_day(df_day: pd.DataFrame) -> pd.DataFrame:
        sub = df_day.dropna(subset=["dv_ma"]).copy()
        if sub.empty:
            return pd.DataFrame(columns=["constituent","weight"])
        sub["weight"] = sub["dv_ma"] / sub["dv_ma"].sum()
        # cap to top N for tractability
        sub = sub.sort_values("weight", ascending=False).head(max_constituents)
        return sub[["ticker","weight"]].rename(columns={"ticker":"constituent"})

    frames = []
    for dt, g in pe.groupby("date"):
        wdf = weights_one_day(g)
        if wdf.empty:
            continue
        for etf in etfs:
            tmp = wdf.copy()
            tmp.insert(0, "date", dt.date().isoformat())
            tmp.insert(1, "etf", etf)
            frames.append(tmp)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["date","etf","constituent"]).drop_duplicates(["date","etf","constituent"])
        out.to_csv(out_path, index=False)
        info(f"Wrote: {out_path} ({len(out):,} rows) [auto-holdings: px-volume]")
    else:
        info("[px-vol holdings] produced nothing — writing empty file")
        pd.DataFrame(columns=["date","etf","constituent","weight"]).to_csv(out_path, index=False)

def build_holdings_from_market_cap(prices_equity_csv: str, etfs: list[str], out_path: str,
                                   max_constituents: int = 100):
    """
    TRY: Wikipedia constituents + yfinance shares_outstanding (market-cap weights).
    If that fails to produce any rows, DO NOTHING here; build_holdings(...) will
    fall back to a price/volume-based weights builder which is fully offline.
    """
    import pandas as pd, numpy as np, yfinance as yf, time

    info("Auto-building holdings from Wikipedia + market-cap proxy (preferred)")
    try:
        pe = pd.read_csv(prices_equity_csv, parse_dates=["date"])
    except Exception as e:
        info(f"[auto-holdings] cannot read {prices_equity_csv}: {e}")
        return

    if pe.empty:
        info("[auto-holdings] equity prices empty; skipping market-cap path")
        return

    pe = pe.sort_values(["ticker","date"])
    px = pe.pivot(index="date", columns="ticker", values="close")

    # local helper to scrape symbols
    def _get_wiki_constituents(etf: str) -> list[str]:
        import requests, io
        if etf.upper() == "SPY":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        elif etf.upper() == "QQQ":
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        else:
            return []
        r = requests.get(url, headers=UA_HEADERS, timeout=25)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        if etf.upper() == "SPY":
            df = tables[0]
            tick = df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        else:
            cand = None
            for t in tables:
                if any(str(c).lower().startswith("ticker") for c in t.columns):
                    cand = t; break
            if cand is None: cand = tables[0]
            tcol = [c for c in cand.columns if str(c).lower().startswith("ticker")][0]
            tick = cand[tcol].astype(str).str.upper().str.replace(".", "-", regex=False)
        return tick.tolist()

    frames = []
    for etf in etfs:
        try:
            const = _get_wiki_constituents(etf)
        except Exception as e:
            info(f"[auto-holdings] wiki fetch failed for {etf}: {e}")
            const = []
        const = [t for t in const if t in px.columns]
        if not const:
            info(f"[auto-holdings] No overlapping tickers for {etf} — skipping market-cap weighting")
            continue

        # shares_outstanding via yfinance (may fail offline)
        shares = {}
        for t in const:
            so = None
            for attempt in range(2):
                try:
                    ti = yf.Ticker(t)
                    so = getattr(ti.fast_info, "shares_outstanding", None)
                    if not so:
                        inf = ti.info or {}
                        so = inf.get("sharesOutstanding")
                except Exception:
                    so = None
                if so: break
                time.sleep(0.3)
            if so and so > 0:
                shares[t] = so

        universe = [t for t in const if t in shares]
        if not universe:
            info(f"[auto-holdings] No shares data for {etf} — skipping market-cap weighting")
            continue

        cap = px[universe].multiply(pd.Series(shares), axis=1)
        w = cap.div(cap.sum(axis=1), axis=0)

        rows = []
        for dt, row in w.iterrows():
            sr = row.dropna().sort_values(ascending=False).head(max_constituents)
            for tk, wt in sr.items():
                rows.append((pd.Timestamp(dt).date().isoformat(), etf, tk, float(wt)))
        if rows:
            frames.append(pd.DataFrame(rows, columns=["date","etf","constituent","weight"]))

    if frames:
        allh = (pd.concat(frames, ignore_index=True)
                  .sort_values(["date","etf","constituent"])
                  .drop_duplicates(["date","etf","constituent"], keep="last"))
        allh.to_csv(out_path, index=False)
        info(f"Wrote: {out_path} ({len(allh):,} rows) [auto-holdings: market-cap]")
    else:
        info("[auto-holdings] market-cap path produced 0 rows (offline or fetch blocked)")
        # fall through: build_holdings(...) will call the px-volume fallback

def build_holdings(out_path: str):
    """
    Try, in order:
      1) Programmatic iShares CSVs (if ISHARES_URLS set)
      2) Manual issuer files in data/raw/ (SPY, QQQ)
      3) Wikipedia + market-cap (may use yfinance; OK to skip offline)
      4) Price×Volume fallback (fully offline; creates cross-sectional variation)
    """
    ensure_dir(DATA_DIR)
    ensure_dir(RAW_DIR)
    frames = []

    # 1) iShares programmatic
    import requests, io
    for etf, url in ISHARES_URLS.items():
        try:
            info(f"Fetching iShares holdings for {etf}...")
            r = requests.get(url, headers=UA_HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            frames.append(_standardize_holdings_df(df, etf))
        except Exception as e:
            info(f"  ! iShares fetch failed for {etf}: {e}")

    # 2) SPY / QQQ manual files in data/raw/
    for etf in ["SPY","QQQ"]:
        files = sorted(glob.glob(os.path.join(RAW_DIR, f"{etf}_holdings_*.*")))
        for fp in files:
            try:
                df = _read_any(fp)
                m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(fp))
                d = m.group(1) if m else None
                frames.append(_standardize_holdings_df(df, etf, date_str=d))
                info(f"  parsed {etf} holdings: {os.path.basename(fp)}")
            except Exception as e:
                info(f"  ! could not parse {fp}: {e}")

    # If we already have issuer-based frames, write and return.
    if frames:
        allh = (pd.concat(frames, ignore_index=True)
                  .sort_values(["date","etf","constituent"])
                  .drop_duplicates(["date","etf","constituent"], keep="last"))
        allh.to_csv(out_path, index=False)
        info(f"Wrote: {out_path} ({len(allh):,} rows)")
        return

    # 3) Wikipedia + market-cap (best proxy when online+yf ok)
    #    If this writes rows, we stop here.
    old_exists = os.path.exists(out_path)
    before_rows = 0
    if old_exists:
        try:
            before_rows = max(0, len(pd.read_csv(out_path)))
        except Exception:
            before_rows = 0

    try:
        build_holdings_from_market_cap(PRICES_EQ_CSV, UNIVERSE_ETF, out_path, max_constituents=100)
        # Did it produce rows?
        after_rows = 0
        try:
            after_rows = max(0, len(pd.read_csv(out_path)))
        except Exception:
            after_rows = 0
        if after_rows > 0:
            return
    except Exception as e:
        info(f"[auto-holdings] market-cap path raised: {e}")

    # 4) Fully offline fallback: rolling dollar-volume weights
    build_holdings_from_px_volume(PRICES_EQ_CSV, UNIVERSE_ETF, out_path, max_constituents=100, window=40)

# ---------------------- FLOWS -----------------------------------------------

def build_flows_from_price_volume(etf_prices_csv: str, etfs: list[str], out_path: str):
    import yfinance as yf
    etf=pd.read_csv(etf_prices_csv,parse_dates=["date"])
    if etf.empty: return
    so_map={}
    for t in etfs:
        so=None
        try:
            ti=yf.Ticker(t)
            so=getattr(ti.fast_info,"shares_outstanding",None)
            if not so: so=(ti.info or {}).get("sharesOutstanding")
        except Exception: pass
        if so and so>0: so_map[t]=so
    rows = []
    for t in etfs:
        df = etf[etf["ticker"] == t].sort_values("date").copy()
        if df.empty:
            continue
        df["dollar_volume"] = df["close"] * df["volume"]
        df["dv_chg"] = df["dollar_volume"].pct_change()
        df["dv_ma"] = df["dollar_volume"].rolling(20, min_periods=5).mean()
        mu = df["dv_chg"].rolling(60, min_periods=20).mean()
        sd = df["dv_chg"].rolling(60, min_periods=20).std()
        df["flow_z"] = (df["dv_chg"] - mu) / (sd + 1e-9)

        # Proxy flows (optional, useful for sanity ratios)
        df["net_flow_usd"] = df["flow_z"] * df["dv_ma"].fillna(0.0)

        # Proxy AUM
        if t in so_map:
            df["aum_usd"] = df["close"] * so_map[t]
        else:
            df["aum_usd"] = df["dv_ma"].fillna(df["dollar_volume"].rolling(60, min_periods=20).mean())

        rows.append(
            df[["date", "flow_z", "net_flow_usd", "aum_usd"]]
              .assign(etf=t)
        )

    if rows:
        out = (pd.concat(rows, ignore_index=True)
                 .dropna(subset=["date"])
                 .sort_values(["etf", "date"]))
        # >>> WRITE flow_z too <<<
        out.to_csv(out_path, index=False)
        info(f"Wrote: {out_path} ({len(out):,} rows) [auto-flows proxy incl. flow_z]")

def build_flows(out_path: str):
    nav_so=os.path.join(RAW_DIR,"etf_nav_so.csv")
    manual=os.path.join(RAW_DIR,"etf_flows_manual.csv")
    if os.path.exists(nav_so):
        df=pd.read_csv(nav_so,parse_dates=["date"])
        df=df.sort_values(["etf","date"])
        df["so_lag"]=df.groupby("etf")["shares_outstanding"].shift(1)
        df["delta_so"]=df["shares_outstanding"]-df["so_lag"]
        df["net_flow_usd"]=df["delta_so"]*df["nav_usd"]
        df["aum_usd"]=df["shares_outstanding"]*df["nav_usd"]
        out=df.dropna(subset=["net_flow_usd","aum_usd"])[["date","etf","net_flow_usd","aum_usd"]]
        out.to_csv(out_path,index=False); info(f"Wrote: {out_path} ({len(out):,} rows)")
    elif os.path.exists(manual):
        df=pd.read_csv(manual,parse_dates=["date"])
        out=df[["date","etf","net_flow_usd","aum_usd"]].copy()
        out.to_csv(out_path,index=False); info(f"Wrote: {out_path} ({len(out):,} rows)")
    else:
        info("No flows input — auto-building proxy from price/volume")
        build_flows_from_price_volume(PRICES_ETF_CSV,UNIVERSE_ETF,out_path)

# ---------------------- VALIDATION ------------------------------------------

def validate_prices(path: str):
    if not os.path.exists(path): return
    df=pd.read_csv(path,parse_dates=["date"])
    info(f"Validated prices: {path} (rows={len(df):,}, tickers={df['ticker'].nunique()})")

def validate_holdings(path: str):
    if not os.path.exists(path): return
    df=pd.read_csv(path,parse_dates=["date"])
    if df.empty: info(f"{path} is empty")
    else:
        sums=df.groupby(["date","etf"])["weight"].sum().groupby("etf").median()
        info("Median weight sum per ETF (~1.0):")
        for e,v in sums.items(): info(f"  {e}: {v:.3f}")

def validate_flows(path: str):
    if not os.path.exists(path): return
    df=pd.read_csv(path,parse_dates=["date"])
    if df.empty: info(f"{path} is empty")
    else:
        df["flow_pct"]=df["net_flow_usd"]/df["aum_usd"].replace(0,np.nan)
        pct99=df["flow_pct"].abs().quantile(0.99)
        info(f"Flow sanity (99th pct |flow/AUM|): {pct99:.2%}")

# ---------------------- MAIN -------------------------------------------------

def main():
    ensure_dir(DATA_DIR); ensure_dir(RAW_DIR)
    build_prices(UNIVERSE_EQ,PRICES_EQ_CSV,START_DATE,END_DATE)
    build_prices(UNIVERSE_ETF,PRICES_ETF_CSV,START_DATE,END_DATE)
    validate_prices(PRICES_EQ_CSV); validate_prices(PRICES_ETF_CSV)
    build_holdings(HOLDINGS_CSV); validate_holdings(HOLDINGS_CSV)
    build_flows(FLOWS_CSV); validate_flows(FLOWS_CSV)
    info("Done. You can now run: python main.py --mode backtest")

if __name__=="__main__":
    main()
