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

def build_holdings_from_market_cap(prices_equity_csv: str, etfs: list[str], out_path: str, max_constituents: int=100):
    import yfinance as yf
    pe = pd.read_csv(prices_equity_csv, parse_dates=["date"])
    if pe.empty: return
    px = pe.pivot(index="date", columns="ticker", values="close")
    frames = []
    for etf in etfs:
        const = _get_wiki_constituents(etf)
        const = [t for t in const if t in px.columns]
        if not const: continue
        shares = {}
        for t in const:
            so=None
            try:
                ti=yf.Ticker(t)
                so=getattr(ti.fast_info,"shares_outstanding",None)
                if not so: so=(ti.info or {}).get("sharesOutstanding")
            except Exception: pass
            if so and so>0: shares[t]=so
        universe=[t for t in const if t in shares]
        if not universe: continue
        cap = px[universe].multiply(pd.Series(shares), axis=1)
        w = cap.div(cap.sum(axis=1), axis=0)
        rows=[]
        for dt,row in w.iterrows():
            sr=row.dropna().sort_values(ascending=False).head(max_constituents)
            for tk,wt in sr.items():
                rows.append((dt.date().isoformat(), etf, tk, float(wt)))
        df=pd.DataFrame(rows, columns=["date","etf","constituent","weight"])
        frames.append(df)
    if frames:
        allh=pd.concat(frames, ignore_index=True).drop_duplicates(["date","etf","constituent"], keep="last")
        allh.to_csv(out_path,index=False)
        info(f"Wrote: {out_path} ({len(allh):,} rows) [auto-holdings]")
    else:
        # fallback equal-weights on UNIVERSE_EQ
        dates=sorted(pe["date"].dt.date.unique())
        have=sorted(set(UNIVERSE_EQ)&set(pe["ticker"].unique()))
        rows=[]
        for etf in etfs:
            if not have: continue
            w=1.0/len(have)
            for d in dates:
                for tk in have: rows.append((d.isoformat(), etf, tk, w))
        if rows:
            fb=pd.DataFrame(rows, columns=["date","etf","constituent","weight"])
            fb.to_csv(out_path,index=False)
            info(f"Wrote: {out_path} ({len(fb):,} rows) [fallback equal-weights]")

def build_holdings(out_path: str):
    frames=[]
    # 1) iShares programmatic
    for etf,url in ISHARES_URLS.items():
        try:
            r=requests.get(url,timeout=30); r.raise_for_status()
            df=pd.read_csv(io.StringIO(r.text))
            frames.append(_standardize_holdings_df(df,etf))
        except Exception as e: info(f"! iShares fetch failed for {etf}: {e}")
    # 2) SPY/QQQ manual files
    for etf in ["SPY","QQQ"]:
        files=sorted(glob.glob(os.path.join(RAW_DIR,f"{etf}_holdings_*.*")))
        for fp in files:
            try:
                df=_read_any(fp)
                m=re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(fp))
                d=m.group(1) if m else None
                frames.append(_standardize_holdings_df(df,etf,date_str=d))
                info(f"parsed {etf} holdings: {os.path.basename(fp)}")
            except Exception as e: info(f"! could not parse {fp}: {e}")
    if frames:
        allh=pd.concat(frames,ignore_index=True)
        allh=(allh.sort_values(["date","etf","constituent"])
                  .drop_duplicates(["date","etf","constituent"],keep="last"))
        allh.to_csv(out_path,index=False)
        info(f"Wrote: {out_path} ({len(allh):,} rows)")
    else:
        info("No issuer holdings — auto-building from market cap")
        build_holdings_from_market_cap(PRICES_EQ_CSV, UNIVERSE_ETF, out_path)

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
    rows=[]
    for t in etfs:
        df=etf[etf["ticker"]==t].sort_values("date").copy()
        if df.empty: continue
        df["dollar_volume"]=df["close"]*df["volume"]
        df["dv_chg"]=df["dollar_volume"].pct_change()
        df["dv_ma"]=df["dollar_volume"].rolling(20,min_periods=5).mean()
        df["dv_std"]=df["dv_chg"].rolling(60,min_periods=20).std()
        df["flow_z"]=(df["dv_chg"]-df["dv_chg"].rolling(60,min_periods=20).mean())/(df["dv_std"]+1e-9)
        df["net_flow_usd"]=df["flow_z"]*df["dv_ma"].fillna(0.0)
        if t in so_map: df["aum_usd"]=df["close"]*so_map[t]
        else: df["aum_usd"]=df["dv_ma"].fillna(df["dollar_volume"].rolling(60,min_periods=20).mean())
        rows.append(df[["date"]].assign(etf=t,net_flow_usd=df["net_flow_usd"],aum_usd=df["aum_usd"]))
    if rows:
        out=pd.concat(rows,ignore_index=True).dropna(subset=["date"])
        out.to_csv(out_path,index=False)
        info(f"Wrote: {out_path} ({len(out):,} rows) [auto-flows proxy]")

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
