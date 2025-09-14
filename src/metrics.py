from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def summarize_performance(daily_pnl: pd.DataFrame, trading_days_per_year: int = 252) -> Dict:
    ret = daily_pnl["pnl"].fillna(0.0)
    mu = ret.mean() * trading_days_per_year
    sigma = ret.std(ddof=1) * np.sqrt(trading_days_per_year)
    sharpe = mu / (sigma + 1e-12)

    curve = (1.0 + ret).cumprod()
    peak = curve.cummax()
    dd = (curve / peak) - 1.0
    max_dd = dd.min()

    return {
        "ann_return": float(mu),
        "ann_vol": float(sigma),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "final_equity": float(curve.iloc[-1]) if len(curve) > 0 else 1.0
    }
