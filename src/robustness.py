from __future__ import annotations
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple

def month_edges(dates: pd.DatetimeIndex, months: int) -> List[pd.Timestamp]:
    start = dates.min()
    end = dates.max()
    edges = [start]
    while edges[-1] < end:
        edges.append(edges[-1] + relativedelta(months=months))
    return edges

def walkforward_grid(
    dates: pd.DatetimeIndex,
    param_grid: Dict[str, List],
    train_months: int,
    oos_months: int,
    run_once
) -> pd.DataFrame:
    """
    run_once(params, start_date, end_date) -> dict performance
    Splits the period into windows and evaluates a small grid.
    """
    edges = month_edges(dates, oos_months)
    rows = []
    i = 0
    while True:
        oos_start = edges[i]
        oos_end = edges[i+1] if i+1 < len(edges) else dates.max()
        train_start = oos_start - relativedelta(months=train_months)
        if train_start < dates.min(): 
            i += 1
            if i+1 >= len(edges): break
            continue
        # grid
        from itertools import product
        keys = list(param_grid.keys())
        for combo in product(*[param_grid[k] for k in keys]):
            params = dict(zip(keys, combo))
            perf = run_once(params, train_start, oos_end)  # you can use train for fit if needed
            perf.update(params)
            perf["train_start"] = train_start
            perf["oos_start"] = oos_start
            perf["oos_end"] = oos_end
            rows.append(perf)
        i += 1
        if i+1 >= len(edges): break
    return pd.DataFrame(rows)
