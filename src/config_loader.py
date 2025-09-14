from __future__ import annotations
import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class DataPaths(BaseModel):
    root: str
    prices_equity: str
    prices_etf: str
    etf_flows: str
    etf_holdings: str
    rebalance_events: str

class Universe(BaseModel):
    etfs: List[str]
    max_constituents_per_etf: int = 100

class StrategyParams(BaseModel):
    hold_period_days: int = 1
    top_k_constituents: int = 15
    min_weight_threshold: float = 0.002
    flow_z_score_window: int = 60
    selection_side: str = "both"
    use_risk_parity: bool = True
    hedge_with_parent_etf: bool = True
    beta_hedge_method: str = "notional"
    gross_leverage: float = 1.0

class Costs(BaseModel):
    bps_per_trade_equity: float = 3.0
    bps_per_trade_etf: float = 1.0

class BacktestCfg(BaseModel):
    trade_at: str = "next_open"
    slippage_bps: float = 1.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class RobustnessCfg(BaseModel):
    walkforward_window_months: int = 12
    oos_window_months: int = 3
    param_grid: Dict[str, List[Any]] = Field(default_factory=dict)

class OutputCfg(BaseModel):
    results_dir: str = "results"

class AppConfig(BaseModel):
    random_seed: int = 42
    data: DataPaths
    universe: Universe
    strategy: StrategyParams
    costs: Costs
    backtest: BacktestCfg
    robustness: RobustnessCfg
    output: OutputCfg

def load_config(path: str = "config.json") -> AppConfig:
    with open(path, "r") as f:
        raw = json.load(f)
    return AppConfig(**raw)
