from __future__ import annotations
import pandas as pd
from src.config_loader import load_config
from src.data_loader import load_prices, load_etf_flows, load_etf_holdings, align_calendar
from src.signals import compute_flow_factor, propagate_flow_to_constituents, select_constituents
from src.portfolio import compute_position_sizes
from src.backtester import backtest_flow_strategy, _rolling_vol
from src.metrics import summarize_performance
from src.robustness import walkforward_grid
from src.utils import ensure_dir, save_json, set_seed

def run_backtest(cfg_path: str = "config.json"):
    cfg = load_config(cfg_path)
    set_seed(cfg.random_seed)

    # Load data
    eq = load_prices(cfg.data.prices_equity)     # (date,ticker)
    etf = load_prices(cfg.data.prices_etf)       # (date,ticker)
    flows = load_etf_flows(cfg.data.etf_flows)   # (date,etf)
    holds = load_etf_holdings(cfg.data.etf_holdings, cfg.universe.max_constituents_per_etf)  # (date,etf,constituent)

    # Filter to chosen ETFs
    flows = flows.loc[(slice(None), cfg.universe.etfs), :]
    holds = holds.loc[(slice(None), cfg.universe.etfs, slice(None)), :]

    # Align calendars
    eq, etf, flows, holds = align_calendar(eq, etf, flows, holds)

    # Compute vol for risk parity
    vol = _rolling_vol(eq, window=20)

    # Flow factor & constituent pressure
    flow_f = compute_flow_factor(flows, z_window=cfg.strategy.flow_z_score_window)
    pres = propagate_flow_to_constituents(flow_f, holds, min_weight=cfg.strategy.min_weight_threshold)
    picks = select_constituents(pres, top_k=cfg.strategy.top_k_constituents, side=cfg.strategy.selection_side)

    # After 'picks' creation:
    picks_reset = picks.reset_index().rename(columns={"constituent":"ticker"})
    picks_reset.to_csv(f"{cfg.output.results_dir}/signals_constituent.csv", index=False)

    # Position sizing
    pos = compute_position_sizes(picks, equity_vol=vol, gross_leverage=cfg.strategy.gross_leverage,
                                 use_risk_parity=cfg.strategy.use_risk_parity)

    # After 'pos' creation:
    pos_reset = pos.reset_index().rename(columns={"ticker":"constituent"})
    pos_reset.to_csv(f"{cfg.output.results_dir}/positions_target.csv", index=False)

    # Backtest
    trades, daily = backtest_flow_strategy(
        equity_prices=eq,
        etf_prices=etf,
        target_weights=pos,
        costs_bps_equity=cfg.costs.bps_per_trade_equity,
        costs_bps_etf=cfg.costs.bps_per_trade_etf,
        trade_at=cfg.backtest.trade_at,
        slippage_bps=cfg.backtest.slippage_bps,
        hedge_with_parent=cfg.strategy.hedge_with_parent_etf,
        hedge_method=cfg.strategy.beta_hedge_method
    )

    # Summarize
    summary = summarize_performance(daily)

    # Save
    ensure_dir(cfg.output.results_dir)
    trades.to_csv(f"{cfg.output.results_dir}/trades.csv", index=False)
    daily.to_csv(f"{cfg.output.results_dir}/daily_pnl.csv")
    save_json(summary, f"{cfg.output.results_dir}/summary.json")

    print("Backtest summary:", summary)

def run_walkforward(cfg_path: str = "config.json"):
    cfg = load_config(cfg_path)
    set_seed(cfg.random_seed)

    # Load once
    eq = load_prices(cfg.data.prices_equity)
    etf = load_prices(cfg.data.prices_etf)
    flows = load_etf_flows(cfg.data.etf_flows)
    holds = load_etf_holdings(cfg.data.etf_holdings, cfg.universe.max_constituents_per_etf)
    flows = flows.loc[(slice(None), cfg.universe.etfs), :]
    holds = holds.loc[(slice(None), cfg.universe.etfs, slice(None)), :]
    eq, etf, flows, holds = align_calendar(eq, etf, flows, holds)

    vol = None  # we’ll recompute inside run_once

    all_dates = eq.index.get_level_values(0).unique()

    def run_once(params, train_start, oos_end):
        # slice
        # (We operate “pretend-fit”: params chosen on train, evaluated through oos_end for simplicity)
        eq_ = eq.loc[(slice(train_start, oos_end), slice(None)), :]
        etf_ = etf.loc[(slice(train_start, oos_end), slice(None)), :]
        flows_ = flows.loc[(slice(train_start, oos_end), slice(None)), :]
        holds_ = holds.loc[(slice(train_start, oos_end), slice(None), slice(None)), :]

        vol_ = _rolling_vol(eq_, window=20)

        flow_f = compute_flow_factor(flows_, z_window=params.get("flow_z_score_window", 60))
        pres = propagate_flow_to_constituents(flow_f, holds_, min_weight=0.002)
        picks = select_constituents(pres, top_k=params.get("top_k_constituents", 15), side="both")

        pos = compute_position_sizes(
            picks, equity_vol=vol_, gross_leverage=params.get("gross_leverage", 1.0), use_risk_parity=True
        )

        trades, daily = backtest_flow_strategy(
            equity_prices=eq_,
            etf_prices=etf_,
            target_weights=pos,
            costs_bps_equity=0.000,   # During WF grid we can set to 0 to compare raw signal quality, or keep real costs.
            costs_bps_etf=0.000,
            trade_at="next_open",
            slippage_bps=0.0,
            hedge_with_parent=True,
            hedge_method="notional"
        )
        from src.metrics import summarize_performance
        perf = summarize_performance(daily)
        return perf

    wf = walkforward_grid(
        dates=all_dates,
        param_grid=cfg.robustness.param_grid,
        train_months=cfg.robustness.walkforward_window_months,
        oos_months=cfg.robustness.oos_window_months,
        run_once=run_once
    )

    ensure_dir(cfg.output.results_dir)
    wf.to_csv(f"{cfg.output.results_dir}/walkforward_summary.csv", index=False)
    print("Walkforward rows:", len(wf))
    print("Top by Sharpe:")
    print(wf.sort_values("sharpe", ascending=False).head(10))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["backtest","walkforward"], default="backtest")
    p.add_argument("--config", default="config.json")
    args = p.parse_args()

    if args.mode == "backtest":
        run_backtest(args.config)
    else:
        run_walkforward(args.config)
