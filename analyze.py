from __future__ import annotations
import os
import pandas as pd
from src.reporting import (
    ensure_dir, plot_equity_curve, plot_drawdown, plot_rolling_sharpe,
    plot_daily_pnl_hist, plot_signal_distributions, plot_turnover,
    top_contributors, signal_forward_corr
)

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

def main():
    ensure_dir(FIG_DIR)

    # Load core outputs
    daily = pd.read_csv(os.path.join(RESULTS_DIR, "daily_pnl.csv"), parse_dates=["date"])
    trades = pd.read_csv(os.path.join(RESULTS_DIR, "trades.csv"), parse_dates=["date"])
    # Optional extras saved by main.py patch:
    signals_const = None
    sig_path = os.path.join(RESULTS_DIR, "signals_constituent.csv")
    if os.path.exists(sig_path):
        signals_const = pd.read_csv(sig_path, parse_dates=["date"])

    # Prices (for signal diagnostics)
    pe = pd.read_csv(os.path.join("data","prices_equity.csv"), parse_dates=["date"])

    # 1) Performance plots
    plot_equity_curve(daily, FIG_DIR)
    plot_drawdown(daily, FIG_DIR)
    plot_rolling_sharpe(daily, FIG_DIR, window=63)
    plot_daily_pnl_hist(daily, FIG_DIR)

    # 2) Portfolio structure/ops
    plot_turnover(trades, FIG_DIR)
    top_contributors(trades, FIG_DIR, k=20)

    # 3) Signal distributions & forward effectiveness
    if signals_const is not None:
        plot_signal_distributions(signals_const, FIG_DIR)
        signal_forward_corr(signals_const, pe, FIG_DIR)

    # 4) Emit a short markdown recap
    md = []
    md.append("# Analysis Report")
    md.append("")
    md.append("## Key Figures")
    for fn in ["equity_curve.png","drawdown.png","rolling_sharpe.png","pnl_hist.png",
               "turnover.png","top_contributors.png","flow_z_hist.png","pressure_hist.png",
               "signal_forward_corr.png"]:
        fpath = os.path.join("results","figures", fn)
        if os.path.exists(fpath):
            md.append(f"![{fn}]({fpath})")
    with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(md))

    print("Wrote figures to:", FIG_DIR)
    print("Wrote report:", os.path.join(RESULTS_DIR, "REPORT.md"))

if __name__ == "__main__":
    main()
