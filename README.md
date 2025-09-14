# ETF Flow / Rebalancing Arbitrage (Equities)

A lightweight, professional-grade equities arbitrage framework focused on ETF flow and rebalancing strategies, with hedging, costs, and walk-forward robustness.

The core idea is that ETF fund flows and periodic rebalancing create predictable, mechanical demand/supply in constituent equities, which can be exploited in a market-neutral fashion.

The project is designed for quant researchers, algorithmic traders, and systematic investors aiming to build structural, non-retail strategies.

## ⚙️ Core Logic
The framework follows a clear signal-to-trade pipeline:

1. ETF Flow Signal
- Compute daily flow factor from net flows / AUM.
- Normalize via rolling z-scores.

2. Propagation to Constituents

- Apply daily ETF holdings weights to distribute flow pressure across stocks.
- Result: “pressure” score per constituent.

3. Constituent Selection

- Rank stocks by pressure.
- Select top/bottom k names for long/short positions.

4. Portfolio Construction

- Position sizing via risk parity or equal-weighting.
- Apply gross leverage scaling.

5. Hedging

- Neutralize ETF-level exposure with the parent ETF (SPY, QQQ, etc.).
- Hedge method: notional or regression-based.

6. Backtesting

- Simulate realistic trading:
  - Trade-at-next-open or trade-at-close.
  - Transaction costs & slippage.
  - Daily rebalance & PnL logging.

7. Robustness

- Walk-forward parameter grid.
- Out-of-sample validation.
- Stress tests (costs, slippage, noise).

## 🚀Features
- ETF flow signal engine: uses either issuer NAV/Shares Outstanding or ETF.com flows.
- Holdings propagation: converts ETF-level flows into constituent-level signals.
- Flexible allocation: risk parity or equal-weighted.
- Market-neutral hedging: offset ETF exposure with parent hedge.

- Backtesting engine:
 - Trade-at-next-open or trade-at-close
 - Slippage + cost modeling
 - Daily PnL + equity curve

- Robustness module:
 - Walk-forward analysis
 - Grid search across parameters

- Extensible design:
 - Add rebalance events
 - Add leveraged ETF hedging flows
 - Integrate ADR/local arb in the same structure

## 📂 Repository Structure
etf_flow_arb/
├─ README.md                # Project overview & documentation
├─ requirements.txt         # Python dependencies
├─ .gitignore               # Ignore data/results in Git
├─ LICENSE                  # MIT License
├─ config.json              # Strategy & backtest config
├─ make_data.py             # Data builder (prices, holdings, flows)
│
├─ data/                    # Local data (ignored by Git)
│   ├─ raw/                 # Raw issuer files (SPY/QQQ holdings, NAV/SO, flows)
│   ├─ prices_equity.csv
│   ├─ prices_etf.csv
│   ├─ etf_holdings.csv
│   └─ etf_flows.csv
│
├─ results/                 # Backtest outputs (ignored by Git)
│   ├─ trades.csv
│   ├─ daily_pnl.csv
│   ├─ summary.json
│   └─ walkforward_summary.csv
│
└─ src/                     # Core source code
    ├─ config_loader.py     # Parse & validate config.json
    ├─ data_loader.py       # Read prices, holdings, flows
    ├─ signals.py           # Flow factor & constituent pressure
    ├─ portfolio.py         # Position sizing & risk parity
    ├─ backtester.py        # Execution logic, PnL calculation
    ├─ metrics.py           # Performance stats
    ├─ robustness.py        # Walk-forward & grid search
    ├─ utils.py             # Helpers (seeding, saving, dirs)
    └─ __init__.py

## Quickstart
```bash
git clone https://github.com/leoleontidis/etf_flow_arb.git
cd etf_flow_arb

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install -r requirements.txt

# Reset data (clears data and results folder for a fresh run)
python reset.py

# Build data (fetches prices, standardizes holdings/flows)
python make_data.py

# Backtest
python main.py --mode backtest

# Standard analysis (equity, drawdown, rolling Sharpe, distribs, etc.)
python analyze.py

# Deep dive (exposure, deciles, IC, regimes)
python deep_dive.py

# Walk-forward robustness
python main.py --mode walkforward