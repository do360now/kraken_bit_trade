# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bitcoin accumulation trading bot for the Kraken exchange. It uses a dual-loop architecture with technical analysis, cycle detection, risk management, and LLM-powered market analysis.

## Running the Bot

```bash
python main.py
```

For paper trading (simulated trades without real money):
```bash
# Set in config or .env
paper_trade=True
```

## Running Tests

```bash
pytest tests/                      # All tests
pytest tests/test_main.py          # Single test file
pytest tests/ -v                   # Verbose output
```

## Running Backtests

```bash
python backtest/run_backtest.py
```

Backtest results are stored in `backtest/backtest_results/`.

## Architecture

### Dual-Loop Design

- **Fast loop** (~2 min): Core decision pipeline - ticker → indicators → cycle → signal → risk → size → execute
- **Slow loop** (~30 min): On-chain metrics + LLM analysis (cached for fast loop)

### Pipeline Flow

1. `main.py` orchestrates the entire bot
2. `kraken_api.py` - Exchange connectivity (ticker, OHLC, orders)
3. `indicators.py` - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
4. `cycle_detector.py` - Bitcoin halving cycle analysis
5. `signal_engine.py` - Composite signal generation from all inputs
6. `risk_manager.py` - Risk checks, emergency sells, profit-taking
7. `position_sizer.py` - DCA sizing and sell decisions
8. `trade_executor.py` - Order execution with circuit breakers
9. `ollama_analyst.py` - LLM-powered market analysis (Ollama)
10. `bitcoin_node.py` - On-chain metrics from Bitcoin node

### Configuration

All settings in `config.py`. Uses `.env` for API credentials:
- `KRAKEN_API_KEY`
- `KRAKEN_API_SECRET`
- `BITCOIN_RPC_URL`, `BITCOIN_RPC_USER`, `BITCOIN_RPC_PASSWORD`
- `OLLAMA_URL`, `OLLAMA_MODEL`

### Backtesting

`backtest/backtester.py` replays historical candles through the REAL pipeline modules. Only trade execution is simulated. Data in `backtest/data/`.
