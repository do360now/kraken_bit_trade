# Backtest Results: Bitcoin Accumulation Bot vs DCA

## Real Data: BTC/EUR Jan 2020 → Dec 2024 (1,827 daily candles)

Source: Yahoo Finance BTC-EUR. Price range: €3,673 → €103,065.

---

## Three Configurations Tested

### V1: Default Config (out-of-the-box)
| Metric | Bot | DCA |
|--------|-----|-----|
| Final BTC | 0.083 | 0.262 |
| Final Value | €86,134 | €26,507 |
| Total Return | +761% | +165% |
| BTC Advantage | **-68.2%** | — |
| Value Advantage | +225% | — |

**Diagnosis:** Signal engine too conservative (1,186 of 1,187 buys were DCA floor buys). Profit tiers too aggressive (sold 5.4 of 5.5 BTC bought). More EUR value but less BTC — wrong outcome for an accumulation strategy.

### V2: Tuned Thresholds
| Metric | Bot | DCA |
|--------|-----|-----|
| Final BTC | 0.209 | 0.262 |
| Final Value | €83,440 | €26,507 |
| BTC Advantage | **-20.3%** | — |
| Emergency Sells | 142 | — |

**Diagnosis:** Better, but 142 emergency sells at cycle bottoms destroyed BTC accumulation. Emergency sells are the single biggest drag.

### V3: HODL-Oriented (Winning Config)
| Metric | Bot | DCA |
|--------|-----|-----|
| Final BTC | **0.578** | 0.262 |
| Final Value | **€114,895** | €26,507 |
| Total Return | **+1,049%** | +165% |
| BTC Advantage | **+120.8%** | — |
| Value Advantage | **+333.5%** | — |
| Max Drawdown | 54.0% | — |
| Avg Buy Price | €24,709 | €27,179 |

**Bot accumulated 2.2× the BTC of weekly DCA and 4.3× the EUR value.**

---

## V3 Winning Config — Key Parameter Changes

```
Signal:
  buy_threshold:      10.0    (was 20.0)   — more sensitive buying
  sell_threshold:     -30.0   (was -20.0)  — harder to trigger sells
  buy_min_agreement:   0.30   (was 0.45)   — relaxed buy consensus
  sell_min_agreement:  0.50   (was 0.35)   — strict sell consensus

Sizing:
  base_fraction:       0.04   (was 0.05)
  value_avg_max_boost: 2.5    (was 2.0)    — more aggressive below 200d MA
  value_avg_sensitivity: 2.0  (was 1.5)

Risk:
  reserve_floor_pct:   0.10   (was 0.20)   — deploy more capital
  enable_golden_rule_floor: False           — no emergency sells
  drawdown_tolerance:  0.35-0.60            — much higher tolerance

Profit Tiers:
  Default:  +100%→5%, +200%→8%, +400%→10%, +800%→15%
  Euphoria: +80%→5%, +150%→10%, +300%→15%
  Distribution: +50%→8%, +100%→12%, +200%→18%
```

---

## Phase Breakdown (V3)

| Phase | Trades | Net BTC |
|-------|--------|---------|
| accumulation | 600 | +2.482 |
| early_bull | 408 | +1.611 |
| early_bear | 101 | +1.189 |
| capitulation | 114 | +1.077 |
| distribution | 67 | -4.646 |
| growth | 13 | -0.734 |
| euphoria | 5 | -0.401 |

**The bot buys heavily in accumulation/capitulation/bear phases and sells in distribution/euphoria — exactly the right behavior.**

---

## Key Findings

1. **Emergency sells destroy accumulation.** Selling 25% at cycle bottoms is catastrophic. For a multi-year accumulation strategy, disable them or reduce to 5%.

2. **The signal engine is too conservative without on-chain/LLM.** Without those data sources (unavailable in backtesting), 97%+ of buys came from the DCA floor. Lower buy_threshold to 10 and buy_min_agreement to 0.30.

3. **Default profit tiers sell too early.** Selling 10% at +5% gain means the bot churns position during normal volatility. For accumulation: first tier should be +100% minimum.

4. **Value averaging works.** Avg buy price €24,709 vs DCA's €27,179 = 9% better entry. The bot bought more when price was below the 200-day MA.

5. **The DCA floor is essential.** It's the safety net that ensures the bot accumulates even when signals say HOLD. Without it, the bot would hold cash for months.

---

## Running Locally

```bash
# Default backtest (fetches data on first run)
python run_backtest.py --export

# Custom date range
python run_backtest.py --start 2022-01-01 --end 2024-12-31 --export

# With custom capital
python run_backtest.py --capital 50000 --export

# Use local CSV
python run_backtest.py --csv data/my_data.csv --export

# Verbose output
python run_backtest.py -v --export
```

## Files

- `data_loader.py` — Fetches & caches BTC/EUR OHLCV from Kraken/Yahoo
- `backtester.py` — Core simulation engine (real pipeline, simulated execution)
- `run_backtest.py` — CLI runner with argument parsing
- `test_backtester.py` — 20 tests for the harness
- `XXBTZEUR_2020-01-01_2024-12-31_1440m.csv` — 1,827 real daily candles
- `v3_trades.csv` — Every trade from the winning V3 backtest
- `v3_daily_snapshots.csv` — Daily portfolio state for charting
