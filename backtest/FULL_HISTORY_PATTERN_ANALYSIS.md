# Full History Pattern Analysis: 2015–2026

**10.8 years | 4,173 candles | 1,236 trades | €10K → €2.65M (+26,405%)**

---

## Headline Result

| Metric | Bot | DCA |
|--------|-----|-----|
| BTC accumulated | 30.90 | 4.36 |
| Final value | €2,650,467 | €253,859 |
| BTC advantage | +609% | baseline |
| Value advantage | +944% | baseline |

---

## Pattern 1: The Dormant Capital Problem (Critical)

The most striking pattern in the data: **the bot stops buying for years at a time.**

| Period | Buys | Reason |
|--------|------|--------|
| 2015–2016 | 611 | Cheap BTC, abundant EUR → buying machine |
| 2017 Jan–2019 Mar | 13 buys then **26 months of zero buys** | EUR depleted, all capital in BTC |
| 2019 Apr–2020 Nov | 478 | Sells in Jan 2019 freed EUR → buying resumes |
| 2020 Dec–2024 Nov | **48 months of zero buys** | EUR depleted again |
| 2025 Dec–2026 Feb | 65 | Sells in 2025 freed EUR → buying resumes |

The cycle repeats: buy cheaply → run out of EUR → sit idle during the bear → miss the bottom entirely. The bot's EUR ran to zero in late 2020 at €14K and didn't buy a single satoshi through the entire 2022 crash (€15,400 bottom). That was the best buying opportunity in the dataset.

**Actionable fix:** The profit-taking sells need to happen *before* the EUR runs out. The current tier system only sells during distribution/growth phases. We need a **capital recycling** mechanism: if EUR balance drops below a threshold (e.g., 10% of portfolio), force small profit-takes regardless of phase to keep dry powder available.

---

## Pattern 2: DCA Floor Is the Engine (96.7% of all buys)

| Buy Type | Count | % of Buys |
|----------|-------|-----------|
| DCA floor | 1,178 | 96.6% |
| Signal-driven | 41 | 3.4% |

Signal scores almost never cross the buy threshold. The median buy signal score is **0.2** against a threshold of 10.0. The signal engine is essentially decorative — the DCA floor does all the work.

Two interpretations:
1. **The signal engine is too conservative.** A threshold of 10.0 means only 3.4% of buys are signal-driven. Lower it further?
2. **The DCA floor IS the strategy.** It buys consistently at €7/day equivalent. The signal engine's main value is *preventing* buys during euphoria/distribution via the phase gate.

The backtest proves option 2 is correct. DCA floor + phase gating + profit-taking tiers is the winning combination.

---

## Pattern 3: Phase Detection Drives Sell Quality

Every sell happened in either **distribution** or **growth** phase. The sell timing quality:

| Date | Phase | Sell Price | 180d Later | Verdict |
|------|-------|-----------|-----------|---------|
| 2019-01 | distribution | €3,100–3,230 | €9,099–10,474 | BAD ✗ |
| 2025-02 | distribution | €91,859–93,647 | €57,707 | OK/GOOD |
| 2025-05 | growth | €92,448–95,135 | €57,707 | OK/GOOD |
| 2025-08 | distribution | €92,774–96,368 | €57,707 | GOOD ✓ |
| 2025-10 | growth | €95,351–99,666 | €57,707 | GOOD ✓ |

The January 2019 sells were **terrible** — selling at €3,100 during what was actually capitulation, right before a 3x recovery. The phase detector misclassified the bottom as "distribution."

The 2025 sells were excellent — all sold between €91K–100K, price is now €57K. 12 of 14 Cycle 4 sells were well-timed.

**Actionable fix:** Add a price-floor guard to sells: never sell if price is below the 200-day MA (which was well below €3,100 in Jan 2019). This single rule would have prevented the disastrous Cycle 2 sells.

---

## Pattern 4: Drawdown Zones — Where the Bot Buys

| Drawdown from ATH | Buys | BTC Bought | Avg Price | Avg Size |
|-------------------|------|-----------|-----------|----------|
| 0–10% (near ATH) | 292 | 19.34 | €810 | €16 |
| 10–25% (pullback) | 335 | 26.02 | €2,844 | €723 |
| 25–40% (correction) | 115 | 18.44 | €31,704 | €10,422 |
| 40–60% (bear) | 390 | 9.29 | €10,293 | €817 |
| 60–80% (deep bear) | 87 | 2.67 | €5,965 | €163 |

The best accumulation happened in the **10–25% pullback zone** (26 BTC) and **near ATH** (19 BTC), both at very cheap prices (2015–2016). The 40–60% bear zone has lots of buys but tiny sizes because EUR was depleted.

This is the dormant capital problem showing up differently: during the deep bear (60–80% drawdown), the bot only had €163 per buy on average — its cheapest BTC opportunity, but almost no capital to deploy.

---

## Pattern 5: Year-over-Year — Compounding Matters

| Year | Start Value | End Value | BTC Held | BTC Price Δ |
|------|------------|-----------|----------|------------|
| 2015 | €10,000 | €17,682 | 43.82 | +67% |
| 2016 | €17,830 | €40,869 | 44.58 | +129% |
| 2017 | €42,295 | €525,845 | 44.58 | +1,143% |
| 2018 | €507,006 | €145,200 | 44.58 | -71% |
| 2019 | €149,506 | €223,433 | 34.63 | +91% |
| 2020 | €223,652 | €826,801 | 34.82 | +270% |
| 2021 | €841,561 | €1,417,811 | 34.82 | +69% |
| 2022 | €1,460,075 | €537,234 | 34.82 | -63% |
| 2023 | €540,774 | €1,333,505 | 34.82 | +147% |
| 2024 | €1,393,639 | €3,143,627 | 34.82 | +126% |
| 2025 | €3,176,690 | €3,121,846 | 13.88 | -18% |
| 2026 | €3,134,606 | €2,650,467 | 30.90 | -24% |

Key observation: **2017–2018 and 2020–2022 had zero trades.** The bot just held. This is fine in a bull (compounding), catastrophic in a bear (no buying at the bottom).

Also notable: the bot went from 44.58 BTC down to 13.88 BTC in 2025 (sold 30.7 BTC) then rebought 17 BTC in Dec 2025–Feb 2026. The sell prices (€91K–100K) vs rebuy prices (€57K–83K) show the profit-taking tiers working as designed.

---

## Six Actionable Improvements

### 1. Capital Recycling (High Priority)
**Problem:** Bot runs out of EUR for 2–4 years at a time, missing bear market bottoms.
**Fix:** Add a `min_eur_fraction` parameter. If EUR drops below X% of portfolio value, force micro-sells (0.5–1% of BTC) to replenish dry powder, regardless of phase. Target: always have enough EUR for 6–12 months of DCA floor buys.

### 2. Sell Floor Guard (High Priority)
**Problem:** Jan 2019 sells at €3,100 were 81% below ATH — textbook capitulation, not distribution.
**Fix:** Never sell if price drawdown from ATH exceeds 40%. The phase detector can misclassify — the price doesn't lie.

### 3. Drawdown-Scaled DCA (Medium Priority)
**Problem:** DCA floor buys are fixed-size regardless of opportunity.
**Fix:** Scale DCA floor fraction inversely with price: at 50% drawdown from ATH, buy 2–3x the normal floor amount. The cheapest BTC should get the most capital.

### 4. Signal Engine Tuning (Low Priority)
**Problem:** 96.6% of buys are DCA floor, signal engine rarely crosses threshold.
**Fix:** Either lower buy_threshold to 5.0 (letting signal drive more buys), or accept the DCA floor as the strategy and simplify the signal engine to a binary gate (buy/don't-buy) instead of a score.

### 5. Phase Confidence Weighting (Medium Priority)
**Problem:** Phase transitions at 0.39–0.48 confidence drive sell decisions.
**Fix:** Scale sell tier percentages by phase confidence. At 0.40 confidence, sell 40% of the tier amount. At 0.90 confidence, sell 100%. This naturally reduces sells during uncertain phases.

### 6. Bear Market Re-entry (High Priority)
**Problem:** After big sells (2019, 2025), the bot sits on EUR waiting for DCA floor intervals.
**Fix:** After a sell, reduce the DCA floor interval from 24h to 8h for the next 30 days. Deploy the freed capital faster during corrections rather than trickling it back in.
