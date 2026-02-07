# Bitcoin Trading Bot — Cycle-Aware Upgrade

## The Problem

Your bot currently trades on short-term signals: RSI, MACD, Bollinger Bands, sentiment, on-chain data. These are all **tactical** — they tell you what's happening in the next hours or days.

But the single most powerful predictor for Bitcoin — the **4-year halving cycle** — is completely absent from your decision-making. This means:

- The bot might **sell during bear capitulation** (historically the best accumulation zone)
- It might **aggressively buy during distribution** (the worst time to enter)
- Stop-losses trigger at the worst moments because they don't know about cycle floors
- Position sizing is uniform regardless of whether we're 100 days or 500 days post-halving

## The Solution: `CycleAnalyzer`

A new module that encodes all the patterns from our EUR-denominated halving cycle analysis and injects them into every trading decision.

### Patterns Encoded

| Pattern | Data | How It's Used |
|---------|------|---------------|
| Diminishing returns | 88x → 28x → 7.4x → 1.8x | Ceiling estimation for take-profit |
| Bear drawdowns | ~75-84% consistently | Floor estimation + stop-loss clamping |
| ATL > prev ATH ("golden rule") | Cycle 3 ATL €15,030 ≈ Cycle 2 ATH €16,680 | Emergency sell blocker |
| Days to ATH | 368 → 525 → 549 → 534 (avg 494) | Phase transition timing |
| Pre-halving rally | Cycle 4 ATH before halving (unprecedented) | Pre-halving accumulation signal |
| Pattern tension | 80% drawdown vs golden rule conflict | Conservative floor = max(both) |

### The 7 Cycle Phases

```
POST_HALVING_ACCUMULATION (0-180d)  → Buy aggressively, 1.5x positions
GROWTH (180-365d)                    → Continue buying, 1.2x positions  
EUPHORIA (365-550d)                  → Reduce buys, tighten stops, 0.5x
DISTRIBUTION (550+d)                 → Minimal buying, protect gains, 0.3x
BEAR_EARLY (ATH +0-6mo)             → Start DCA, cautious 0.8x
BEAR_CAPITULATION (ATH +6-18mo)     → MAXIMUM accumulation, 2.0x positions
PRE_HALVING (6mo before halving)     → Strong buying, 1.3x positions
```

## Files Changed

### New File: `cycle_analyzer.py`

The core engine. Simple public interface, complex private implementation (Ousterhout-style).

```python
cycle = CycleAnalyzer(current_cycle=4)

# What phase are we in?
phase = cycle.get_cycle_phase(current_price_eur=85000)
# → CyclePhase.DISTRIBUTION (we're ~657 days post-halving as of Feb 2026)

# Get all trading parameter multipliers
adj = cycle.get_cycle_adjustments(current_price_eur=85000)
# → position_size=0.3x, buy_aggr=0.2x, sell_reluct=0.3x, dca=0.3x

# How good is this moment for accumulation? (0-100)
score = cycle.get_accumulation_score(current_price_eur=85000)
# → ~35 (moderate — we're in distribution phase but with some drawdown)

# Full dashboard data
summary = cycle.get_cycle_summary(current_price_eur=85000)
```

### Modified File: `risk_manager.py`

Every risk decision now flows through cycle awareness:

**Position sizing** — base size × risk adjustment × cycle multiplier × DCA intensity:
- Bear capitulation at €25,000: `base × 1.0 × 2.0 × 2.0 = 4x normal`
- Distribution at €100,000: `base × 1.0 × 0.3 × 0.3 = 0.09x normal`

**Stop losses** — width adjusted by phase:
- Accumulation: 1.5x wider (expect volatility, don't get shaken out)
- Distribution: 0.5x tighter (protect gains)
- Floor clamp: stop loss never goes below 95% of golden rule floor (€58,100)

**Emergency sell blocker** — if price is above golden rule floor estimate, emergency sells are blocked even at critical risk levels. The pattern says it recovers.

**Can-sell gate** — during bear capitulation (sell_reluctance=3.0), selling requires 15% profit margin. During distribution (sell_reluctance=0.3), selling is easy.

**Risk block override** — normally, CRITICAL risk blocks all trading. But during bear capitulation, small DCA buys are allowed through. This is the "be greedy when others are fearful" rule.

### Modified File: `trading_bot.py`

Signal evaluation now combines technicals with cycle phase:

**RSI thresholds shift by phase:**
- Accumulation: buy when RSI < 38 (normally 30) — more eager
- Distribution: buy when RSI < 22 (normally 30) — much harder to trigger

**Signal scoring includes cycle:**
- Accumulation score ≥ 70: adds 1.5 buy signals (strong cycle bias)
- Accumulation score ≤ 15: adds 1.5 sell signals

**Decision thresholds shift by phase:**
- Accumulation: buy needs 1.5 signals (easy), sell needs 3.5 (hard)
- Distribution: buy needs 3.5 (hard), sell needs 1.5 (easy)

**Sell sizing by phase:**
- Distribution: sell 25% of position
- Early bear: sell 10%
- Accumulation: sell 5% maximum

### Modified File: `main.py`

Initializes `CycleAnalyzer` and passes it to `RiskManager`, which flows into `TradingBot`. The cycle summary is logged at startup.

## Where We Are Right Now (February 2026)

```
Cycle 4 | Day ~657 of ~1460 | 45% complete
Phase: DISTRIBUTION
Halving: April 20, 2024 | Next: ~April 2028
ATH: €107,662 (Oct 6, 2025)
Estimated floor: €58,100 (golden rule) 
Estimated ceiling: €123,811
Position multiplier: 0.3x | DCA: 0.3x

Translation: The bot should be in defensive mode right now.
Minimal new buying, tight stops, willing to take profits.
```

## How This Plays Out Across a Full Cycle

### Scenario: Full cycle from here to 2029

**Now → Mid 2026 (Distribution):**
Bot runs at 0.3x capacity. Takes profits on rallies. Tight stops. Accumulation score ~20-35.

**Late 2026 → Late 2027 (Bear):**
Price drops 40-80% from ATH. Bot detects BEAR_EARLY then BEAR_CAPITULATION. Position multiplier ramps to 2.0x. DCA intensifies. Emergency sells blocked by golden rule floor. Accumulation score hits 70-90.

**Early 2028 (Pre-Halving):**
Bot detects PRE_HALVING phase 6 months before April 2028 halving. 1.3x buying with strong accumulation bias. Sell reluctance high.

**April 2028 → October 2029 (Post-Halving → Growth → Euphoria):**
Maximum accumulated position from the bear now rides the new cycle. Position sizing scales from 1.5x (post-halving) down to 0.5x (euphoria) as we approach the ATH window at ~500 days.

**The key insight:** By the time the next ATH window opens (late 2029), the bot has been accumulating for 2+ years at deeply discounted prices with cycle-appropriate position sizing.

## Configuration

The cycle parameters are defined as constants in `cycle_analyzer.py` (not config.py) because they're derived from historical analysis and shouldn't be casually changed. The phase adjustment tables are easy to tune if you want to modify aggressiveness.

To change the current cycle number (when cycle 5 starts):
```python
cycle_analyzer = CycleAnalyzer(current_cycle=5)
```

You'll need to update the `HALVINGS` and `CYCLE_HISTORY` lists with cycle 5 data when the next halving occurs.
