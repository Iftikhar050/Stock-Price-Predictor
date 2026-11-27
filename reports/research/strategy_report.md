# PSX strategy research: findings and selection

Run: 25 Nov 2026 · Universe: the app's 105 KSE-100 stocks, plus a point-in-time check · Out-of-sample period 2015-01-01 to 2026-11-20 · Holdout 2024-11-21 to 2026-11-20, run once.
Reproduce: `python scripts/run_strategy_research.py --reveal-holdout` (settings in `config/research.yaml`, seed 42). Dashboard: `frontend/results_dashboard.html` reads `frontend/results.json`.

## Bottom line

**No strategy met the pre-set bar for a robust, profitable edge, so none is recommended for the live app.**

The best candidate is a simple monthly stock ranking:
- **What it holds:** the top 20 liquid stocks by 6-month momentum, closeness to the 52-week high and low volatility.
- **Turnover control:** a holding is kept while it stays in the top 40.
- **Why it isn't recommended:** it beat the fair benchmarks over 2015-2026, but not by enough to rule out luck after 30 variants were tried. In the untouched holdout it matched the KSE-100 rather than beat it.

The current daily PSO and MEBL models should not be used for trading. PSO loses money after costs, and MEBL earns well under buy-and-hold.

## Key results (2015-2026, after costs, before tax)

| Strategy | CAGR | Sharpe | Max drawdown | Holdout CAGR (2025-26) |
|---|---|---|---|---|
| Factor ranking, top 20, monthly, buffer (best candidate) | 21.3% | 1.13 | -47.6% | 32.3% |
| Same, point-in-time universe (survivorship check) | 22.5% | 1.11 | -44.9% | 48.5% |
| Equal-weight portfolio of the same stocks (fair benchmark) | 16.8% | 0.81 | -58.4% | 30.0% |
| KSE-100 price index (excludes dividends) | 15.5% | 0.87 | -48.5% | 35.2% |
| Random picks, same turnover (median of 20) | 14.6% | 0.69 | -56.0% | 23.2% |
| ML ranking, LightGBM, 20-day, top 20 | 16.5% | 0.77 | -61.7% | 37.2% |
| Confidence-gated direction model, 20-day, p>0.55 | 10.6% | 0.56 | -65.7% | 29.4% |
| Current MEBL model, trade every signal | 14.9% | 0.73 | -44.3% | 24.4% |
| Buy and hold MEBL | 37.9% | 1.26 | -46.2% | 64.8% |
| Current PSO model, trade every signal | -17.3% | -0.64 | -90.5% | -19.6% |
| Buy and hold PSO | 8.8% | 0.42 | -56.2% | 19.1% |

All 37 strategies are in `results.json`, and the 30 counted trials are in `reports/research/trials.csv`.

## Why nothing was selected

The selection rule was fixed before any results were seen and applied to the 2015-2024 development period only. A strategy needed all of the following:
- a **deflated Sharpe ratio of at least 0.95**, which corrects for all 30 variants tried;
- a better Sharpe and CAGR than the equal-weight portfolio;
- a CAGR above the 95th percentile of 20 random portfolios with the same turnover.

The best candidate passed the second and third tests:
- Development Sharpe 1.07 vs 0.72 for equal-weight.
- CAGR 18.9% vs 15.5% for the random 95th percentile.

Its deflated Sharpe was **0.61**, so it failed the first. Ten years of noisy PSX returns and 30 tries are not enough to call a Sharpe gap of about 0.35 real.

The edge is also uneven across periods:
- **Sharpe by period:** 0.98 in 2015-17, 0.32 in 2018-20, 2.0 in 2021-24, 1.35 in the holdout.
- **By market trend:** +45% a year while the KSE-100 was above its 200-day average, and -17% a year while it was below. Equal-weight returned +50% and -33% over the same stretches.

## What the data says about prediction

- **Point forecasts are worse than "no change".** Average error was 1.71% vs 1.66% at 1 day, and 9.3% vs 8.8% at 20 days. This holds for the current models as well as the new ones.
- **Direction is barely better than always guessing "up".** The models were right 51-53% of the time, against 49-52% for always "up", across 1-20 days. Per-stock accuracy spreads only as much as chance would produce.
- **Confidence is overstated.** When the model says a 73% chance of a rise over 5 days, the stock rises 53% of the time. Filtering on confidence therefore lowered returns.
- **What works is ranking stocks against each other over weeks.** Momentum, nearness to the 52-week high and low volatility each held up in every sub-period. These are well-known effects internationally.
- **Machine learning added nothing over the three-signal rule.** Results were the same or worse, with deeper drawdowns.
- **A market-trend filter traded return for little safety.** Holding 50% when the KSE-100 was below its 200-day average cut CAGR by 4.6 points and drawdown by only 4 points.

## Method (short)

- **Data:** Yahoo Finance prices, dividend- and split-adjusted, with dividends included.
  - Days not in the exchange calendar were dropped (193 days, calendar taken from the KSE-100 file).
  - 2,686 impossible prints were neutralised and made untradeable. These are moves beyond any circuit limit, such as splits that weren't adjusted and spikes that reverse.
  - Zero-volume days and days locked at the circuit limit were untradeable.
- **Execution:**
  - Signals use the close of day t and fill at the close of day t+1.
  - About 0.6% round trip for liquid names: 0.15% commission + 13% sales tax on it, fees, CVT and tiered slippage.
  - A day's trade in one stock is capped at 10% of its 20-day median traded value, for a PKR 10m account.
  - T+2 settlement is modelled as no leverage. The strategies are long-only.
- **Models:**
  - LightGBM with fixed settings, refit quarterly on an expanding window.
  - Training rows whose label overlaps the test block were removed, plus a 5-day embargo.
  - Tests check that features don't change when future data is removed, that labels start at the fill, that there's no skill on noise, and that the backtest engine gives known answers.

## Assumptions to verify [NEEDS VERIFICATION]

- **CVT rate** on share purchases (0.02% assumed).
- **Current fee schedule:** exchange, NCCPL, CDC and SECP fees.
- **Circuit breaker band** after the planned move to 10%.
- **3 paisa minimum commission:** not modelled.

## Recommendation

1. **Leave the live models unchanged.**
2. **Show forecast ranges instead of one exact price.** Move size is predictable even though direction isn't, so the site should show a likely range rather than a precise price.
3. **Paper-trade the best candidate for 6-12 months with no changes**, then re-test it on that new data.
4. **Get cleaner inputs.** Source PSX-adjusted prices and the KSE-100 total-return index, and confirm costs with a broker.

**Rollback:** nothing in the live app changed. The research code is self-contained in `src/psx_predictor/research/`, `scripts/run_strategy_research.py` and `tests/test_research/`.
