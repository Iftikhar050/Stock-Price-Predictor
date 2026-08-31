# PSX Feature Coverage Validation Report
Generated: 2026-08-31 15:32

## PSO
- **Rows**: 4,846
- **Columns**: 364
- **Date Range**: 2008-01-01 -> 2026-08-27
- **Duplicate Dates**: 0
- **Columns with >20% Nulls**: 1

### Top Null Rate Columns
| Column | Null Rate |
|---|---|
| `adjusted_close` | 100.0% |

### Outlier Columns (|z| > 5 on >1% rows)
- `roe`: 3.5% of rows
- `free_cash_flow`: 1.3% of rows
- `operating_cash_flow`: 1.3% of rows
- `ebitda`: 3.5% of rows
- `global_oil_supply_shock_flag`: 1.5% of rows
- `event_sentiment_decay`: 1.1% of rows
- `year_end_flag`: 1.7% of rows
- `tax_deadline_flag`: 1.6% of rows
- `return_kurtosis_20d`: 1.2% of rows
- `panic_selling_proxy`: 3.3% of rows
- `bond_volatility_20d`: 1.7% of rows
- `eps_qoq_surprise`: 1.3% of rows

### PDF Group Coverage
| Group | Present | Total | Coverage | Missing |
|---|---|---|---|---|
| Company Fundamentals | 12 | 15 | 80.0% | `interest_bearing_debt`, `forward_pe`, `price_to_cash_flow` |
| Earnings-Related | 6 | 6 | 100.0% |  |
| Pakistan Macro - Interest | 15 | 15 | 100.0% |  |
| Pakistan Macro - FX | 4 | 4 | 100.0% |  |
| Pakistan Macro - CPI | 3 | 3 | 100.0% |  |
| Global Markets | 13 | 13 | 100.0% |  |
| Technical - OHLCV | 12 | 12 | 100.0% |  |
| Technical - Indicators | 10 | 10 | 100.0% |  |
| Technical - Returns | 13 | 13 | 100.0% |  |
| Technical - Volatility | 9 | 9 | 100.0% |  |
| Market Breadth | 8 | 8 | 100.0% |  |
| Institutional Flows | 8 | 8 | 100.0% |  |
| Sentiment | 6 | 9 | 66.7% | `sent_lag_1`, `sent_lag_2`, `sent_lag_3` |
| Corporate Events | 10 | 10 | 100.0% |  |
| Calendar Events | 9 | 9 | 100.0% |  |
| Banking Metrics (MEBL) | 9 | 9 | 100.0% |  |
| Energy Metrics (PSO) | 3 | 3 | 100.0% |  |
| IMF Indicators | 5 | 5 | 100.0% |  |
| Valuation Metrics | 7 | 7 | 100.0% |  |

## MEBL
- **Rows**: 4,846
- **Columns**: 365
- **Date Range**: 2008-01-01 -> 2026-08-27
- **Duplicate Dates**: 0
- **Columns with >20% Nulls**: 1

### Top Null Rate Columns
| Column | Null Rate |
|---|---|
| `adjusted_close` | 100.0% |

### Outlier Columns (|z| > 5 on >1% rows)
- `sma_50_dist`: 1.1% of rows
- `debt_to_equity`: 2.7% of rows
- `free_cash_flow`: 1.3% of rows
- `operating_cash_flow`: 1.3% of rows
- `total_debt`: 2.7% of rows
- `global_oil_supply_shock_flag`: 1.5% of rows
- `event_sentiment_decay`: 1.4% of rows
- `mpc_date_flag`: 1.2% of rows
- `budget_season_flag`: 3.8% of rows
- `downside_beta_252d`: 1.1% of rows
- `return_kurtosis_20d`: 1.5% of rows
- `panic_selling_proxy`: 3.5% of rows
- `bond_volatility_20d`: 1.7% of rows
- `eps_qoq_surprise`: 1.3% of rows

### PDF Group Coverage
| Group | Present | Total | Coverage | Missing |
|---|---|---|---|---|
| Company Fundamentals | 12 | 15 | 80.0% | `interest_bearing_debt`, `forward_pe`, `price_to_cash_flow` |
| Earnings-Related | 6 | 6 | 100.0% |  |
| Pakistan Macro - Interest | 15 | 15 | 100.0% |  |
| Pakistan Macro - FX | 4 | 4 | 100.0% |  |
| Pakistan Macro - CPI | 3 | 3 | 100.0% |  |
| Global Markets | 13 | 13 | 100.0% |  |
| Technical - OHLCV | 12 | 12 | 100.0% |  |
| Technical - Indicators | 10 | 10 | 100.0% |  |
| Technical - Returns | 13 | 13 | 100.0% |  |
| Technical - Volatility | 9 | 9 | 100.0% |  |
| Market Breadth | 8 | 8 | 100.0% |  |
| Institutional Flows | 8 | 8 | 100.0% |  |
| Sentiment | 9 | 9 | 100.0% |  |
| Corporate Events | 10 | 10 | 100.0% |  |
| Calendar Events | 7 | 9 | 77.8% | `sbp_mpc_date_flag`, `tax_deadline_flag` |
| Banking Metrics (MEBL) | 9 | 9 | 100.0% |  |
| Energy Metrics (PSO) | 3 | 3 | 100.0% |  |
| IMF Indicators | 5 | 5 | 100.0% |  |
| Valuation Metrics | 7 | 7 | 100.0% |  |
