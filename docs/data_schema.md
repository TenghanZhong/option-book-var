# Data schema for full-pipeline reproduction

The empirical results require licensed OptionMetrics IvyDB US option-chain data. Raw option-chain data are not redistributed in this artifact.

## Option-chain files

Default file pattern:

```text
data_{year}.parquet
```

The pipeline can also read CSV files if the configuration uses `data_{year}.csv`.

Expected option-chain fields, using case-insensitive matching after lower-casing column names:

```text
secid, date, symbol, exdate, last_date, cp_flag,
strike_price, best_bid, best_offer, volume, open_interest,
impl_volatility, delta, gamma, vega, theta, optionid,
contract_size, ss_flag, forward_price, expiry_indicator,
root, suffix, ticker, index_flag, issuer,
div_convention, exercise_style, am_settlement, am_set_flag
```

Minimum required fields for the main pipeline are:

```text
secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
volume, open_interest, impl_volatility, delta, vega, optionid,
spot merge key fields, and either secid/ticker/root for underlying filtering
```

## Underlying prices

Default file:

```text
security_prices.parquet
```

Required columns:

```text
secid, date, close
```

The pipeline renames `close` to `spot` after filtering by `secid`.

## Zero-coupon yield curve

Default file:

```text
zero_coupon_yield.parquet
```

Required columns:

```text
date, days, rate
```

The pipeline merges the nearest available maturity to each option's DTE.

## Dividend yield file

Default file:

```text
index_dividend_yield.parquet
```

Required columns:

```text
secid, date, expiration, rate
```

If the file uses `exdate` instead of `expiration`, the pipeline renames it. If an ETF dividend-yield file has no `secid`, set `OPTION_SECID` so the pipeline can attach it.

## VIX and VIX3M/VXV files

Defaults:

```text
VIXCLS.parquet
VXVCLS.parquet
```

Expected columns:

```text
date or observation_date, value column
```

The value column is renamed to `vix` or `vxv`.

## Forecast-time information rule

The model feature matrix must not contain current realized date-`t` to date-`t+1` marking diagnostics. The allowed marking features are lagged counts and lagged rolling rates only. Run:

```bash
python src/validate_no_leakage.py --root .
```
