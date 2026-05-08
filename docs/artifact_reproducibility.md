# Artifact reproducibility guide

## Path layout

The code is repository-relative. Source code lives in `src/`, derived results in `results/spx` and `results/qqq`, and figures in `figures/spx` or `figures/qqq`.

## Fast reproduction from derived result CSVs

This path does not require licensed raw data.

```bash
python src/validate_no_leakage.py --root .
python src/make_tables.py --root . --out tables
OPTION_BOOK_VAR_ASSET=SPX python src/make_figures.py
OPTION_BOOK_VAR_ASSET=QQQ python src/make_figures.py
```

Expected outputs:

```text
tables/table_main_results.csv
tables/table_baseline_pooled.csv
tables/table_backtest_main_recalibrated.csv
figures/spx/*.png
figures/qqq/*.png
validation/no_leakage_report.csv
```

## Full reproduction with licensed data

This path requires OptionMetrics IvyDB US access and local files matching `docs/data_schema.md`.

```bash
OPTION_DATA_DIR=/path/to/licensed/spx/data OPTION_TICKER=SPX OPTION_SECID=108105 OPTION_CALIBRATION_CONFIG=configs/spx_config.json python src/run_option_book_var.py
OPTION_DATA_DIR=/path/to/licensed/qqq/data OPTION_TICKER=QQQ OPTION_SECID=107899 OPTION_CALIBRATION_CONFIG=configs/qqq_config.json python src/run_option_book_var.py
```

The pipeline writes paper-facing outputs to `results/spx` or `results/qqq`, unless `OPTION_RESULTS_DIR` or the config file overrides the output directory.

## Randomness and model settings

The main pipeline fixes NumPy random seed 42 and uses fixed learner hyperparameters across books and markets. The LightGBM reference uses the quantile objective at the 0.90 upper-tail level. Classical baselines include historical VaR, EWMA historical VaR, CAViaR, and GARCH-t VaR.

## Known limitation

The provided synthetic sample data are schema examples only. They are not long enough to reproduce rolling 252-day empirical results.
