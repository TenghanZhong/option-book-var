# Artifact reproducibility guide

## Package type

This is the GitHub-light artifact. It is designed for blind-review upload and fast inspection. It does not include raw OptionMetrics data or large date-level derived panels.

## Path layout

```text
src/                  source code
configs/              anonymous configuration templates
results/spx/          SPX aggregate derived outputs
results/qqq/          QQQ aggregate derived outputs
ablation_results/     compact calibration-memory ablation outputs
figures/              generated paper figures
sample_data/          synthetic schema sample
docs/                 schema, license, reproducibility notes
```

## Fast reproduction from included files

This path does not require licensed raw data.

```bash
python src/validate_no_leakage.py --root .
python src/make_tables.py --root . --out tables
```

Expected outputs:

```text
validation/no_leakage_report.csv
tables/table_main_results.csv
tables/table_baseline_pooled.csv
tables/table_backtest_main_recalibrated.csv
```

## Figure reproduction

The generated figure PNGs are included. Regenerating all figures from scratch requires the omitted date-level result panels listed in `docs/omitted_large_files.md`, because rolling and crisis figures depend on date-level outputs.

With a full result directory, figures can be regenerated with:

```bash
OPTION_BOOK_VAR_ASSET=SPX python src/make_figures.py
OPTION_BOOK_VAR_ASSET=QQQ python src/make_figures.py
```

## Full reproduction with licensed data

Full empirical reproduction requires OptionMetrics IvyDB US access and local files matching `docs/data_schema.md`.

```bash
OPTION_DATA_DIR=/path/to/licensed/spx/data OPTION_TICKER=SPX OPTION_SECID=108105 OPTION_CALIBRATION_CONFIG=configs/spx_config.json python src/run_option_book_var.py
OPTION_DATA_DIR=/path/to/licensed/qqq/data OPTION_TICKER=QQQ OPTION_SECID=107899 OPTION_CALIBRATION_CONFIG=configs/qqq_config.json python src/run_option_book_var.py
```

The pipeline fixes NumPy random seed 42 and uses fixed learner settings across books and markets. The reference model is a LightGBM quantile model at the 0.90 upper-tail level. Classical baselines include historical VaR, EWMA historical VaR, CAViaR, and GARCH-t VaR.

## Synthetic sample

The synthetic sample data are schema examples only. They are not long enough to reproduce the rolling 252-day empirical results.
