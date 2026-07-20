# Artifact reproducibility guide

## Artifact scope

This repository is the public replication artifact for the accepted IEEE CIFEr 2026 paper *Marking-Aware Sequential VaR Recalibration for Standardized Option Books*.

It includes the implementation, configuration files, aggregate derived outputs, compact robustness outputs, generated figures and tables, and a synthetic schema sample. It does not include raw OptionMetrics data or large date-level derived panels.

## Repository layout

```text
src/                  source code
configs/              SPX, QQQ, and synthetic configuration files
results/spx/          SPX aggregate derived outputs
results/qqq/          QQQ aggregate derived outputs
ablation_results/     compact calibration-memory ablation outputs
tables/               generated tables and alpha-robustness outputs
figures/              generated paper figures
sample_data/          synthetic schema sample
docs/                 schema, data-rights, and reproducibility notes
```

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Fast verification from included files

This path does not require licensed raw data.

```bash
python src/validate_no_leakage.py --root .
python src/make_tables.py --root . --out tables
```

Expected generated outputs include:

```text
validation/no_leakage_report.csv
tables/table_main_results.csv
tables/table_baseline_pooled.csv
tables/table_backtest_main_recalibrated.csv
```

The alpha-0.05 files under `tables/` are compact outputs from a separate full rerun. They are retained for auditability and are not regenerated from the included alpha-0.10 aggregate files.

## Figure reproduction

Generated PNG files are included. Regenerating all figures from scratch requires the omitted date-level result panels listed in `docs/omitted_large_files.md`, because rolling and crisis diagnostics depend on date-level outputs.

With a full result directory, run:

```bash
OPTION_BOOK_VAR_ASSET=SPX python src/make_figures.py
OPTION_BOOK_VAR_ASSET=QQQ python src/make_figures.py
```

## Full reproduction with licensed data

Full empirical reproduction requires OptionMetrics IvyDB US access and local files matching `docs/data_schema.md`.

```bash
OPTION_DATA_DIR=/path/to/licensed/spx/data \
OPTION_TICKER=SPX \
OPTION_SECID=108105 \
OPTION_CALIBRATION_CONFIG=configs/spx_config.json \
OPTION_RESULTS_SUBDIR=main_results_spx_alpha_0p10_no_leakage \
python src/run_option_book_var.py

OPTION_DATA_DIR=/path/to/licensed/qqq/data \
OPTION_TICKER=QQQ \
OPTION_SECID=107899 \
OPTION_CALIBRATION_CONFIG=configs/qqq_config.json \
OPTION_RESULTS_SUBDIR=main_results_qqq_alpha_0p10_no_leakage \
python src/run_option_book_var.py
```

The main specification fixes NumPy seed 42, uses a 252-day rolling learner window, a 126-residual recalibration window, target exceedance probability `ALPHA = 0.10`, and the same learner settings across books and markets. The classical baselines are historical VaR, EWMA historical VaR, CAViaR, and Student-t GARCH VaR.

For the alpha-0.05 robustness rerun, change only `ALPHA` to `0.05`, keep the rolling windows unchanged, and use a separate result directory.

## Synthetic sample

The files in `sample_data/synthetic_option_chain_sample/` document the expected schema only. They are not long enough to reproduce the rolling 252-day empirical results.

## Data rights

Raw OptionMetrics data are licensed and are not redistributed. The MIT License applies to the source code, not to third-party datasets. See `docs/licensed_data_notice.md`.
