# Marking-Aware Sequential VaR Recalibration for Standardized Option Books

Code and aggregate replication artifact for the accepted IEEE CIFEr 2026 paper:

**Tenghan Zhong and Keyuan Wu, “Marking-Aware Sequential VaR Recalibration for Standardized Option Books.”**

The study evaluates one-day VaR for standardized SPX and QQQ option books under explicit book-construction, next-day marking, loss-normalization, and forecast-time information rules. A rolling quantile learner produces the reference threshold, and a leakage-safe residual layer recalibrates that threshold using only previously realized forecast residuals.

Raw OptionMetrics IvyDB US data are licensed and are not redistributed.

## Repository contents

- `src/run_option_book_var.py`: full SPX/QQQ option-book VaR pipeline.
- `src/make_tables.py`: paper-table generation from included aggregate CSV files.
- `src/validate_no_leakage.py`: audit of the forecast-time feature set.
- `src/make_figures.py`: figure generation when date-level outputs are available.
- `configs/`: SPX, QQQ, and synthetic-sample configuration files.
- `results/spx/` and `results/qqq/`: aggregate empirical outputs.
- `ablation_results/`: calibration-memory ablation outputs.
- `tables/`: generated paper tables and target-probability robustness outputs.
- `figures/`: generated SPX and QQQ figures.
- `sample_data/`: synthetic files that document the expected input schema.
- `docs/`: data schema, reproducibility instructions, licensed-data notice, and omitted-file inventory.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce included tables and run the leakage audit

These commands use only files included in the repository and do not require licensed raw data:

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

The compact alpha-0.05 files under `tables/` come from a separate full rerun. They are included to audit the deeper-tail robustness results and are not regenerated from the alpha-0.10 aggregate files by `src/make_tables.py`.

## Re-run the full empirical pipeline

Researchers with licensed OptionMetrics IvyDB US access can reproduce the full pipeline by preparing local files that match `docs/data_schema.md`.

### SPX

```bash
OPTION_DATA_DIR=/path/to/licensed/spx/data \
OPTION_TICKER=SPX \
OPTION_SECID=108105 \
OPTION_CALIBRATION_CONFIG=configs/spx_config.json \
OPTION_RESULTS_SUBDIR=main_results_spx_alpha_0p10_no_leakage \
python src/run_option_book_var.py
```

### QQQ

```bash
OPTION_DATA_DIR=/path/to/licensed/qqq/data \
OPTION_TICKER=QQQ \
OPTION_SECID=107899 \
OPTION_CALIBRATION_CONFIG=configs/qqq_config.json \
OPTION_RESULTS_SUBDIR=main_results_qqq_alpha_0p10_no_leakage \
python src/run_option_book_var.py
```

The canonical script uses the main target probability:

```python
ALPHA = 0.10
QUANTILE_LEVEL = 1.0 - ALPHA
TRAIN_WINDOW = 252
CALIB_WINDOW = 126
```

For the alpha-0.05 robustness rerun, change only `ALPHA` to `0.05`, keep the rolling windows unchanged, and write to a separate output directory.

## Reproducibility boundary

Included:

- implementation code and configuration files;
- aggregate SPX and QQQ outputs used for table-level verification;
- calibration-memory ablations;
- generated figures and tables;
- a synthetic input-schema sample;
- a feature-leakage audit.

Not included:

- raw licensed OptionMetrics data;
- large date-level derived panels listed in `docs/omitted_large_files.md`.

The omitted derived panels can be regenerated from licensed data with `src/run_option_book_var.py`. Figure reproduction from scratch requires those date-level panels because rolling and crisis diagnostics are date indexed.

## Citation

Citation metadata are provided in `CITATION.cff`. Until proceedings metadata and a DOI are available, cite the accepted paper by title and authors.

## License and data rights

The source code is released under the MIT License. This license does not grant redistribution rights for OptionMetrics data or other third-party datasets. See `docs/licensed_data_notice.md`.
