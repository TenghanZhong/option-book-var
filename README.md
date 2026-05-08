# Marking-Aware Sequential VaR Recalibration for Standardized Option Books

This repository contains an anonymized replication artifact for an option-book Value-at-Risk (VaR) recalibration study. It is structured for blind review and for private GitHub upload before generating an anonymous review link. Raw OptionMetrics IvyDB data are not included.

This GitHub-light version keeps code, configuration files, aggregate result tables, paper figures, synthetic sample data, and documentation. Large date-level result panels are omitted to avoid GitHub browser-upload limits and to reduce redistribution risk for licensed-data derivatives.

## What is included

- `src/run_option_book_var.py`: main SPX/QQQ option-book VaR pipeline.
- `src/make_tables.py`: table-generation script from aggregate result CSVs.
- `src/validate_no_leakage.py`: forecast-time feature-leakage audit.
- `src/make_figures.py`: figure-generation script for use with the full date-level outputs.
- `configs/`: anonymous SPX and QQQ configuration templates.
- `results/spx` and `results/qqq`: aggregate review outputs only.
- `ablation_results/`: compact calibration-memory ablation outputs.
- `figures/`: generated paper figures.
- `sample_data/`: synthetic schema sample, not empirical data.
- `docs/`: data schema, reproducibility guide, licensed-data notice, and omitted-file list.

## What is omitted

Large date-level result files are excluded:

```text
ablation_results/book_var_calibration_memory_ablation_detail.csv
results/*/book_var_results_v25_paper_ready.csv
results/*/book_var_rolling_v25_paper_ready.csv
results/*/book_var_marking_intersection_dates_v25.csv
results/*/book_var_marking_diag_dates.csv
```

See `docs/omitted_large_files.md` for sizes and rationale.

## Reproduce tables and leakage audit from included files

These commands do not require licensed raw data and work with this GitHub-light package:

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

## Figures

Paper figures are already included under `figures/spx` and `figures/qqq`.

Regenerating all figures from scratch requires the omitted date-level files or a full rerun with licensed data, because rolling and crisis plots are built from date-level outputs.

## Re-run the full empirical pipeline with licensed data

Researchers with licensed OptionMetrics IvyDB US access can reproduce the full pipeline by providing local raw files that match `docs/data_schema.md`.

```bash
OPTION_DATA_DIR=/path/to/licensed/spx/data \
OPTION_TICKER=SPX \
OPTION_SECID=108105 \
OPTION_CALIBRATION_CONFIG=configs/spx_config.json \
python src/run_option_book_var.py

OPTION_DATA_DIR=/path/to/licensed/qqq/data \
OPTION_TICKER=QQQ \
OPTION_SECID=107899 \
OPTION_CALIBRATION_CONFIG=configs/qqq_config.json \
python src/run_option_book_var.py
```

The pipeline writes paper-facing outputs to the configured results directory. The omitted date-level files can be regenerated from licensed raw data.
