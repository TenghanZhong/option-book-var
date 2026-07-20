# SPX aggregate results

This directory contains compact derived SPX outputs used for table-level verification and diagnostics.

The following date-level panels are intentionally omitted from the repository:

```text
book_var_results_v25_paper_ready.csv
book_var_rolling_v25_paper_ready.csv
book_var_marking_intersection_dates_v25.csv
book_var_marking_diag_dates.csv
```

They can be regenerated from licensed raw data by running `src/run_option_book_var.py` with `configs/spx_config.json`. See `docs/omitted_large_files.md` and `docs/artifact_reproducibility.md`.
