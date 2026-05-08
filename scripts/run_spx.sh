#!/usr/bin/env bash
set -euo pipefail
OPTION_DATA_DIR="${OPTION_DATA_DIR:-/path/to/licensed/spx/data}" OPTION_TICKER=SPX OPTION_SECID=108105 OPTION_CALIBRATION_CONFIG=configs/spx_config.json python src/run_option_book_var.py
