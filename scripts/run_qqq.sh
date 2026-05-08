#!/usr/bin/env bash
set -euo pipefail
OPTION_DATA_DIR="${OPTION_DATA_DIR:-/path/to/licensed/qqq/data}" OPTION_TICKER=QQQ OPTION_SECID=107899 OPTION_CALIBRATION_CONFIG=configs/qqq_config.json python src/run_option_book_var.py
