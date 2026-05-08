#!/usr/bin/env bash
set -euo pipefail

# Works with the GitHub-light artifact.
python src/validate_no_leakage.py --root .
python src/make_tables.py --root . --out tables

cat <<'MSG'
Tables and no-leakage audit regenerated.
Figure PNGs are already included.
Full figure regeneration requires omitted date-level panels or a full rerun with licensed raw data.
MSG
