#!/usr/bin/env python3
"""Validate that realized current t->t+1 marking diagnostics are not in X_t."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

FORBIDDEN = {
    "n_option_mark_exact_t",
    "n_option_mark_contract_t",
    "n_option_mark_interp_t",
    "n_option_mark_nearest_t",
    "n_option_mark_fallback_t",
}


def _display_path(path: Path, root: Path | None = None) -> str:
    """Return a stable repository-relative path for reproducible reports."""
    try:
        if root is not None:
            return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        pass
    return str(path)


def validate_one(results_dir: Path, root: Path | None = None) -> dict:
    audit_path = results_dir / "book_var_feature_columns_no_leakage.csv"
    display_dir = _display_path(results_dir, root)
    if not audit_path.exists():
        return {"results_dir": display_dir, "status": "missing_audit", "forbidden_count": None}
    audit = pd.read_csv(audit_path)
    if "is_realized_current_marking_diagnostic" in audit.columns:
        forbidden_count = int(pd.to_numeric(audit["is_realized_current_marking_diagnostic"], errors="coerce").fillna(0).sum())
    elif "feature_col" in audit.columns:
        forbidden_count = int(audit["feature_col"].astype(str).isin(FORBIDDEN).sum())
    elif "feature" in audit.columns:
        forbidden_count = int(audit["feature"].astype(str).isin(FORBIDDEN).sum())
    else:
        return {"results_dir": display_dir, "status": "unreadable_audit_schema", "forbidden_count": None}
    return {
        "results_dir": display_dir,
        "status": "pass" if forbidden_count == 0 else "fail",
        "forbidden_count": forbidden_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", nargs="*", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result_dirs = args.results or [root / "results" / "spx", root / "results" / "qqq"]
    rows = [validate_one(Path(p), root=root) for p in result_dirs]
    report = pd.DataFrame(rows)
    out_dir = root / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "no_leakage_report.csv"
    report.to_csv(out_path, index=False)
    print(report.to_string(index=False))
    print(f"Wrote {out_path}")
    failures = report[report["status"] != "pass"]
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
