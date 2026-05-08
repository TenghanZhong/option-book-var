#!/usr/bin/env python3
"""Generate paper-facing CSV and LaTeX tables from derived result CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

BOOK_ORDER = [
    "atm_straddle_30d",
    "risk_reversal_25d_30d",
    "short_put_spread_25delta_10delta_30d",
]
BOOK_LABEL = {
    "atm_straddle_30d": "ATM straddle",
    "risk_reversal_25d_30d": "25d risk reversal",
    "short_put_spread_25delta_10delta_30d": "25d/10d put spread",
    "pooled": "Pooled",
}
METHOD_ORDER = [
    "Historical VaR",
    "EWMA Historical VaR",
    "CAViaR",
    "GARCH-t VaR",
    "LightGBM Quantile",
    "LightGBM + Calibration",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _main_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["experiment_group"].astype(str).eq("main")].copy()
    if out.empty:
        out = df[df["experiment_id"].astype(str).str.startswith("main_lightgbm_robust_all")].copy()
    out["_book_order"] = out["book_type"].map({b: i for i, b in enumerate(BOOK_ORDER)}).fillna(999)
    return out.sort_values("_book_order").drop(columns=["_book_order"])


def _fmt(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def build_main_results(root: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for market in ["spx", "qqq"]:
        s = _main_summary(_read(root / "results" / market / "book_var_summary_v25_paper_ready.csv"))
        for _, r in s.iterrows():
            rows.append({
                "Market": market.upper(),
                "Book": BOOK_LABEL.get(str(r["book_type"]), str(r["book_type"])),
                "n": int(r["n_backtest_days"]),
                "Reference exceedance": r["empirical_exceedance_rate_base"],
                "Recalibrated exceedance": r["empirical_exceedance_rate_conf"],
                "Reference avg violation": r["avg_violation_base"],
                "Recalibrated avg violation": r["avg_violation_conf"],
                "Reference pinball": r["avg_pinball_base"],
                "Recalibrated pinball": r["avg_pinball_conf"],
                "Reference max roll50": r["max_roll50_exceed_base"],
                "Recalibrated max roll50": r["max_roll50_exceed_conf"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table_main_results.csv", index=False)
    return out


def build_baseline_pooled(root: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    for market in ["spx", "qqq"]:
        b = _read(root / "results" / market / "book_var_baseline_comparison_v25.csv")
        b = b[b["book_type"].astype(str).eq("pooled")].copy()
        b["_method_order"] = b["method"].map(order).fillna(999)
        b = b.sort_values("_method_order")
        for _, r in b.iterrows():
            rows.append({
                "Market": market.upper(),
                "Method": r["method"],
                "Exceedance": r["empirical_exceedance_rate"],
                "Average violation": r["avg_violation"],
                "Pinball loss": r["avg_pinball_loss"],
                "Average VaR": r["avg_threshold"],
                "Max 50-day exceedance": r["max_roll50_exceedance"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table_baseline_pooled.csv", index=False)
    return out


def build_backtest_main(root: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for market in ["spx", "qqq"]:
        t = _read(root / "results" / market / "book_var_backtest_tests_v25.csv")
        t = t[
            t["experiment_group"].astype(str).eq("main")
            & t["method_suffix"].astype(str).eq("conf")
            & t["book_type"].astype(str).isin(BOOK_ORDER)
        ].copy()
        t["_book_order"] = t["book_type"].map({b: i for i, b in enumerate(BOOK_ORDER)}).fillna(999)
        t = t.sort_values("_book_order")
        for _, r in t.iterrows():
            rows.append({
                "Market": market.upper(),
                "Book": BOOK_LABEL.get(str(r["book_type"]), str(r["book_type"])),
                "n": int(r["n"]),
                "Exceedance": r["exceedance_rate"],
                "p_UC": r["p_uc"],
                "p_IND": r["p_ind"],
                "p_CC": r["p_cc"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table_backtest_main_recalibrated.csv", index=False)
    return out


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    display = df.copy()
    for c in display.columns:
        if c == "n":
            continue
        if pd.api.types.is_float_dtype(display[c]):
            display[c] = display[c].map(lambda x: _fmt(x, 3))
    latex = display.to_latex(index=False, escape=True, caption=caption, label=label)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out if args.out is not None else root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_df = build_main_results(root, out_dir)
    baseline_df = build_baseline_pooled(root, out_dir)
    backtest_df = build_backtest_main(root, out_dir)

    write_latex_table(main_df, out_dir / "table_main_results.tex", "Main out-of-sample VaR results.", "tab:main_results_generated")
    write_latex_table(baseline_df, out_dir / "table_baseline_pooled.tex", "Aggregate comparison with classical VaR baselines.", "tab:baseline_pooled_generated")
    write_latex_table(backtest_df, out_dir / "table_backtest_main_recalibrated.tex", "Backtest diagnostics for main recalibrated VaR.", "tab:backtest_main_generated")

    print(f"Wrote generated tables to {out_dir}")


if __name__ == "__main__":
    main()
