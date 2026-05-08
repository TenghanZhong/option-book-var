from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# Config
# ============================================================
# No-leakage plotting version.
# Defaults are repository-relative:
#     results/spx for SPX
#     results/qqq for QQQ
# You can override with OPTION_BOOK_VAR_RESULTS_DIR and OPTION_BOOK_VAR_FIG_DIR.

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
ASSET_NAME = os.getenv("OPTION_BOOK_VAR_ASSET", "SPX").upper().strip()

_DEFAULT_RESULTS_SUBDIR = Path("results") / ("qqq" if ASSET_NAME == "QQQ" else "spx")
BASE_DIR = Path(os.getenv("OPTION_BOOK_VAR_RESULTS_DIR", str(REPO_ROOT / _DEFAULT_RESULTS_SUBDIR)))

FIG_DIR = Path(os.getenv("OPTION_BOOK_VAR_FIG_DIR", str(REPO_ROOT / "figures" / ASSET_NAME.lower())))
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = BASE_DIR / "book_var_summary_v25_paper_ready.csv"
RESULTS_FILE = BASE_DIR / "book_var_results_v25_paper_ready.csv"
ROLLING_FILE = BASE_DIR / "book_var_rolling_v25_paper_ready.csv"
CRISIS_FILE = BASE_DIR / "book_var_crisis_v25_paper_ready.csv"
YEARLY_FILE = BASE_DIR / "book_var_yearly_v25_paper_ready.csv"
BASELINE_FILE = BASE_DIR / "book_var_baseline_comparison_v25.csv"
FEATURE_AUDIT_FILE = BASE_DIR / "book_var_feature_columns_no_leakage.csv"

FORBIDDEN_CURRENT_MARKING_FEATURES = {
    "n_option_mark_exact_t",
    "n_option_mark_contract_t",
    "n_option_mark_interp_t",
    "n_option_mark_nearest_t",
    "n_option_mark_fallback_t",
}

ALPHA = 0.10

# These are resolved automatically after loading summary.
MAIN_EXPERIMENT_ID = None
STRICT_MARKING_EXPERIMENT_ID = None

CRISIS_START = pd.Timestamp("2020-02-20")
CRISIS_END = pd.Timestamp("2020-04-15")

BOOK_ORDER = [
    "atm_straddle_30d",
    "risk_reversal_25d_30d",
    "short_put_spread_25delta_10delta_30d",
]

BOOK_LABELS = {
    "atm_straddle_30d": "ATM Straddle",
    "risk_reversal_25d_30d": "25d Risk Reversal",
    "short_put_spread_25delta_10delta_30d": "25d/10d Short Put Spread",
    "pooled": "Pooled",
}

BOOK_LABELS_SHORT = {
    "atm_straddle_30d": "ATM",
    "risk_reversal_25d_30d": "25d RR",
    "short_put_spread_25delta_10delta_30d": "25d/10d Spread",
    "pooled": "Pooled",
}

# Extra-compact labels for single-column figures. The full names are already
# defined in captions and tables, so the axes should preserve data area.
BOOK_LABELS_SINGLE_COL = {
    "atm_straddle_30d": "ATM",
    "risk_reversal_25d_30d": "25d RR",
    "short_put_spread_25delta_10delta_30d": "25/10d\nPut Spr.",
    "pooled": "Pooled",
}

BOOK_LABELS_HEATMAP = {
    "atm_straddle_30d": "ATM",
    "risk_reversal_25d_30d": "25d RR",
    "short_put_spread_25delta_10delta_30d": "25/10d\nSpread",
    "pooled": "Pooled",
}

# Base experiment roots. The resolver below will also accept _quality_loose
# and _quality_strict_economic suffixes.
MAIN_EXPERIMENT_ROOT = "main_lightgbm_robust_all_floor_0p0"
STRICT_MARKING_EXPERIMENT_ROOT = "marking_lightgbm_strict_exact_contract_floor_0p0"

ROBUSTNESS_EXPERIMENT_ROOTS = [
    ("Main", "main_lightgbm_robust_all_floor_0p0"),
    ("GBR", "learner_gbr_robust_all_floor_0p0"),
    ("XGB", "learner_xgboost_robust_all_floor_0p0"),
    ("Strict", "marking_lightgbm_strict_exact_contract_floor_0p0"),
    ("No Floor", "floor_lightgbm_robust_all_none"),
]

METHOD_SPECS = [
    # Base here is the floor-applied LightGBM reference threshold q_hat_ref, not raw q_hat_base_raw.
    ("empirical_exceedance_rate_base", "LGBM ref.", "C0"),
    ("empirical_exceedance_rate_hist", "Historical", "C1"),
    ("empirical_exceedance_rate_conf", "Recalibrated", "C2"),
]

BASELINE_METHOD_ALIAS = {
    "Historical VaR": "Historical VaR",
    "EWMA Historical VaR": "EWMA Historical VaR",
    "CAViaR": "CAViaR",
    "GARCH-t VaR": "GARCH-t VaR",
    "LightGBM Quantile": "LightGBM Quantile",
    "LightGBM + Sequential Conformal": "LightGBM + Recalibrated",
    "LightGBM + Sequential Calibration": "LightGBM + Recalibrated",
    "LightGBM + Calibration": "LightGBM + Recalibrated",
    "LightGBM + Recalibration": "LightGBM + Recalibrated",
    "LightGBM + Calibrated": "LightGBM + Recalibrated",
    "LightGBM + Recalibrated": "LightGBM + Recalibrated",
}

BASELINE_METHOD_ORDER = [
    "Historical VaR",
    "EWMA Historical VaR",
    "CAViaR",
    "GARCH-t VaR",
    "LightGBM Quantile",
    "LightGBM + Recalibrated",
]

BASELINE_METHOD_LABELS = {
    "Historical VaR": "Hist.",
    "EWMA Historical VaR": "EWMA Hist.",
    "CAViaR": "CAViaR",
    "GARCH-t VaR": "GARCH-t",
    "LightGBM Quantile": "LightGBM",
    "LightGBM + Recalibrated": "LightGBM+Recal.",
}

BASELINE_COLORS = {
    "Historical VaR": "C1",
    "EWMA Historical VaR": "C3",
    "CAViaR": "C4",
    "GARCH-t VaR": "C5",
    "LightGBM Quantile": "C0",
    "LightGBM + Recalibrated": "C2",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    # Font sizes are chosen for the final IEEE display size:
    # single-column figures are about 3.5 in wide; two-column figures are about 6.6 in wide.
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# ============================================================
# Helpers
# ============================================================

def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def normalize_baseline_methods(baseline: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty or "method" not in baseline.columns:
        return baseline

    out = baseline.copy()
    out["method"] = out["method"].map(lambda x: BASELINE_METHOD_ALIAS.get(str(x), str(x)))
    return out


def load_csvs():
    summary = _read_csv_required(SUMMARY_FILE)
    results = _read_csv_required(RESULTS_FILE)
    rolling = _read_csv_required(ROLLING_FILE)
    crisis = _read_csv_required(CRISIS_FILE)
    yearly = _read_csv_required(YEARLY_FILE)

    baseline = pd.read_csv(BASELINE_FILE) if BASELINE_FILE.exists() else pd.DataFrame()
    baseline = normalize_baseline_methods(baseline)

    for df in [results, rolling]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    if "next_date" in results.columns:
        results["next_date"] = pd.to_datetime(results["next_date"])

    return summary, results, rolling, crisis, yearly, baseline


def audit_no_leakage_feature_file(required: bool = True):
    """Fail fast if the plotted result directory does not pass the no-leakage feature audit."""
    if not FEATURE_AUDIT_FILE.exists():
        msg = f"Missing no-leakage feature audit CSV: {FEATURE_AUDIT_FILE}"
        if required:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}")
        return

    audit = pd.read_csv(FEATURE_AUDIT_FILE)
    if "is_realized_current_marking_diagnostic" in audit.columns:
        forbidden_count = int(pd.to_numeric(
            audit["is_realized_current_marking_diagnostic"],
            errors="coerce",
        ).fillna(0).sum())
    elif "feature_col" in audit.columns:
        forbidden_count = int(audit["feature_col"].astype(str).isin(FORBIDDEN_CURRENT_MARKING_FEATURES).sum())
    elif "feature" in audit.columns:
        forbidden_count = int(audit["feature"].astype(str).isin(FORBIDDEN_CURRENT_MARKING_FEATURES).sum())
    else:
        raise ValueError(
            f"Cannot identify feature column in audit file: {FEATURE_AUDIT_FILE}. "
            f"Columns={list(audit.columns)}"
        )

    print(f"No-leakage audit file: {FEATURE_AUDIT_FILE}")
    print(f"Forbidden realized current marking features in X_t: {forbidden_count}")
    if forbidden_count != 0:
        raise ValueError(
            "This result directory still contains current t->t+1 marking diagnostics in X_t. "
            "Do not plot or use it for the no-leakage paper tables."
        )


def prettify_book(book_type: str) -> str:
    return BOOK_LABELS.get(str(book_type), str(book_type))


def prettify_book_short(book_type: str) -> str:
    return BOOK_LABELS_SHORT.get(str(book_type), str(book_type))


def prettify_book_single_col(book_type: str) -> str:
    return BOOK_LABELS_SINGLE_COL.get(str(book_type), str(book_type))


def prettify_book_heatmap(book_type: str) -> str:
    return BOOK_LABELS_HEATMAP.get(str(book_type), str(book_type))


def savefig(fig, filename: str):
    """Save without tight_layout.

    The individual plotting functions use explicit subplots_adjust calls. Calling
    tight_layout here would often pull outside legends and long labels back into
    the axes and recreate the overlap problem.
    """
    out = FIG_DIR / filename
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out}")


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def wilson_interval(k, n, z=1.96):
    if n <= 0 or pd.isna(n) or pd.isna(k):
        return np.nan, np.nan

    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = (
        z
        * np.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))
        / denom
    )
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


def experiment_candidates(root: str):
    return [
        f"{root}_quality_loose",
        f"{root}_quality_strict_economic",
        root,
    ]


def resolve_experiment_id(summary: pd.DataFrame, root: str, label: str, required: bool = True):
    available = sorted(summary["experiment_id"].astype(str).unique().tolist())

    for candidate in experiment_candidates(root):
        if candidate in available:
            return candidate

    # Fallback: accept any quality suffix generated by newer main code.
    prefix = f"{root}_quality_"
    matches = [x for x in available if x.startswith(prefix)]
    if matches:
        # Prefer quality_loose if present, otherwise deterministic first.
        for x in matches:
            if x.endswith("_quality_loose"):
                return x
        return sorted(matches)[0]

    if required:
        raise ValueError(
            f"No matching {label} experiment id found.\n"
            f"Root: {root}\n"
            f"Candidates: {experiment_candidates(root)}\n"
            f"Available IDs: {available}"
        )
    return None


def resolve_runtime_experiments(summary: pd.DataFrame):
    global MAIN_EXPERIMENT_ID, STRICT_MARKING_EXPERIMENT_ID

    MAIN_EXPERIMENT_ID = resolve_experiment_id(
        summary,
        MAIN_EXPERIMENT_ROOT,
        label="main",
        required=True,
    )

    STRICT_MARKING_EXPERIMENT_ID = resolve_experiment_id(
        summary,
        STRICT_MARKING_EXPERIMENT_ROOT,
        label="strict marking",
        required=False,
    )

    print(f"Using MAIN_EXPERIMENT_ID = {MAIN_EXPERIMENT_ID}")
    print(f"Using STRICT_MARKING_EXPERIMENT_ID = {STRICT_MARKING_EXPERIMENT_ID}")


def resolve_robustness_ids(summary: pd.DataFrame):
    rows = []
    for display_label, root in ROBUSTNESS_EXPERIMENT_ROOTS:
        eid = resolve_experiment_id(summary, root, label=display_label, required=False)
        if eid is not None:
            rows.append((display_label, eid))
        else:
            print(f"[SKIP] robustness experiment not found: {display_label} / {root}")
    return rows


def get_main_summary(summary: pd.DataFrame):
    df = summary.loc[summary["experiment_id"] == MAIN_EXPERIMENT_ID].copy()
    df["book_type"] = pd.Categorical(
        df["book_type"], categories=BOOK_ORDER, ordered=True
    )
    return df.sort_values("book_type")


def get_results_counts(results: pd.DataFrame):
    counts = (
        results.groupby(["experiment_id", "book_type"], as_index=False)
        .size()
        .rename(columns={"size": "n_obs"})
    )
    return counts


# ============================================================
# Figure 1: Base / Historical / Recalibrated
# ============================================================

def plot_fig1_overall_dot_ci(summary: pd.DataFrame, results: pd.DataFrame):
    """Single-column Figure 1: main SPX exceedance dots, Wilson intervals, and value labels.

    Numeric labels are placed just outside the Wilson intervals, not on the
    colored center lines. This keeps the exact values visible while preserving
    the main visual comparison against the 10% target line.
    """
    df = get_main_summary(summary)
    if df.empty:
        print("[SKIP] fig1: no main summary rows")
        return

    counts = get_results_counts(results)

    df = df.merge(
        counts.loc[
            counts["experiment_id"] == MAIN_EXPERIMENT_ID,
            ["book_type", "n_obs"],
        ],
        on="book_type",
        how="left",
    )

    # Export exact values behind the figure for replication / appendix checks.
    fig1_value_cols = [
        "book_type",
        "n_obs",
        "empirical_exceedance_rate_base",
        "empirical_exceedance_rate_hist",
        "empirical_exceedance_rate_conf",
    ]
    fig1_value_cols = [c for c in fig1_value_cols if c in df.columns]
    fig1_values = df[fig1_value_cols].copy()
    fig1_values = fig1_values.rename(
        columns={
            "book_type": "book",
            "n_obs": "n",
            "empirical_exceedance_rate_base": "lgbm_reference",
            "empirical_exceedance_rate_hist": "historical",
            "empirical_exceedance_rate_conf": "recalibrated",
        }
    )
    fig1_values.to_csv(FIG_DIR / "fig1_overall_dot_ci_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.5, 2.55))

    base_y = np.arange(len(df))[::-1]
    offsets = np.array([0.20, 0.00, -0.20])

    # Precompute values and Wilson intervals so labels can be placed outside CI bars.
    plot_cache = []
    x_max_for_axis = ALPHA

    for j, (col, label, color) in enumerate(METHOD_SPECS):
        if col not in df.columns:
            continue

        ys = base_y + offsets[j]
        xs = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        ns = pd.to_numeric(df["n_obs"], errors="coerce").to_numpy(dtype=float)

        lows, highs = [], []
        for p, n in zip(xs, ns):
            if np.isfinite(p) and np.isfinite(n) and n > 0:
                k = int(round(p * n))
                low, high = wilson_interval(k, int(n))
            else:
                low, high = np.nan, np.nan
            lows.append(low)
            highs.append(high)

        lows = np.asarray(lows, dtype=float)
        highs = np.asarray(highs, dtype=float)
        finite_for_axis = np.concatenate([
            xs[np.isfinite(xs)],
            highs[np.isfinite(highs)],
        ])
        if finite_for_axis.size:
            x_max_for_axis = max(x_max_for_axis, float(np.nanmax(finite_for_axis)))

        xerr = np.vstack([xs - lows, highs - xs])
        plot_cache.append((j, col, label, color, xs, ys, lows, highs, xerr))

    # Leave room for value labels placed beyond the Wilson upper endpoint.
    x_right = max(0.275, float(x_max_for_axis) + 0.055)

    # Draw target line behind all series and annotations.
    ax.axvline(
        ALPHA,
        linestyle="--",
        linewidth=1.35,
        color="black",
        label="_nolegend_",
        zorder=1,
    )

    for j, col, label, color, xs, ys, lows, highs, xerr in plot_cache:
        for x, y in zip(xs, ys):
            if np.isfinite(x):
                ax.hlines(
                    y=y,
                    xmin=min(ALPHA, x),
                    xmax=max(ALPHA, x),
                    linewidth=1.35,
                    alpha=0.9,
                    color=color,
                    zorder=2,
                )

        ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            fmt="o",
            color=color,
            elinewidth=0.95,
            capsize=2.3,
            markersize=4.8,
            label=label,
            zorder=3,
        )

        # Put each number just outside the Wilson CI, not on the colored line.
        # This produces a compact paper figure and avoids the floating labels
        # caused by large point-offset annotations.
        for x, y, low, high in zip(xs, ys, lows, highs):
            if not np.isfinite(x):
                continue

            label_gap = 0.004
            label_x = x + label_gap
            ha = "left"

            if np.isfinite(high):
                label_x = high + label_gap

            # If a label would run into the right boundary, place it to the left
            # of the Wilson interval instead.
            if label_x > x_right - 0.026:
                if np.isfinite(low):
                    label_x = low - label_gap
                else:
                    label_x = x - label_gap
                ha = "right"

            ax.text(
                label_x,
                y,
                f"{x:.3f}",
                fontsize=5.2,
                ha=ha,
                va="center",
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.10",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
                zorder=6,
                clip_on=False,
            )

    ax.set_yticks(base_y)
    ax.set_yticklabels([prettify_book_single_col(b) for b in df["book_type"]], fontsize=7.8)
    ax.set_xlabel("Empirical exceedance rate", fontsize=8.4)
    ax.tick_params(axis="x", labelsize=7.6)
    ax.grid(axis="x", alpha=0.25)
    clean_axes(ax)

    ax.set_xlim(0.0, x_right)
    ax.set_ylim(float(base_y.min()) - 0.50, float(base_y.max()) + 0.50)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fontsize=6.8,
        columnspacing=0.8,
        handletextpad=0.45,
        handlelength=1.35,
        borderpad=0.25,
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.80)

    savefig(fig, "fig1_overall_dot_ci.png")

# ============================================================
# Figure 1b: Full baseline comparison
# ============================================================

def plot_fig1b_baseline_comparison(baseline: pd.DataFrame):
    """Two-column appendix figure: full baseline comparison."""
    if baseline.empty:
        print(f"[SKIP] fig1b: {BASELINE_FILE} not found")
        return

    use_books = BOOK_ORDER + ["pooled"]

    df = baseline.loc[baseline["book_type"].isin(use_books)].copy()
    df = df.loc[df["method"].isin(BASELINE_METHOD_ORDER)].copy()

    if df.empty:
        print("[SKIP] fig1b: no requested methods found")
        print(f"Available methods: {sorted(baseline['method'].astype(str).unique().tolist())}")
        return

    book_pos = {b: i for i, b in enumerate(use_books)}
    method_pos = {m: i for i, m in enumerate(BASELINE_METHOD_ORDER)}

    df["book_order"] = df["book_type"].map(book_pos)
    df["method_order"] = df["method"].map(method_pos)
    df = df.sort_values(["book_order", "method_order"])

    fig, ax = plt.subplots(figsize=(6.6, 3.0))

    base_y = np.arange(len(use_books))[::-1]
    offsets = np.linspace(0.30, -0.30, len(BASELINE_METHOD_ORDER))

    max_x = 0.0

    for j, method in enumerate(BASELINE_METHOD_ORDER):
        sub = df.loc[df["method"] == method].copy()
        if sub.empty:
            continue

        xs = sub["empirical_exceedance_rate"].to_numpy(dtype=float)
        ns = sub["n_backtest_days"].to_numpy(dtype=float)
        ys = np.array(
            [base_y[use_books.index(b)] for b in sub["book_type"]]
        ) + offsets[j]

        lows, highs = [], []
        for p, n in zip(xs, ns):
            if np.isfinite(p) and np.isfinite(n) and n > 0:
                k = int(round(p * n))
                low, high = wilson_interval(k, int(n))
            else:
                low, high = np.nan, np.nan
            lows.append(low)
            highs.append(high)

        lows = np.asarray(lows, dtype=float)
        highs = np.asarray(highs, dtype=float)
        xerr = np.vstack([xs - lows, highs - xs])

        ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            fmt="o",
            color=BASELINE_COLORS.get(method, f"C{j}"),
            markersize=5.2,
            elinewidth=0.95,
            capsize=2.4,
            label=BASELINE_METHOD_LABELS.get(method, method),
            alpha=0.95,
        )

        if len(xs):
            max_x = max(max_x, np.nanmax(xs))

    ax.axvline(
        ALPHA,
        linestyle="--",
        linewidth=1.4,
        color="black",
        label=f"Target = {ALPHA:.2f}",
    )

    ax.set_yticks(base_y)
    ax.set_yticklabels([prettify_book_short(b) for b in use_books])
    ax.set_xlabel("Empirical exceedance rate")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0.0, max(0.24, max_x + 0.05))
    clean_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fontsize=7,
        columnspacing=0.7,
        handletextpad=0.4,
        borderpad=0.25,
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.80)

    savefig(fig, "fig1b_baseline_comparison_ci.png")

# ============================================================
# Figure 1c: Baseline exceedance-gap heatmap
# Usually appendix-only.
# ============================================================

def plot_fig1c_baseline_gap_heatmap(baseline: pd.DataFrame):
    """Two-column appendix heatmap of exceedance gaps."""
    if baseline.empty:
        print(f"[SKIP] fig1c: {BASELINE_FILE} not found")
        return

    use = baseline.loc[baseline["book_type"].isin(BOOK_ORDER + ["pooled"])].copy()
    use = use.loc[use["method"].isin(BASELINE_METHOD_ORDER)].copy()

    if use.empty:
        print("[SKIP] fig1c: no requested methods found")
        return

    use["gap"] = use["empirical_exceedance_rate"] - ALPHA

    pivot = (
        use.pivot(index="book_type", columns="method", values="gap")
        .reindex(index=BOOK_ORDER + ["pooled"], columns=BASELINE_METHOD_ORDER)
    )

    fig, ax = plt.subplots(figsize=(6.6, 2.35))

    max_abs = np.nanmax(np.abs(pivot.values))
    max_abs = max(max_abs, 0.01)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([prettify_book_short(x) for x in pivot.index])

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [BASELINE_METHOD_LABELS.get(x, x) for x in pivot.columns],
        rotation=22,
        ha="right",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
    cbar.ax.tick_params(labelsize=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7)

    clean_axes(ax)
    fig.subplots_adjust(left=0.12, right=0.93, bottom=0.33, top=0.96)
    savefig(fig, "fig1c_baseline_gap_heatmap.png")

# ============================================================
# Figure 2: Rolling 50-day exceedance gap
# ============================================================

def plot_fig2_rolling_gap(rolling: pd.DataFrame):
    """Two-column main rolling 50-day exceedance-gap figure."""
    df = rolling.loc[rolling["experiment_id"] == MAIN_EXPERIMENT_ID].copy()
    if df.empty:
        print("[SKIP] fig2: no main rolling rows")
        return

    df["book_type"] = pd.Categorical(
        df["book_type"], categories=BOOK_ORDER, ordered=True
    )
    df = df.sort_values(["book_type", "date"])

    gap_cols = [
        "roll50_exceed_base",
        "roll50_exceed_hist",
        "roll50_exceed_conf",
    ]

    valid_gap_cols = [c for c in gap_cols if c in df.columns]
    if not valid_gap_cols:
        print("[SKIP] fig2: no rolling exceedance columns found")
        return

    tmp = df[valid_gap_cols] - ALPHA

    global_abs_max = np.nanmax(np.abs(tmp.to_numpy())) if tmp.size else 0.08
    y_lim = max(0.08, np.ceil(global_abs_max * 100) / 100 + 0.02)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.6, 4.55), sharex=True)

    line_specs = [
        ("roll50_exceed_base", "LGBM ref.", "C0"),
        ("roll50_exceed_hist", "Historical", "C1"),
        ("roll50_exceed_conf", "Recalibrated", "C2"),
    ]

    for ax, book in zip(axes, BOOK_ORDER):
        sub = df.loc[df["book_type"] == book].copy()

        for col, label, color in line_specs:
            if col not in sub.columns:
                continue
            gap = sub[col] - ALPHA
            ax.plot(sub["date"], gap, linewidth=1.3, label=label, color=color)

        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black")
        ax.axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color="grey")
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(prettify_book(book), pad=2, fontsize=10)
        ax.grid(alpha=0.25)
        clean_axes(ax)

    fig.supylabel("50-day exceedance gap", x=0.02, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=True,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.25,
    )
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.10, top=0.84, hspace=0.55)
    savefig(fig, "fig2_rolling50_gap.png")

# ============================================================
# Appendix Figure: Rolling gaps for all baselines
# ============================================================

def plot_fig2b_rolling_gap_all_baselines(rolling: pd.DataFrame):
    """Two-column appendix rolling gap figure for all baselines."""
    df = rolling.loc[rolling["experiment_id"] == MAIN_EXPERIMENT_ID].copy()
    if df.empty:
        return

    needed_any = [
        "roll50_exceed_ewma",
        "roll50_exceed_caviar",
        "roll50_exceed_garch_t",
    ]

    if not any(c in df.columns for c in needed_any):
        print("[SKIP] rolling all-baselines: new baseline rolling columns not present")
        return

    df["book_type"] = pd.Categorical(
        df["book_type"], categories=BOOK_ORDER, ordered=True
    )
    df = df.sort_values(["book_type", "date"])

    line_specs = [
        ("roll50_exceed_hist", "Hist.", "C1"),
        ("roll50_exceed_ewma", "EWMA Hist.", "C3"),
        ("roll50_exceed_caviar", "CAViaR", "C4"),
        ("roll50_exceed_garch_t", "GARCH-t", "C5"),
        ("roll50_exceed_base", "LightGBM", "C0"),
        ("roll50_exceed_conf", "LightGBM+Recal.", "C2"),
    ]

    gap_cols = [c for c, _, _ in line_specs if c in df.columns]
    tmp = df[gap_cols] - ALPHA

    global_abs_max = np.nanmax(np.abs(tmp.to_numpy())) if tmp.size else 0.08
    y_lim = max(0.08, np.ceil(global_abs_max * 100) / 100 + 0.02)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.6, 4.85), sharex=True)

    for ax, book in zip(axes, BOOK_ORDER):
        sub = df.loc[df["book_type"] == book].copy()

        for col, label, color in line_specs:
            if col not in sub.columns:
                continue
            ax.plot(
                sub["date"],
                sub[col] - ALPHA,
                linewidth=1.05,
                label=label,
                color=color,
                alpha=0.92,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1.1, color="black")
        ax.axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color="grey")
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(prettify_book(book), pad=2, fontsize=10)
        ax.grid(alpha=0.25)
        clean_axes(ax)

    fig.supylabel("50-day gap", x=0.02, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=6,
        frameon=True,
        fontsize=7,
        columnspacing=0.7,
        handletextpad=0.35,
        borderpad=0.25,
    )
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.10, top=0.82, hspace=0.58)
    savefig(fig, "appendix_rolling50_gap_all_baselines.png")

# ============================================================
# Figure 3: Crisis-window exceedance gap
# Usually appendix-only.
# ============================================================

def plot_fig3_crisis_gap(results: pd.DataFrame):
    """Single-column appendix figure for the COVID crisis window."""
    df = results.loc[
        (results["experiment_id"] == MAIN_EXPERIMENT_ID)
        & (results["date"] >= CRISIS_START)
        & (results["date"] <= CRISIS_END)
    ].copy()

    if df.empty:
        print("[SKIP] fig3: no crisis rows")
        return

    df["book_type"] = pd.Categorical(
        df["book_type"], categories=BOOK_ORDER, ordered=True
    )
    df = df.sort_values(["book_type", "date"])

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.5, 4.5), sharex=True)

    for ax, book in zip(axes, BOOK_ORDER):
        sub = df.loc[df["book_type"] == book].copy()

        if "q_hat_base" not in sub.columns or "var_conf" not in sub.columns:
            print("[SKIP] fig3: q_hat_base or var_conf missing")
            return

        sub["gap_base"] = sub["loss_norm_tp1"] - sub["q_hat_base"]
        sub["gap_cal"] = sub["loss_norm_tp1"] - sub["var_conf"]

        ax.plot(
            sub["date"],
            sub["gap_base"],
            linewidth=1.2,
            label="Base gap",
            color="C0",
        )
        ax.plot(
            sub["date"],
            sub["gap_cal"],
            linewidth=1.2,
            label="Recalibrated gap",
            color="C2",
        )

        ax.fill_between(
            sub["date"],
            0,
            sub["gap_base"],
            where=sub["gap_base"] > 0,
            alpha=0.15,
            color="C0",
            interpolate=True,
        )
        ax.fill_between(
            sub["date"],
            0,
            sub["gap_cal"],
            where=sub["gap_cal"] > 0,
            alpha=0.15,
            color="C2",
            interpolate=True,
        )

        base_pos = sub.loc[sub["gap_base"] > 0]
        cal_pos = sub.loc[sub["gap_cal"] > 0]

        ax.scatter(
            base_pos["date"],
            base_pos["gap_base"],
            s=12,
            marker="o",
            color="C0",
            alpha=0.9,
        )
        ax.scatter(
            cal_pos["date"],
            cal_pos["gap_cal"],
            s=14,
            marker="x",
            color="C2",
            alpha=0.9,
        )

        ax.axhline(0.0, linestyle="--", linewidth=1.1, color="black")
        ax.set_title(prettify_book(book), pad=2)
        ax.grid(alpha=0.25)
        clean_axes(ax)

    fig.supylabel("Exceedance gap", x=0.01, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h

    axes[0].legend(
        seen.values(),
        seen.keys(),
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fontsize=7,
        borderpad=0.25,
    )

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axes[-1].get_xticklabels(), rotation=35, ha="right")
    axes[-1].set_xlabel("Date")

    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.88, hspace=0.52)
    savefig(fig, "fig3_crisis_gap.png")

# ============================================================
# Figure 4: Robustness heatmap
# Clean single-column version.
# ============================================================

def plot_fig4_robustness_heatmap(summary: pd.DataFrame):
    resolved = resolve_robustness_ids(summary)
    if not resolved:
        print("[SKIP] fig4: no robustness experiments found")
        return

    labels = [x[0] for x in resolved]
    keep_ids = [x[1] for x in resolved]

    df = summary.loc[summary["experiment_id"].isin(keep_ids)].copy()
    if df.empty:
        print("[SKIP] fig4: no matching rows")
        return

    df["book_type"] = pd.Categorical(
        df["book_type"], categories=BOOK_ORDER, ordered=True
    )
    df["experiment_id"] = pd.Categorical(
        df["experiment_id"], categories=keep_ids, ordered=True
    )

    df["distance_to_target"] = (
        df["empirical_exceedance_rate_conf"] - ALPHA
    ).abs()

    pivot = (
        df.pivot(
            index="book_type",
            columns="experiment_id",
            values="distance_to_target",
        )
        .reindex(index=BOOK_ORDER, columns=keep_ids)
    )

    fig, ax = plt.subplots(figsize=(3.5, 1.85))

    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([prettify_book_heatmap(x) for x in pivot.index], fontsize=8)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.025)
    cbar.ax.tick_params(labelsize=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if val < 0.004 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.34, top=0.96)
    savefig(fig, "fig4_robustness_heatmap.png")

# ============================================================
# Figure 5: Feasibility + approximate mark share
# Safe against empty strict/main intersections.
# ============================================================

def plot_fig5_feasibility_fallback(summary: pd.DataFrame):
    """Two-column feasibility/marking-retention figure."""
    if MAIN_EXPERIMENT_ID is None:
        print("[SKIP] fig5: main experiment id unresolved")
        return

    if STRICT_MARKING_EXPERIMENT_ID is None:
        print("[SKIP] fig5: strict marking experiment id unresolved")
        return

    summary = summary.copy()

    robust = summary.loc[summary["experiment_id"] == MAIN_EXPERIMENT_ID].copy()
    strict = summary.loc[summary["experiment_id"] == STRICT_MARKING_EXPERIMENT_ID].copy()

    if robust.empty or strict.empty:
        print("[SKIP] fig5: no matching main or strict marking rows.")
        return

    robust["book_type"] = pd.Categorical(
        robust["book_type"], categories=BOOK_ORDER, ordered=True
    )
    strict["book_type"] = pd.Categorical(
        strict["book_type"], categories=BOOK_ORDER, ordered=True
    )

    robust = robust.sort_values("book_type")
    strict = strict.sort_values("book_type")

    needed_cols = [
        "book_type",
        "n_backtest_days",
        "sum_option_mark_exact_t",
        "sum_option_mark_contract_t",
        "sum_option_mark_interp_t",
        "sum_option_mark_nearest_t",
    ]

    missing_cols = [c for c in needed_cols if c not in robust.columns]
    if missing_cols:
        print(f"[SKIP] fig5: missing columns: {missing_cols}")
        return

    merged = robust[needed_cols].rename(
        columns={"n_backtest_days": "n_robust"}
    ).merge(
        strict[["book_type", "n_backtest_days"]].rename(
            columns={"n_backtest_days": "n_strict"}
        ),
        on="book_type",
        how="left",
    )

    if merged.empty:
        print("[SKIP] fig5: merged main/strict marking table is empty.")
        return

    merged["retention_robust"] = 1.0
    merged["retention_strict"] = merged["n_strict"] / merged["n_robust"]

    total_marks = (
        merged["sum_option_mark_exact_t"]
        + merged["sum_option_mark_contract_t"]
        + merged["sum_option_mark_interp_t"]
        + merged["sum_option_mark_nearest_t"]
    ).replace(0, np.nan)

    merged["approx_mark_share"] = (
        merged["sum_option_mark_interp_t"] + merged["sum_option_mark_nearest_t"]
    ) / total_marks

    finite_share = pd.to_numeric(merged["approx_mark_share"], errors="coerce")
    finite_share = finite_share[np.isfinite(finite_share)]

    if len(finite_share) == 0:
        print("[SKIP] fig5: approximate-mark share is empty or all NaN.")
        return

    fig, ax1 = plt.subplots(figsize=(6.6, 2.8))

    x = np.arange(len(merged))
    width = 0.28

    bars1 = ax1.bar(
        x - width / 2,
        merged["retention_robust"],
        width=width,
        label="Robust retention",
    )
    bars2 = ax1.bar(
        x + width / 2,
        merged["retention_strict"],
        width=width,
        label="Strict retention",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([prettify_book_short(b) for b in merged["book_type"]])
    ax1.set_ylabel("Retention vs robust marking")
    ax1.set_ylim(0, 1.16)
    ax1.grid(axis="y", alpha=0.25)
    clean_axes(ax1)

    for rect, n in zip(bars1, merged["n_robust"]):
        if pd.notna(n):
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.015,
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for rect, n in zip(bars2, merged["n_strict"]):
        if pd.notna(n):
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.015,
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        merged["approx_mark_share"],
        marker="o",
        linewidth=1.6,
        label="Approx. mark share",
    )
    ax2.set_ylabel("Approx. mark share")
    ax2.set_ylim(0, max(0.3, float(finite_share.max()) + 0.05))
    ax2.spines["top"].set_visible(False)

    for xx, yy in zip(x, merged["approx_mark_share"]):
        if pd.notna(yy):
            ax2.text(xx, yy + 0.01, f"{yy:.3f}", ha="center", va="bottom", fontsize=7)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.25,
    )

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.20, top=0.80)
    savefig(fig, "fig5_feasibility_fallback.png")

# ============================================================
# Appendix yearly figure
# ============================================================

def plot_appendix_yearly(yearly: pd.DataFrame):
    """Single-column appendix yearly exceedance figure."""
    if yearly.empty:
        print("[SKIP] yearly figure: yearly file empty")
        return

    df = yearly.loc[yearly["experiment_id"].astype(str) == str(MAIN_EXPERIMENT_ID)].copy()
    if df.empty:
        print("[SKIP] yearly figure: no main experiment rows")
        return

    if "year" not in df.columns:
        print(f"[SKIP] yearly figure: missing year column. Columns={list(df.columns)}")
        return

    # Robust year parsing.
    year_num = pd.to_numeric(df["year"], errors="coerce")

    # If year is stored as date-like strings, parse it.
    if year_num.notna().sum() == 0:
        year_num = pd.to_datetime(df["year"], errors="coerce").dt.year

    # If year is accidentally stored as YYYYMMDD-like numbers, parse to year.
    elif year_num.dropna().median() > 10000:
        year_num = pd.to_datetime(df["year"].astype(str), errors="coerce").dt.year

    df["year_plot"] = year_num
    df = df.loc[df["year_plot"].notna()].copy()
    if df.empty:
        print("[SKIP] yearly figure: no valid years after parsing")
        return

    df["year_plot"] = df["year_plot"].astype(int)

    df["book_type"] = pd.Categorical(
        df["book_type"], categories=BOOK_ORDER, ordered=True
    )
    df = df.sort_values(["book_type", "year_plot"])

    # Preferred yearly-summary columns.
    line_specs = [
        ("empirical_exceedance_rate_base", "LGBM ref.", "C0"),
        ("empirical_exceedance_rate_hist", "Historical", "C1"),
        ("empirical_exceedance_rate_conf", "Recalibrated", "C2"),
    ]

    # Fallback for older yearly files that may use rolling-style column names.
    if not any(c in df.columns for c, _, _ in line_specs):
        alt_specs = [
            ("roll50_exceed_base", "LGBM ref.", "C0"),
            ("roll50_exceed_hist", "Historical", "C1"),
            ("roll50_exceed_conf", "Recalibrated", "C2"),
        ]

        if any(c in df.columns for c, _, _ in alt_specs):
            line_specs = alt_specs
        else:
            print("[SKIP] yearly figure: no yearly exceedance columns found.")
            print(f"Available columns: {list(df.columns)}")
            return

    years = sorted(df["year_plot"].unique().tolist())
    if not years:
        print("[SKIP] yearly figure: no year values")
        return

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.5, 4.25), sharex=True)

    for ax, book in zip(axes, BOOK_ORDER):
        sub = df.loc[df["book_type"] == book].copy()
        plotted = False

        for col, label, color in line_specs:
            if col not in sub.columns:
                continue

            y = pd.to_numeric(sub[col], errors="coerce")
            mask = y.notna() & sub["year_plot"].notna()

            if not mask.any():
                continue

            ax.plot(
                sub.loc[mask, "year_plot"],
                y.loc[mask],
                marker="o",
                linewidth=1.2,
                label=label,
                color=color,
            )
            plotted = True

        ax.axhline(ALPHA, linestyle="--", linewidth=1.1, color="black")
        ax.set_title(prettify_book(book), pad=2)
        ax.grid(alpha=0.25)
        clean_axes(ax)

        # Force x-limits even if one book has missing series.
        ax.set_xlim(min(years) - 0.35, max(years) + 0.35)

        if plotted and "n_days" in sub.columns:
            ymin, ymax = ax.get_ylim()
            y_text = ymin + 0.06 * (ymax - ymin)

            for _, row in sub.iterrows():
                n_val = pd.to_numeric(row.get("n_days"), errors="coerce")
                if pd.notna(n_val):
                    ax.text(
                        int(row["year_plot"]),
                        y_text,
                        f"n={int(n_val)}",
                        ha="center",
                        va="bottom",
                        fontsize=6.2,
                        clip_on=True,
                    )

    fig.supylabel("Yearly exceedance", x=0.015, fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(
            handles,
            labels,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
            fontsize=7,
            columnspacing=0.8,
            handletextpad=0.4,
            borderpad=0.25,
        )

    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(years)
    axes[-1].set_xticklabels([str(y) for y in years])

    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.11, top=0.88, hspace=0.50)
    savefig(fig, "appendix_yearly_exceedance_with_n.png")

# ============================================================
# Main
# ============================================================

def main():
    print(f"Reading results from: {BASE_DIR}")
    print(f"Saving figures to:   {FIG_DIR}")
    summary, results, rolling, crisis, yearly, baseline = load_csvs()
    audit_no_leakage_feature_file(required=True)
    resolve_runtime_experiments(summary)

    plot_fig1_overall_dot_ci(summary, results)
    plot_fig1b_baseline_comparison(baseline)
    plot_fig1c_baseline_gap_heatmap(baseline)

    plot_fig2_rolling_gap(rolling)
    plot_fig2b_rolling_gap_all_baselines(rolling)

    plot_fig3_crisis_gap(results)
    plot_fig4_robustness_heatmap(summary)
    plot_fig5_feasibility_fallback(summary)
    plot_appendix_yearly(yearly)

    print(f"\nAll figures saved to:\n{FIG_DIR}")


if __name__ == "__main__":
    main()