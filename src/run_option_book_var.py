"""
Main pipeline for the option-book VaR adaptive calibration study.

This is the canonical calibration entry point for both SPX and QQQ.
Asset-specific behavior is controlled through environment variables or an
optional calibration_config.json file. The script directory and the data
directory are intentionally decoupled, so the code can live in a paper
repository while the large option files stay in a separate data folder.
The legacy robustness_config.json file is still accepted for backward
compatibility.

Internal legacy result columns such as `_conf` are retained only for
compatibility with earlier plotting and table code. Paper-facing wording should
use calibrated, calibration, or sequential calibration.

No-leakage revision: realized current t-to-t+1 next-day marking outcome
columns are retained as diagnostics in the output CSVs but are deliberately
excluded from the forecast-time feature matrix. Only lagged marking diagnostics
constructed from already-realized past transitions are eligible as features.
"""

import os
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.stats import t as student_t, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Reproducibility
# ============================================================
np.random.seed(42)

# ============================================================
# Asset and path configuration
# ============================================================
# The script can run on SPX or on an ETF option universe such as QQQ.
# The script location and the data location are deliberately separate.
# This matters when run_option_book_var.py is stored in a paper/code folder while the
# large parquet files are stored in a data folder.
#
# Path roles:
# - SCRIPT_DIR: directory that contains this Python file.
# - DATA_DIR: directory that contains data_YYYY.parquet and auxiliary parquet files.
# - OUT_DIR: directory where result CSV files are written.
#
# Resolution priority:
# 1) OPTION_CALIBRATION_CONFIG or OPTION_ROBUSTNESS_CONFIG, if provided;
# 2) calibration_config.json or calibration_config_QQQ.json next to this script;
# 3) calibration_config.json or robustness_config.json in the data directory;
# 4) explicit environment variables;
# 5) QQQ data directory fallback for run_option_book_var.py.

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR


def _read_json_if_exists(path: Path) -> dict:
    try:
        if path is not None and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


def _first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        try:
            if p is not None and Path(p).exists():
                return Path(p)
        except Exception:
            continue
    return None


_DEFAULT_DATA_DIR = REPO_ROOT / "data"
_ENV_DATA_DIR = os.environ.get(
    "OPTION_DATA_DIR",
    os.environ.get(
        "OPTION_PRICING_BASE_DIR",
        os.environ.get(
            "OPTION_CALIBRATION_BASE_DIR",
            os.environ.get("OPTION_ROBUSTNESS_BASE_DIR", ""),
        ),
    ),
)
_PRECONFIG_DATA_DIR = Path(_ENV_DATA_DIR) if str(_ENV_DATA_DIR).strip() else _DEFAULT_DATA_DIR

_CONFIG_ENV = os.environ.get("OPTION_CALIBRATION_CONFIG", os.environ.get("OPTION_ROBUSTNESS_CONFIG", ""))
if str(_CONFIG_ENV).strip():
    _CONFIG_PATH = Path(_CONFIG_ENV)
else:
    _CONFIG_PATH = _first_existing_path([
        SCRIPT_DIR / "calibration_config.json",
        SCRIPT_DIR / "calibration_config_QQQ.json",
        SCRIPT_DIR / "robustness_config.json",
        REPO_ROOT / "configs" / "spx_config.json",
        REPO_ROOT / "configs" / "qqq_config.json",
        _PRECONFIG_DATA_DIR / "calibration_config.json",
        _PRECONFIG_DATA_DIR / "calibration_config_QQQ.json",
        _PRECONFIG_DATA_DIR / "robustness_config.json",
    ])

ASSET_CONFIG = _read_json_if_exists(_CONFIG_PATH) if _CONFIG_PATH is not None else {}

if str(_ENV_DATA_DIR).strip():
    DATA_DIR = Path(_ENV_DATA_DIR)
elif ASSET_CONFIG.get("data_dir"):
    DATA_DIR = Path(ASSET_CONFIG["data_dir"])
else:
    DATA_DIR = _DEFAULT_DATA_DIR

# BASE_DIR is kept as an internal alias for compatibility with the older code.
BASE_DIR = DATA_DIR

TICKER = str(os.environ.get("OPTION_TICKER", ASSET_CONFIG.get("ticker", "QQQ"))).upper()
_ASSET_SECID_RAW = os.environ.get("OPTION_SECID", ASSET_CONFIG.get("secid", "107899" if TICKER == "QQQ" else ""))
try:
    ASSET_SECID = int(_ASSET_SECID_RAW) if str(_ASSET_SECID_RAW).strip() != "" else None
except Exception:
    ASSET_SECID = None

YEAR_LIST = [int(y) for y in ASSET_CONFIG.get("years", list(range(2018, 2026)))]
OPTION_FILE_FMT = str(ASSET_CONFIG.get("option_file_fmt", "data_{year}.parquet"))
SECURITY_PRICE_FILE = str(ASSET_CONFIG.get("price_file", "security_prices.parquet"))
ZCY_FILE = str(ASSET_CONFIG.get("zcy_file", "zero_coupon_yield.parquet"))
DIVIDEND_FILE = str(ASSET_CONFIG.get("div_file", "index_dividend_yield.parquet"))
VIX_FILE_NAME = str(ASSET_CONFIG.get("vix_file", "VIXCLS.parquet"))
VXV_FILE_NAME = str(ASSET_CONFIG.get("vxv_file", "VXVCLS.parquet"))

_RECOMMENDED = ASSET_CONFIG.get("recommended_main_pipeline_settings", {}) if isinstance(ASSET_CONFIG, dict) else {}
SETTLEMENT_POLICY = str(os.environ.get("OPTION_SETTLEMENT_POLICY", _RECOMMENDED.get("settlement_policy", "all"))).lower()
REQUIRE_EUROPEAN_STYLE = str(
    os.environ.get("OPTION_REQUIRE_EUROPEAN_STYLE", _RECOMMENDED.get("require_european_style", False))
).lower() in {"1", "true", "yes", "y"}
USE_VENDOR_FORWARD_PRICE = str(
    os.environ.get("OPTION_USE_VENDOR_FORWARD_PRICE", _RECOMMENDED.get("use_vendor_forward_price", True))
).lower() in {"1", "true", "yes", "y"}

# Use a no-leakage default result folder so reruns do not overwrite legacy results.
_RESULTS_SUBDIR_DEFAULT = str(Path("results") / ("qqq" if TICKER == "QQQ" else "spx"))
_RESULTS_DIR_CONFIG = ASSET_CONFIG.get("results_dir", "") if isinstance(ASSET_CONFIG, dict) else ""
if os.environ.get("OPTION_RESULTS_DIR", "").strip():
    OUT_DIR = Path(os.environ["OPTION_RESULTS_DIR"])
elif str(_RESULTS_DIR_CONFIG).strip():
    OUT_DIR = Path(_RESULTS_DIR_CONFIG)
else:
    OUT_DIR = REPO_ROOT / os.environ.get("OPTION_RESULTS_SUBDIR", _RESULTS_SUBDIR_DEFAULT)
OUT_DIR.mkdir(exist_ok=True, parents=True)

VIX_FILE = DATA_DIR / VIX_FILE_NAME
VXV_FILE = DATA_DIR / VXV_FILE_NAME

print(f"Script directory: {SCRIPT_DIR}")
print(f"Data directory:   {DATA_DIR}")
print(f"Results directory:{OUT_DIR}")
print(f"Ticker: {TICKER}; secid: {ASSET_SECID}")
if _CONFIG_PATH is not None:
    print(f"Config file:      {_CONFIG_PATH}")
else:
    print("Config file:      not found; using built-in defaults")

# ============================================================
# Core parameters
# ============================================================
MIN_DTE = 14
MAX_DTE = 120
K_MIN = float(os.environ.get("OPTION_K_MIN", ASSET_CONFIG.get("k_min", -0.35 if TICKER != "SPX" else -0.20)))
K_MAX = float(os.environ.get("OPTION_K_MAX", ASSET_CONFIG.get("k_max", 0.25 if TICKER != "SPX" else 0.10)))
SURFACE_K_MIN = float(os.environ.get("OPTION_SURFACE_K_MIN", ASSET_CONFIG.get("surface_k_min", K_MIN)))
SURFACE_K_MAX = float(os.environ.get("OPTION_SURFACE_K_MAX", ASSET_CONFIG.get("surface_k_max", K_MAX)))

MIN_BID = 0.0
MIN_MID = 0.05
MAX_REL_SPREAD = 0.50
MIN_OI = 1
MIN_VOL = 1

LOOKBACK = 20
TRAIN_WINDOW = 252
RETRAIN_EVERY = 5

ALPHA = 0.10
QUANTILE_LEVEL = 1.0 - ALPHA

CALIB_WINDOW = 126
MIN_CAL_SCORES = 30
TIME_DECAY = 0.01

TARGET_DTE = 30
DEFAULT_BUFFER = 0.0
EPS = 1e-12

CRISIS_START = pd.Timestamp("2020-02-20")
CRISIS_END = pd.Timestamp("2020-04-15")

ROLL_DIAG_WINDOWS = [50, 100]
SHORT_PUT_SPREAD_BOOK_TYPE = "short_put_spread_25delta_10delta_30d"
MAIN_BOOK_TYPES = [
    "atm_straddle_30d",
    "risk_reversal_25d_30d",
    SHORT_PUT_SPREAD_BOOK_TYPE,
]
BOOK_TYPES = MAIN_BOOK_TYPES

# ============================================================
# Forecast-time information-set guardrails
# ============================================================
# These columns are produced only after marking the t -> t+1 transition.
# They are valid realized diagnostics and can remain in the output files,
# but they are not date-t forecast features. Including them in X_t would
# contaminate the out-of-sample backtest with next-day marking information.
REALIZED_CURRENT_MARKING_DIAGNOSTIC_COLS = [
    "n_option_mark_exact_t",
    "n_option_mark_contract_t",
    "n_option_mark_interp_t",
    "n_option_mark_nearest_t",
    "n_option_mark_fallback_t",
]

# These lagged marking diagnostics use only already-matured transitions.
# For example, mark_fallback_count_lag1 at date t is based on the observed
# marking outcome for the previous book transition, so it is known at the
# date-t forecast time.
LAGGED_MARKING_FEATURE_COLS = [
    "mark_exact_count_lag1",
    "mark_contract_count_lag1",
    "mark_interp_count_lag1",
    "mark_nearest_count_lag1",
    "mark_fallback_count_lag1",
    "mark_proxy_count_lag1",
    "mark_exact_rate_5",
    "mark_contract_rate_5",
    "mark_interp_rate_5",
    "mark_nearest_rate_5",
    "mark_fallback_rate_5",
    "mark_proxy_rate_5",
    "mark_exact_rate_21",
    "mark_contract_rate_21",
    "mark_interp_rate_21",
    "mark_nearest_rate_21",
    "mark_fallback_rate_21",
    "mark_proxy_rate_21",
]

STRICT_EX_ANTE_FEATURE_CHECK = str(
    os.environ.get("OPTION_STRICT_EX_ANTE_FEATURE_CHECK", "1")
).lower() in {"1", "true", "yes", "y"}

# Main experiment and robustness settings
PRIMARY_LEARNER = "lightgbm"
PRIMARY_MARKING_MODE = "robust_all"          # "strict_exact_contract" or "robust_all"
PRIMARY_VAR_FLOOR = 0.0                        # None means no VaR floor

RUN_LEARNER_ROBUSTNESS = True
RUN_MARKING_SENSITIVITY = True
RUN_VAR_FLOOR_SENSITIVITY = True
RUN_BOOK_QUALITY_ROBUSTNESS = True

LEARNER_ROBUSTNESS_MODELS = ["gbr", "lightgbm", "xgboost"]
MARKING_SENSITIVITY_MODES = ["strict_exact_contract", "robust_all"]
VAR_FLOOR_SENSITIVITY_VALUES = [None, 0.0]

# ============================================================
# Book-selection quality robustness
# ============================================================
# loose: keep the original nearest-feasible standardized-book construction.
# strict_economic: require the selected legs to be economically close to the target book.
BOOK_QUALITY_MODE_LOOSE = "loose"
BOOK_QUALITY_MODE_STRICT = "strict_economic"
PRIMARY_BOOK_QUALITY_MODE = BOOK_QUALITY_MODE_LOOSE
BOOK_QUALITY_ROBUSTNESS_MODES = [BOOK_QUALITY_MODE_STRICT]

# The thresholds below are intentionally transparent and can be tightened in the paper appendix.
# For ATM, |log(K/F)| <= 0.05 is a clean economic-book screen.
# For delta-targeted legs, |delta - target_delta| <= 0.10 prevents late-sample chain thinning
# from turning a 25d/10d book into a deep-ITM or near-zero-delta substitute.
STRICT_ATM_MAX_ABS_K = 0.05
STRICT_25D_DELTA_MAX_ERR = 0.10
STRICT_10D_DELTA_MAX_ERR = 0.10
STRICT_MAX_ABS_DTE_ERROR = 7.0

BOOK_QUALITY_OUTPUT_COLS = [
    "book_quality_mode",
    "quality_pass_strict_economic",
    "book_quality_max_abs_k_t",
    "book_quality_mean_abs_k_t",
    "book_quality_max_abs_dte_error_t",
    "atm_call_abs_k_t",
    "atm_put_abs_k_t",
    "atm_max_abs_k_t",
    "rr_call_delta_error_t",
    "rr_put_delta_error_t",
    "rr_max_delta_error_t",
    "putspread_short_delta_error_t",
    "putspread_long_delta_error_t",
    "putspread_max_delta_error_t",
]

# ============================================================
# Classical VaR baselines added for paper comparison
# ============================================================
RUN_CLASSICAL_BASELINES = True
# By default, expensive classical baselines are computed for the main specification.
# Set to None to compute them for every robustness specification as well.
CLASSICAL_BASELINE_GROUPS = {"main"}

EWMA_LAMBDA = 0.97
CAVIAR_MIN_OBS = 120
GARCH_MIN_OBS = 120

# Numerical controls for rolling CAViaR and GARCH-t fits.
CAVIAR_MAXITER = 600
GARCH_MAXITER = 600

# Columns used by the yearly filter-attrition report
DIAG_USECOLS = [
    "secid", "date", "symbol", "exdate", "last_date", "cp_flag",
    "strike_price", "best_bid", "best_offer", "volume", "open_interest",
    "impl_volatility", "delta", "gamma", "vega", "theta", "optionid",
    "contract_size", "ss_flag", "forward_price", "expiry_indicator",
    "root", "suffix", "ticker", "index_flag", "issuer",
    "div_convention", "exercise_style", "am_settlement", "am_set_flag"
]

# ============================================================
# Risk-reversal standardization
# ============================================================
# A two-option risk reversal alone cannot be made delta-neutral with only
# the two option weights because the option deltas add in the same direction.
# The implementation therefore uses an option risk reversal plus a spot hedge.
# The default version is a unit risk reversal with delta hedging and gross
# option-premium normalization. A vega-normalized version can be used as a
# robustness extension if needed.
# ============================================================
RR_CALL_TARGET_DELTA = 0.25
RR_PUT_TARGET_DELTA = -0.25
RR_OPTION_SCALING = "unit"                 # "unit" or "vega_normalized"
RR_INCLUDE_SPOT_HEDGE = True
RR_NORMALIZATION = "gross_option_premium"  # This can also be set to gross_option_premium_plus_spot_notional

SHORT_PUT_SPREAD_SHORT_DELTA = -0.25
SHORT_PUT_SPREAD_LONG_DELTA = -0.10
SHORT_PUT_SPREAD_INCLUDE_SPOT_HEDGE = True

# Next-day option-marking rule:
# exact option id, exact contract, same-expiry interpolation, then nearest fallback.
NEXT_DAY_STRIKE_ABS_TOL = 1e-8
NEXT_DAY_MAX_DTE_GAP = 7.0
NEXT_DAY_NEAREST_STRIKE_WEIGHT = 0.05

# Base quantile learner settings
# The default base learner is LightGBM quantile regression.
BASE_MODEL_KIND = "lightgbm"   # "lightgbm", "xgboost", "gbr"

GBR_N_ESTIMATORS = 500
GBR_LEARNING_RATE = 0.03
GBR_MAX_DEPTH = 4
GBR_MIN_SAMPLES_LEAF = 10
GBR_SUBSAMPLE = 0.9

LGBM_N_ESTIMATORS = 400
LGBM_LEARNING_RATE = 0.03
LGBM_NUM_LEAVES = 31
LGBM_MIN_CHILD_SAMPLES = 20
LGBM_SUBSAMPLE = 0.9
LGBM_COLSAMPLE_BYTREE = 0.9
LGBM_REG_ALPHA = 0.0
LGBM_REG_LAMBDA = 1.0

XGB_N_ESTIMATORS = 500
XGB_LEARNING_RATE = 0.03
XGB_MAX_DEPTH = 4
XGB_MIN_CHILD_WEIGHT = 8.0
XGB_SUBSAMPLE = 0.9
XGB_COLSAMPLE_BYTREE = 0.9
XGB_REG_ALPHA = 0.0
XGB_REG_LAMBDA = 1.0

# ============================================================
# Utility functions
# ============================================================
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def find_file_stem(prefix: str, base_dir: Optional[Path] = None) -> Path:
    search_dir = Path(base_dir) if base_dir is not None else BASE_DIR
    files = list(search_dir.glob(f"{prefix}*"))
    if not files:
        raise FileNotFoundError(f"file not found: {search_dir / (prefix + '*')}")
    files = sorted(files, key=lambda p: str(p))
    return files[0]


def resolve_input_file(configured_name: Optional[str], fallback_prefix: str) -> Path:
    """Resolve an input file from an exact configured name or a prefix fallback."""
    if configured_name is not None and str(configured_name).strip() != "":
        configured = Path(str(configured_name))
        candidates = []
        if configured.is_absolute():
            candidates.append(configured)
        else:
            candidates.append(BASE_DIR / configured)
        for p in candidates:
            if p.exists():
                return p
            same_stem = sorted(p.parent.glob(f"{p.stem}*"), key=lambda x: str(x))
            if same_stem:
                return same_stem[0]
    return find_file_stem(fallback_prefix)


def resolve_option_year_file(year: int) -> Path:
    """Resolve the yearly option-chain file for the configured asset."""
    configured = OPTION_FILE_FMT.format(year=int(year))
    return resolve_input_file(configured, f"data_{int(year)}")


def build_underlying_filter(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for the configured underlying ticker or secid."""
    mask = pd.Series(False, index=df.index)
    if ASSET_SECID is not None and "secid" in df.columns:
        secid_num = pd.to_numeric(df["secid"], errors="coerce")
        mask = mask | secid_num.eq(ASSET_SECID)
    if "ticker" in df.columns:
        tick = df["ticker"].fillna("").astype(str).str.upper().str.strip()
        mask = mask | tick.eq(TICKER)
    if "root" in df.columns:
        root = df["root"].fillna("").astype(str).str.upper().str.strip()
        mask = mask | root.eq(TICKER) | root.str.startswith(TICKER)
    return mask.fillna(False)


def apply_contract_style_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply optional contract-style and settlement filters."""
    out = df.copy()
    if REQUIRE_EUROPEAN_STYLE and "exercise_style" in out.columns:
        style = out["exercise_style"].fillna("").astype(str).str.upper().str.strip()
        out = out[style.isin({"E", "EUROPEAN"})].copy()
    if SETTLEMENT_POLICY in {"am", "pm"} and "am_settlement" in out.columns:
        am_flag = pd.to_numeric(out["am_settlement"], errors="coerce")
        if SETTLEMENT_POLICY == "am":
            out = out[am_flag.eq(1)].copy()
        elif SETTLEMENT_POLICY == "pm":
            out = out[am_flag.eq(0)].copy()
    return out

def read_table(path: Path, usecols=None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        if usecols is not None:
            if callable(usecols):
                keep = [c for c in df.columns if usecols(c)]
            else:
                keep = [c for c in usecols if c in df.columns]
            df = df[keep]
        return df
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, usecols=usecols)
    else:
        try:
            return pd.read_csv(path, usecols=usecols, low_memory=False)
        except Exception:
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.read_excel(path, usecols=usecols)

def safe_to_datetime(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

def safe_to_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def load_fred_series(path: Path, value_name: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        candidates = sorted(BASE_DIR.glob(f"{path.stem}*"), key=lambda x: str(x))
        if not candidates:
            return None
        path = candidates[0]

    df = read_table(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "observation_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"observation_date": "date"})

    if "date" not in df.columns:
        return None

    value_candidates = [c for c in df.columns if c != "date"]
    if len(value_candidates) == 0:
        return None

    value_col = value_candidates[0]
    df = df.rename(columns={value_col: value_name})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    return df[["date", value_name]].dropna(subset=["date"])

def safe_nanquantile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.nanquantile(x, q))

def apply_var_floor(x: float, floor_value: Optional[float]) -> float:
    x = _safe_float(x) if "_safe_float" in globals() else float(x)
    if floor_value is None or not np.isfinite(floor_value):
        return x
    if not np.isfinite(x):
        return x
    return float(max(x, floor_value))


def build_route_a_thresholds(q_hat_base_raw: float,
                             b_t: float,
                             floor_value: Optional[float]) -> Tuple[float, float, float]:
    """
    Route A: make the reported/backtested threshold exactly match the theoretical object.

    q_ref      = max(q_base_raw, floor)
    q_core_op  = q_ref + b_t
    q_rep      = max(q_core_op, floor)
    """
    q_hat_ref = apply_var_floor(q_hat_base_raw, floor_value)
    q_hat_core_op = q_hat_ref + b_t
    q_hat_rep = apply_var_floor(q_hat_core_op, floor_value)
    return float(q_hat_ref), float(q_hat_core_op), float(q_hat_rep)

def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_w = np.cumsum(weights)

    if len(cum_w) == 0 or not np.isfinite(cum_w[-1]) or cum_w[-1] <= 0:
        return np.nan

    cutoff = q * cum_w[-1]
    idx = np.searchsorted(cum_w, cutoff, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])

def build_time_decay_weights(cal_items, pred_idx):
    w = []
    for x in cal_items:
        age = pred_idx - x["pred_idx"]
        w.append(np.exp(-TIME_DECAY * age))
    return np.asarray(w, dtype=float)

# ============================================================
# Classical VaR baseline helpers
# ============================================================
def should_compute_classical_baselines(experiment_group: str) -> bool:
    if not RUN_CLASSICAL_BASELINES:
        return False
    if CLASSICAL_BASELINE_GROUPS is None:
        return True
    return str(experiment_group) in set(CLASSICAL_BASELINE_GROUPS)


def _pinball_loss(y: np.ndarray, q: np.ndarray, tau: float = QUANTILE_LEVEL) -> float:
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    u = y - q
    loss = np.where(u >= 0.0, tau * u, (tau - 1.0) * u)
    return float(np.sum(loss[np.isfinite(loss)]))


def _pinball_loss_one(y: float, q: float, tau: float = QUANTILE_LEVEL) -> float:
    """One-observation upper-tail quantile check loss.

    For VaR with exceedance target alpha, tau=1-alpha. Lower is better conditional
    quantile efficiency, conditional on the same realized loss sample.
    """
    try:
        y = float(y)
        q = float(q)
    except Exception:
        return np.nan
    if (not np.isfinite(y)) or (not np.isfinite(q)):
        return np.nan
    u = y - q
    return float(tau * u if u >= 0.0 else (tau - 1.0) * u)


def ewma_historical_var(y_hist: np.ndarray,
                        quantile_level: float = QUANTILE_LEVEL,
                        lam: float = EWMA_LAMBDA) -> float:
    """Exponentially weighted historical upper quantile of realized book losses."""
    y = np.asarray(y_hist, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan
    # Most recent observation gets weight 1, older observations get lambda^age.
    age = np.arange(len(y) - 1, -1, -1, dtype=float)
    weights = np.power(float(lam), age)
    return weighted_quantile(y, weights, quantile_level)


def _caviar_filter(y: np.ndarray, params: np.ndarray, q0: Optional[float] = None) -> np.ndarray:
    """Symmetric absolute value CAViaR filter on the loss series."""
    y = np.asarray(y, dtype=float)
    beta0, beta1, beta2 = [float(x) for x in params]
    q = np.full(len(y), np.nan, dtype=float)
    if len(y) == 0:
        return q
    if q0 is None or not np.isfinite(q0):
        q0 = safe_nanquantile(y, QUANTILE_LEVEL)
        if not np.isfinite(q0):
            q0 = 0.0
    q[0] = float(q0)
    for i in range(1, len(y)):
        prev_q = q[i - 1] if np.isfinite(q[i - 1]) else q0
        prev_y = y[i - 1] if np.isfinite(y[i - 1]) else 0.0
        q[i] = beta0 + beta1 * prev_q + beta2 * abs(prev_y)
    return q


def fit_caviar_sav(y_train: np.ndarray) -> Optional[np.ndarray]:
    """Fit a simple CAViaR-SAV baseline by quantile loss minimization."""
    y = np.asarray(y_train, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < CAVIAR_MIN_OBS:
        return None

    qbar = safe_nanquantile(y, QUANTILE_LEVEL)
    if not np.isfinite(qbar):
        qbar = 0.0
    scale = float(np.nanstd(y)) if np.isfinite(np.nanstd(y)) and np.nanstd(y) > EPS else 1.0

    def objective(beta: np.ndarray) -> float:
        beta = np.asarray(beta, dtype=float)
        if not np.all(np.isfinite(beta)):
            return 1e12
        q = _caviar_filter(y, beta, q0=qbar)
        if not np.all(np.isfinite(q)):
            return 1e12
        # Mild penalty discourages explosive or implausibly jagged quantile paths.
        penalty = 1e-4 * float(np.sum(beta ** 2))
        return _pinball_loss(y, q, QUANTILE_LEVEL) + penalty

    starts = [
        np.array([0.05 * qbar, 0.85, 0.05], dtype=float),
        np.array([0.10 * qbar, 0.70, 0.10], dtype=float),
        np.array([0.00, 0.90, 0.05], dtype=float),
        np.array([0.05 * scale, 0.80, 0.10], dtype=float),
    ]
    bounds = [(-5.0 * scale, 5.0 * scale), (0.0, 0.999), (-5.0, 5.0)]

    best = None
    best_val = np.inf
    for x0 in starts:
        try:
            res = minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": CAVIAR_MAXITER, "ftol": 1e-9},
            )
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best = np.asarray(res.x, dtype=float)
        except Exception:
            continue
    return best


def predict_caviar_sav(y_hist: np.ndarray, params: Optional[np.ndarray]) -> float:
    """One-step-ahead CAViaR-SAV VaR forecast."""
    y = np.asarray(y_hist, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0 or params is None:
        return np.nan
    q0 = safe_nanquantile(y, QUANTILE_LEVEL)
    q = _caviar_filter(y, params, q0=q0)
    if len(q) == 0 or not np.isfinite(q[-1]):
        return np.nan
    beta0, beta1, beta2 = [float(x) for x in params]
    return float(beta0 + beta1 * q[-1] + beta2 * abs(y[-1]))


def _garch_t_filter(y: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return residuals and conditional scale^2 path for constant-mean GARCH(1,1)-t."""
    y = np.asarray(y, dtype=float)
    mu, omega, alpha, beta, nu = [float(x) for x in params]
    eps = y - mu
    n = len(y)
    sigma2 = np.full(n, np.nan, dtype=float)
    sample_var = float(np.nanvar(eps)) if np.isfinite(np.nanvar(eps)) and np.nanvar(eps) > EPS else 1.0
    sigma2[0] = sample_var
    for i in range(1, n):
        sigma2[i] = omega + alpha * eps[i - 1] ** 2 + beta * sigma2[i - 1]
        if (not np.isfinite(sigma2[i])) or sigma2[i] <= EPS:
            sigma2[i] = sample_var
    return eps, sigma2


def fit_garch_t(y_train: np.ndarray) -> Optional[np.ndarray]:
    """Fit a lightweight constant-mean GARCH(1,1)-Student-t VaR baseline."""
    y = np.asarray(y_train, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < GARCH_MIN_OBS:
        return None

    mu0 = float(np.nanmean(y))
    var0 = float(np.nanvar(y)) if np.isfinite(np.nanvar(y)) and np.nanvar(y) > EPS else 1.0
    y_scale = float(np.nanstd(y)) if np.isfinite(np.nanstd(y)) and np.nanstd(y) > EPS else 1.0

    def objective(theta: np.ndarray) -> float:
        mu, omega, alpha, beta, nu = [float(x) for x in theta]
        if omega <= EPS or alpha < 0 or beta < 0 or alpha + beta >= 0.999 or nu <= 2.05:
            return 1e12
        eps, sigma2 = _garch_t_filter(y, theta)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= EPS):
            return 1e12
        scale = np.sqrt(sigma2)
        z = eps / scale
        ll = student_t.logpdf(z, df=nu) - np.log(scale)
        if not np.all(np.isfinite(ll)):
            return 1e12
        return float(-np.sum(ll))

    starts = [
        np.array([mu0, 0.05 * var0, 0.05, 0.90, 8.0], dtype=float),
        np.array([mu0, 0.10 * var0, 0.08, 0.85, 6.0], dtype=float),
        np.array([safe_nanquantile(y, 0.50), 0.05 * var0, 0.03, 0.92, 10.0], dtype=float),
    ]
    bounds = [
        (mu0 - 5.0 * y_scale, mu0 + 5.0 * y_scale),
        (1e-8, 10.0 * var0 + 1e-8),
        (1e-6, 0.40),
        (1e-6, 0.998),
        (2.10, 50.0),
    ]
    best = None
    best_val = np.inf
    for x0 in starts:
        try:
            res = minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": GARCH_MAXITER, "ftol": 1e-8},
            )
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best = np.asarray(res.x, dtype=float)
        except Exception:
            continue
    return best


def predict_garch_t_var(y_hist: np.ndarray, params: Optional[np.ndarray]) -> float:
    """One-step-ahead upper-tail VaR forecast from fitted GARCH-t parameters."""
    y = np.asarray(y_hist, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0 or params is None:
        return np.nan
    mu, omega, alpha, beta, nu = [float(x) for x in params]
    eps, sigma2 = _garch_t_filter(y, params)
    if len(eps) == 0 or len(sigma2) == 0 or not np.isfinite(sigma2[-1]):
        return np.nan
    next_sigma2 = omega + alpha * eps[-1] ** 2 + beta * sigma2[-1]
    if not np.isfinite(next_sigma2) or next_sigma2 <= EPS:
        return np.nan
    return float(mu + np.sqrt(next_sigma2) * student_t.ppf(QUANTILE_LEVEL, df=nu))

# ============================================================
# Data containers
# ============================================================
@dataclass
class DailySurfaceObs:
    date: pd.Timestamp
    spot: float
    iv_surface: np.ndarray
    state_core: dict

@dataclass
class FeatureTransform:
    fill_values: np.ndarray
    scaler: StandardScaler

# ============================================================
# Feature scaling
# ============================================================
def fit_feature_transform(X: np.ndarray) -> Tuple[np.ndarray, FeatureTransform]:
    X = np.asarray(X, dtype=float)

    all_nan_cols = np.all(~np.isfinite(X), axis=0)
    fill_values = np.zeros(X.shape[1], dtype=float)

    valid_cols = ~all_nan_cols
    if np.any(valid_cols):
        fill_values[valid_cols] = np.nanmedian(X[:, valid_cols], axis=0)

    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
    X_filled = np.where(np.isfinite(X), X, fill_values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    return X_scaled, FeatureTransform(fill_values=fill_values, scaler=scaler)

def apply_feature_transform(X: np.ndarray, trans: FeatureTransform) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_filled = np.where(np.isfinite(X), X, trans.fill_values)
    return trans.scaler.transform(X_filled)

# ============================================================
# Auxiliary table loading
# ============================================================
def load_aux_tables():
    sec_path = resolve_input_file(SECURITY_PRICE_FILE, "security_prices")
    zero_path = resolve_input_file(ZCY_FILE, "zero_coupon_yield")
    div_path = resolve_input_file(DIVIDEND_FILE, "index_dividend_yield")

    sec = std_cols(read_table(sec_path))
    safe_to_datetime(sec, ["date"])
    safe_to_numeric(sec, ["secid", "close"])
    if "close" not in sec.columns:
        raise ValueError("security_prices file is missing the close column")
    sec = sec[["secid", "date", "close"]].copy()
    if ASSET_SECID is not None:
        sec = sec[pd.to_numeric(sec["secid"], errors="coerce").eq(ASSET_SECID)].copy()
    sec = sec.rename(columns={"close": "spot"})

    zero = std_cols(read_table(zero_path))
    safe_to_datetime(zero, ["date"])
    safe_to_numeric(zero, ["days", "rate"])
    zero = zero[["date", "days", "rate"]].copy()
    zero = zero.dropna(subset=["date", "days", "rate"]).copy()
    zero = zero.sort_values(["date", "days"]).drop_duplicates(["date", "days"], keep="last")

    div = std_cols(read_table(div_path))

    # Some ETF dividend-yield files are stored as an underlying-level panel
    # without a secid column. For QQQ, attach the configured secid so the
    # downstream merge on secid/date/exdate remains identical to the SPX path.
    if "expiration" not in div.columns and "exdate" in div.columns:
        div = div.rename(columns={"exdate": "expiration"})
    if "rate" not in div.columns:
        for candidate in ["div_yield", "dividend_yield", "yield", "q", "div_rate"]:
            if candidate in div.columns:
                div = div.rename(columns={candidate: "rate"})
                break
    if "secid" not in div.columns:
        if ASSET_SECID is None:
            raise ValueError("index_dividend_yield file has no secid column; set OPTION_SECID.")
        div["secid"] = ASSET_SECID

    safe_to_datetime(div, ["date", "expiration"])
    safe_to_numeric(div, ["secid", "rate"])
    if "expiration" not in div.columns:
        raise ValueError("index_dividend_yield file is missing the expiration/exdate column")
    if "rate" not in div.columns:
        raise ValueError("index_dividend_yield file is missing the rate/dividend-yield column")
    div = div[["secid", "date", "expiration", "rate"]].copy()
    if ASSET_SECID is not None:
        div = div[pd.to_numeric(div["secid"], errors="coerce").eq(ASSET_SECID)].copy()
    div = div.rename(columns={"expiration": "exdate", "rate": "div_yield"})

    vix = load_fred_series(VIX_FILE, "vix")
    vxv = load_fred_series(VXV_FILE, "vxv")

    return sec, zero, div, vix, vxv

def merge_zero_curve_nearest(opt_df: pd.DataFrame, zero_df: pd.DataFrame) -> pd.DataFrame:
    left = opt_df.copy()
    right = zero_df.copy()

    left = left.dropna(subset=["date", "dte"]).copy()
    right = right.dropna(subset=["date", "days", "rate"]).copy()

    left["dte"] = pd.to_numeric(left["dte"], errors="coerce")
    right["days"] = pd.to_numeric(right["days"], errors="coerce")
    right["rate"] = pd.to_numeric(right["rate"], errors="coerce")

    left = left.dropna(subset=["dte"]).copy()
    right = right.dropna(subset=["days", "rate"]).copy()

    out = []
    zero_by_date = {dt: g.sort_values("days").copy() for dt, g in right.groupby("date", sort=False)}

    for dt, g_left in left.groupby("date", sort=False):
        g_left = g_left.sort_values("dte").copy()
        g_right = zero_by_date.get(dt, None)

        if g_right is None or g_right.empty:
            g_left["rf_rate"] = np.nan
        else:
            merged = pd.merge_asof(
                g_left,
                g_right[["date", "days", "rate"]].sort_values("days"),
                left_on="dte",
                right_on="days",
                direction="nearest"
            )
            g_left["rf_rate"] = merged["rate"].to_numpy()

        out.append(g_left)

    if not out:
        left["rf_rate"] = np.nan
        return left

    merged_all = pd.concat(out, axis=0).sort_index()
    return merged_all

# ============================================================
# One-year option-chain loading and cleaning
# ============================================================
def load_and_clean_one_year(year_file: Path, sec_df, zero_df, div_df) -> pd.DataFrame:
    usecols = [
        "secid", "date", "symbol", "exdate", "last_date", "cp_flag",
        "strike_price", "best_bid", "best_offer", "volume", "open_interest",
        "impl_volatility", "delta", "gamma", "vega", "theta", "optionid",
        "contract_size", "ss_flag", "forward_price", "expiry_indicator",
        "root", "suffix", "ticker", "index_flag", "issuer",
        "div_convention", "exercise_style", "am_settlement", "am_set_flag"
    ]

    df = std_cols(read_table(
        year_file,
        usecols=lambda c: str(c).strip().lower() in usecols or str(c).strip() in usecols
    ))

    for col in ["ticker", "root", "forward_price", "volume", "open_interest",
                "delta", "vega", "optionid",
                "best_bid", "best_offer", "impl_volatility", "strike_price"]:
        if col not in df.columns:
            df[col] = np.nan

    safe_to_datetime(df, ["date", "exdate", "last_date"])
    safe_to_numeric(df, [
        "secid", "strike_price", "best_bid", "best_offer", "volume",
        "open_interest", "impl_volatility", "delta", "gamma", "vega",
        "theta", "optionid", "contract_size", "ss_flag", "forward_price",
        "index_flag", "am_settlement"
    ])

    asset_mask = build_underlying_filter(df)
    df = df[asset_mask].copy()
    df = apply_contract_style_filters(df)

    df["cp_flag"] = df["cp_flag"].astype(str).str.upper()
    df["strike"] = df["strike_price"] / 1000.0
    df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0
    df["dte"] = (df["exdate"] - df["date"]).dt.days
    df["tau"] = df["dte"] / 365.0

    df = df.merge(sec_df, on=["secid", "date"], how="left")
    df = df.merge(div_df, on=["secid", "date", "exdate"], how="left")
    df = merge_zero_curve_nearest(df, zero_df)

    df["div_yield"] = df["div_yield"].fillna(0.0)
    df["rf_rate"] = df["rf_rate"].fillna(0.0)

    df["forward_formula"] = df["spot"] * np.exp((df["rf_rate"] - df["div_yield"]) * df["tau"])
    if USE_VENDOR_FORWARD_PRICE:
        df["forward_used"] = np.where(
            (df["forward_price"].notna()) & (df["forward_price"] > 0),
            df["forward_price"],
            df["forward_formula"]
        )
    else:
        df["forward_used"] = df["forward_formula"]

    df["k"] = np.log(df["strike"] / df["forward_used"])

    df = df[df["date"].notna() & df["exdate"].notna()]
    df = df[df["cp_flag"].isin(["C", "P"])]
    df = df[df["spot"].notna() & (df["spot"] > 0)]
    df = df[df["dte"].between(MIN_DTE, MAX_DTE)]
    df = df[df["best_bid"] > MIN_BID]
    df = df[df["best_offer"] > df["best_bid"]]
    df = df[df["mid"] > MIN_MID]
    df = df[df["k"].between(K_MIN, K_MAX)]
    df = df[df["impl_volatility"].notna() & (df["impl_volatility"] > 0)]

    rel_spread = (df["best_offer"] - df["best_bid"]) / np.maximum(df["mid"], 1e-8)
    df = df[
        (rel_spread <= MAX_REL_SPREAD) &
        (((df["open_interest"].fillna(0) >= MIN_OI) | (df["volume"].fillna(0) >= MIN_VOL)))
    ].copy()

    df["optionid_key"] = pd.to_numeric(df["optionid"], errors="coerce").round().astype("Int64")

    return df.sort_values(["date", "exdate", "strike", "cp_flag"]).reset_index(drop=True)

# ============================================================
# State features
# ============================================================
def nearest_value_by_target(sub: pd.DataFrame, value_col: str, target_dte=None, target_k=None, target_delta=None, cp=None):
    x = sub.copy()
    if cp is not None and "cp_flag" in x.columns:
        x = x[x["cp_flag"] == cp]
    x = x.dropna(subset=[value_col])
    if x.empty:
        return np.nan

    dist = np.zeros(len(x), dtype=float)
    if target_dte is not None and "dte" in x.columns:
        dist += np.abs(x["dte"].to_numpy() - target_dte) * 10.0
    if target_k is not None and "k" in x.columns:
        dist += np.abs(x["k"].to_numpy() - target_k)
    if target_delta is not None and "delta" in x.columns:
        dist += np.abs(x["delta"].to_numpy() - target_delta) * 5.0

    idx = np.argmin(dist)
    return float(x.iloc[idx][value_col])

def build_daily_obs(day_df: pd.DataFrame) -> Optional[DailySurfaceObs]:
    use = day_df.dropna(subset=["k", "tau", "impl_volatility", "spot"]).copy()
    if len(use) < 30:
        return None

    try:
        KK = np.linspace(SURFACE_K_MIN, SURFACE_K_MAX, 15)
        TT = np.array([14, 21, 30, 45, 60, 75, 90, 120], dtype=float) / 365.0
        G1, G2 = np.meshgrid(KK, TT, indexing="xy")
        pts = np.column_stack([use["k"].to_numpy(), use["tau"].to_numpy()])
        z_lin = griddata(pts, use["impl_volatility"].to_numpy(), (G1, G2), method="linear")
        z_near = griddata(pts, use["impl_volatility"].to_numpy(), (G1, G2), method="nearest")
        iv_surface = np.where(np.isnan(z_lin), z_near, z_lin)
    except Exception:
        return None

    dt = use["date"].iloc[0]
    spot = float(use["spot"].iloc[0])

    atm30 = nearest_value_by_target(use, "impl_volatility", target_dte=30, target_k=0.0)
    atm60 = nearest_value_by_target(use, "impl_volatility", target_dte=60, target_k=0.0)
    atm90 = nearest_value_by_target(use, "impl_volatility", target_dte=90, target_k=0.0)

    p25_30 = nearest_value_by_target(use, "impl_volatility", target_dte=30, target_delta=-0.25, cp="P")
    c25_30 = nearest_value_by_target(use, "impl_volatility", target_dte=30, target_delta=0.25, cp="C")
    p25_60 = nearest_value_by_target(use, "impl_volatility", target_dte=60, target_delta=-0.25, cp="P")
    c25_60 = nearest_value_by_target(use, "impl_volatility", target_dte=60, target_delta=0.25, cp="C")
    p25_90 = nearest_value_by_target(use, "impl_volatility", target_dte=90, target_delta=-0.25, cp="P")
    c25_90 = nearest_value_by_target(use, "impl_volatility", target_dte=90, target_delta=0.25, cp="C")

    left30 = nearest_value_by_target(use, "impl_volatility", target_dte=30, target_k=-0.15)
    center30 = nearest_value_by_target(use, "impl_volatility", target_dte=30, target_k=0.0)
    right30 = nearest_value_by_target(use, "impl_volatility", target_dte=30, target_k=0.10)

    state_core = {
        "date": dt,
        "spot": spot,
        "atm30": atm30,
        "atm60": atm60,
        "atm90": atm90,
        "skew30": p25_30 - c25_30 if np.isfinite(p25_30) and np.isfinite(c25_30) else np.nan,
        "skew60": p25_60 - c25_60 if np.isfinite(p25_60) and np.isfinite(c25_60) else np.nan,
        "skew90": p25_90 - c25_90 if np.isfinite(p25_90) and np.isfinite(c25_90) else np.nan,
        "term30_90": atm30 - atm90 if np.isfinite(atm30) and np.isfinite(atm90) else np.nan,
        "left_minus_center30": left30 - center30 if np.isfinite(left30) and np.isfinite(center30) else np.nan,
        "curvature30": (left30 + right30 - 2.0 * center30) if np.isfinite(left30) and np.isfinite(right30) and np.isfinite(center30) else np.nan,
        "avg_oi": float(use["open_interest"].mean()),
        "avg_vol": float(use["volume"].mean()),
        "avg_spread": float((use["best_offer"] - use["best_bid"]).mean()),
    }

    return DailySurfaceObs(
        date=dt,
        spot=spot,
        iv_surface=iv_surface,
        state_core=state_core
    )

# ============================================================
# Collect the state panel and daily compact option chains
# ============================================================
def collect_state_and_daily_chains():
    sec_df, zero_df, div_df, vix_df, vxv_df = load_aux_tables()
    vix_map = dict(zip(vix_df["date"], vix_df["vix"])) if vix_df is not None else {}
    vxv_map = dict(zip(vxv_df["date"], vxv_df["vxv"])) if vxv_df is not None else {}

    obs_list = []
    daily_chain_map: Dict[pd.Timestamp, pd.DataFrame] = {}

    keep_chain_cols = [
        "date", "exdate", "cp_flag", "strike", "dte", "tau", "k",
        "delta", "vega", "mid", "impl_volatility",
        "open_interest", "volume", "best_bid", "best_offer",
        "spot", "optionid_key"
    ]

    for year in YEAR_LIST:
        year_file = resolve_option_year_file(year)
        print(f"Reading: {year_file.name}")
        year_df = load_and_clean_one_year(year_file, sec_df, zero_df, div_df)

        for dt, day_df in year_df.groupby("date", sort=True):
            day_df = day_df.copy()

            obj = build_daily_obs(day_df)
            if obj is not None:
                obs_list.append(obj)

            compact = day_df[keep_chain_cols].copy()
            compact = compact.dropna(subset=["mid", "optionid_key", "cp_flag", "exdate", "strike", "dte"])
            if len(compact) > 0:
                compact = compact.sort_values(["optionid_key", "mid"], ascending=[True, False])
                compact = compact.drop_duplicates(["optionid_key"], keep="first").reset_index(drop=True)
            daily_chain_map[pd.Timestamp(dt)] = compact

    obs_list = sorted(obs_list, key=lambda x: x.date)
    if len(obs_list) == 0:
        raise ValueError("no valid daily states were constructed.")

    feat_df = pd.DataFrame([o.state_core for o in obs_list]).sort_values("date").reset_index(drop=True)

    feat_df["vix"] = feat_df["date"].map(vix_map)
    feat_df["vxv"] = feat_df["date"].map(vxv_map)
    feat_df["vix_minus_vxv"] = feat_df["vix"] - feat_df["vxv"]

    feat_df["log_spot"] = np.log(feat_df["spot"])
    feat_df["ret"] = feat_df["log_spot"].diff()
    feat_df["abs_ret"] = feat_df["ret"].abs()
    feat_df["rv5"] = feat_df["ret"].rolling(5).std() * np.sqrt(252)
    feat_df["rv21"] = feat_df["ret"].rolling(21).std() * np.sqrt(252)
    feat_df["drawdown60"] = feat_df["spot"] / feat_df["spot"].rolling(60, min_periods=1).max() - 1.0
    feat_df["downside_semivar21"] = np.minimum(feat_df["ret"], 0.0) ** 2
    feat_df["downside_semivar21"] = feat_df["downside_semivar21"].rolling(21).mean() * 252

    # V2 features: changes and jump proxy, useful for the risk-reversal book
    feat_df["atm30_chg1"] = feat_df["atm30"].diff()
    feat_df["skew30_chg1"] = feat_df["skew30"].diff()
    feat_df["term30_90_chg1"] = feat_df["term30_90"].diff()
    feat_df["vix_chg1"] = feat_df["vix"].diff()
    feat_df["vxv_chg1"] = feat_df["vxv"].diff()
    feat_df["vix_minus_vxv_chg1"] = feat_df["vix_minus_vxv"].diff()
    feat_df["jump_proxy"] = feat_df["abs_ret"] / feat_df["rv21"].replace(0.0, np.nan)

    state_cols = [
        "atm30", "atm60", "atm90",
        "skew30", "skew60", "skew90",
        "term30_90", "left_minus_center30", "curvature30",
        "avg_oi", "avg_vol", "avg_spread",
        "vix", "vxv", "vix_minus_vxv",
        "ret", "abs_ret", "rv5", "rv21", "drawdown60", "downside_semivar21",
        "atm30_chg1", "skew30_chg1", "term30_90_chg1",
        "vix_chg1", "vxv_chg1", "vix_minus_vxv_chg1",
        "jump_proxy",
    ]

    return feat_df, state_cols, daily_chain_map


def diagnose_filter_attrition_one_year(year: int, sec_df, zero_df, div_df) -> pd.DataFrame:
    year_file = resolve_option_year_file(year)
    df = std_cols(read_table(
        year_file,
        usecols=lambda c: str(c).strip().lower() in DIAG_USECOLS or str(c).strip() in DIAG_USECOLS
    ))

    for col in [
        "ticker", "root", "forward_price", "volume", "open_interest",
        "delta", "vega", "optionid", "best_bid", "best_offer",
        "impl_volatility", "strike_price", "am_settlement"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    safe_to_datetime(df, ["date", "exdate", "last_date"])
    safe_to_numeric(df, [
        "secid", "strike_price", "best_bid", "best_offer", "volume",
        "open_interest", "impl_volatility", "delta", "gamma", "vega",
        "theta", "optionid", "contract_size", "ss_flag", "forward_price",
        "index_flag", "am_settlement"
    ])

    df["flag_underlying"] = build_underlying_filter(df)
    df = apply_contract_style_filters(df)

    df["cp_flag"] = df["cp_flag"].astype(str).str.upper()
    df["strike"] = df["strike_price"] / 1000.0
    df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0

    sec_small = sec_df[["secid", "date", "spot"]].drop_duplicates(["secid", "date"], keep="last")
    div_small = div_df[["secid", "date", "exdate", "div_yield"]].drop_duplicates(["secid", "date", "exdate"], keep="last")

    df = df.merge(sec_small, on=["secid", "date"], how="left")
    df = df.merge(div_small, on=["secid", "date", "exdate"], how="left")

    # Compute DTE before merging the zero-coupon curve
    df["dte"] = (df["exdate"] - df["date"]).dt.days
    df = merge_zero_curve_nearest(df, zero_df)

    df["tau"] = df["dte"] / 365.0
    df["div_yield"] = df["div_yield"].fillna(0.0)
    df["rf_rate"] = df["rf_rate"].fillna(0.0)

    df["forward_formula"] = df["spot"] * np.exp((df["rf_rate"] - df["div_yield"]) * df["tau"])
    if USE_VENDOR_FORWARD_PRICE:
        df["forward_used"] = np.where(
            (df["forward_price"].notna()) & (df["forward_price"] > 0),
            df["forward_price"],
            df["forward_formula"]
        )
    else:
        df["forward_used"] = df["forward_formula"]
    df["k"] = np.log(df["strike"] / df["forward_used"])

    rel_spread = (df["best_offer"] - df["best_bid"]) / np.maximum(df["mid"], 1e-8)

    stage_flags = {
        "underlying": df["flag_underlying"],
        "date_exdate_ok": df["date"].notna() & df["exdate"].notna(),
        "cp_ok": df["cp_flag"].isin(["C", "P"]),
        "spot_ok": df["spot"].notna() & (df["spot"] > 0),
        "dte_window": df["dte"].between(MIN_DTE, MAX_DTE),
        "bid_positive": df["best_bid"] > MIN_BID,
        "offer_gt_bid": df["best_offer"] > df["best_bid"],
        "mid_gt_min": df["mid"] > MIN_MID,
        "forward_ok": np.isfinite(df["forward_used"]) & (df["forward_used"] > 0),
        "k_window": df["k"].between(K_MIN, K_MAX),
        "iv_positive": df["impl_volatility"].notna() & (df["impl_volatility"] > 0),
        "spread_ok": rel_spread <= MAX_REL_SPREAD,
        "activity_ok": (
            (df["open_interest"].fillna(0) >= MIN_OI) |
            (df["volume"].fillna(0) >= MIN_VOL)
        ),
    }

    rows = []
    cumulative = np.ones(len(df), dtype=bool)
    prev_n = len(df)

    for stage_name, stage_mask in stage_flags.items():
        stage_mask = pd.Series(stage_mask).fillna(False).to_numpy(dtype=bool)
        drop_here = cumulative & (~stage_mask)
        cumulative = cumulative & stage_mask
        curr_n = int(cumulative.sum())

        rows.append({
            "year": year,
            "stage": stage_name,
            "n_raw": int(len(df)),
            "n_after_stage": curr_n,
            "share_of_raw": curr_n / max(len(df), 1),
            "dropped_at_stage": int(drop_here.sum()),
            "drop_rate_from_prev": (prev_n - curr_n) / max(prev_n, 1),
        })
        prev_n = curr_n

    return pd.DataFrame(rows)


def build_yearly_filter_attrition_report() -> pd.DataFrame:
    sec_df, zero_df, div_df, _, _ = load_aux_tables()
    all_rows = []
    for year in YEAR_LIST:
        print(f"Building filter-attrition diagnostics: {year}")
        all_rows.append(diagnose_filter_attrition_one_year(year, sec_df, zero_df, div_df))

    out = pd.concat(all_rows, axis=0, ignore_index=True)
    out_path = OUT_DIR / "book_var_filter_attrition_by_year.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
    return out


def diagnose_one_transition(day_t: pd.DataFrame,
                            day_t1: pd.DataFrame,
                            book_type: str,
                            marking_mode: str,
                            book_quality_mode: str = PRIMARY_BOOK_QUALITY_MODE) -> dict:
    date_t = pd.to_datetime(day_t["date"].iloc[0], errors="coerce")
    date_t1 = pd.to_datetime(day_t1["date"].iloc[0], errors="coerce")

    try:
        legs = build_standard_book(day_t, book_type=book_type, book_quality_mode=book_quality_mode)
    except Exception as e:
        return {
            "date": date_t,
            "next_date": date_t1,
            "book_type": book_type,
            "marking_mode": marking_mode,
            "book_quality_mode": book_quality_mode,
            "build_ok": 0,
            "mark_ok": 0,
            "reason": f"build_exception:{type(e).__name__}",
        }

    if legs is None or len(legs) == 0:
        return {
            "date": date_t,
            "next_date": date_t1,
            "book_type": book_type,
            "marking_mode": marking_mode,
            "book_quality_mode": book_quality_mode,
            "build_ok": 0,
            "mark_ok": 0,
            "reason": "build_standard_book_failed",
        }

    gap_days = _calendar_gap_days(day_t, day_t1)

    for _, leg in legs.iterrows():
        inst = str(leg.get("instrument_type", "option"))
        if inst == "option":
            mid_next, mark_source = lookup_next_option_mid(
                day_t1, leg, gap_days, marking_mode=marking_mode
            )
            if not np.isfinite(mid_next):
                return {
                    "date": date_t,
                    "next_date": date_t1,
                    "book_type": book_type,
                    "marking_mode": marking_mode,
                    "build_ok": 1,
                    "mark_ok": 0,
                    "reason": str(mark_source),
                }

    return {
        "date": date_t,
        "next_date": date_t1,
        "book_type": book_type,
        "marking_mode": marking_mode,
        "build_ok": 1,
        "mark_ok": 1,
        "reason": "success",
    }


def build_book_marking_diagnostics(daily_chain_map: Dict[pd.Timestamp, pd.DataFrame]):
    all_dates = sorted(daily_chain_map.keys())
    rows = []

    for i in range(len(all_dates) - 1):
        d0, d1 = all_dates[i], all_dates[i + 1]
        day_t = daily_chain_map.get(d0, None)
        day_t1 = daily_chain_map.get(d1, None)

        if day_t is None or day_t1 is None or len(day_t) == 0 or len(day_t1) == 0:
            continue

        for book_type in BOOK_TYPES:
            for mm in MARKING_SENSITIVITY_MODES:
                rows.append(diagnose_one_transition(day_t, day_t1, book_type, mm))

    diag = pd.DataFrame(rows)
    if len(diag) == 0:
        return pd.DataFrame(), pd.DataFrame()

    diag["year"] = pd.to_datetime(diag["date"]).dt.year
    summary = (
        diag.groupby(["year", "book_type", "marking_mode", "build_ok", "mark_ok", "reason"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["year", "book_type", "marking_mode", "n"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

    diag_path = OUT_DIR / "book_var_marking_diag_dates.csv"
    summary_path = OUT_DIR / "book_var_marking_diag_by_year_reason.csv"
    diag.to_csv(diag_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved to: {diag_path}")
    print(f"Saved to: {summary_path}")
    return diag, summary

# ============================================================
# Standardized book construction
# ============================================================
def choose_target_expiry(day_df: pd.DataFrame, target_dte: int = TARGET_DTE) -> Optional[pd.Timestamp]:
    x = day_df.dropna(subset=["exdate", "dte"]).copy()
    if x.empty:
        return None
    exp = x[["exdate", "dte"]].drop_duplicates().copy()
    exp["score"] = np.abs(exp["dte"] - target_dte)
    exp = exp.sort_values(["score", "dte", "exdate"]).reset_index(drop=True)
    return pd.Timestamp(exp.loc[0, "exdate"])

def build_atm_straddle(day_df: pd.DataFrame, target_dte: int = TARGET_DTE) -> Optional[pd.DataFrame]:
    exdate = choose_target_expiry(day_df, target_dte=target_dte)
    if exdate is None:
        return None

    sub = day_df[day_df["exdate"] == exdate].copy()
    if len(sub) == 0:
        return None

    calls = sub[sub["cp_flag"] == "C"].copy()
    puts = sub[sub["cp_flag"] == "P"].copy()
    if calls.empty or puts.empty:
        return None

    calls = calls.sort_values("mid", ascending=False).drop_duplicates(["strike"], keep="first")
    puts = puts.sort_values("mid", ascending=False).drop_duplicates(["strike"], keep="first")

    pairs = calls.merge(
        puts,
        on="strike",
        suffixes=("_c", "_p")
    )
    if pairs.empty:
        return None

    pairs["score"] = np.abs(pairs["k_c"]) + np.abs(pairs["k_p"])
    pairs = pairs.sort_values(["score", "dte_c", "strike"]).reset_index(drop=True)
    best = pairs.iloc[0]

    legs = pd.DataFrame([
        {
            "book_type": "atm_straddle_30d",
            "leg_name": "call",
            "instrument_type": "option",
            "weight": 1.0,
            "optionid_key": best["optionid_key_c"],
            "mid_t": _safe_float(best["mid_c"]),
            "delta_t": _safe_float(best["delta_c"]),
            "vega_t": _safe_float(best["vega_c"]),
            "iv_t": _safe_float(best["impl_volatility_c"]),
            "dte_t": _safe_float(best["dte_c"]),
            "strike_t": _safe_float(best["strike"]),
            "k_t": _safe_float(best["k_c"]),
            "cp_flag_t": "C",
            "exdate_t": pd.to_datetime(best["exdate_c"], errors="coerce"),
        },
        {
            "book_type": "atm_straddle_30d",
            "leg_name": "put",
            "instrument_type": "option",
            "weight": 1.0,
            "optionid_key": best["optionid_key_p"],
            "mid_t": _safe_float(best["mid_p"]),
            "delta_t": _safe_float(best["delta_p"]),
            "vega_t": _safe_float(best["vega_p"]),
            "iv_t": _safe_float(best["impl_volatility_p"]),
            "dte_t": _safe_float(best["dte_p"]),
            "strike_t": _safe_float(best["strike"]),
            "k_t": _safe_float(best["k_p"]),
            "cp_flag_t": "P",
            "exdate_t": pd.to_datetime(best["exdate_p"], errors="coerce"),
        }
    ])
    return legs

def build_risk_reversal(day_df: pd.DataFrame, target_dte: int = TARGET_DTE) -> Optional[pd.DataFrame]:
    exdate = choose_target_expiry(day_df, target_dte=target_dte)
    if exdate is None:
        return None

    sub = day_df[day_df["exdate"] == exdate].copy()
    if len(sub) == 0:
        return None

    calls = sub[sub["cp_flag"] == "C"].dropna(subset=["delta", "mid"]).copy()
    puts = sub[sub["cp_flag"] == "P"].dropna(subset=["delta", "mid"]).copy()
    if calls.empty or puts.empty:
        return None

    calls["score"] = np.abs(calls["delta"] - RR_CALL_TARGET_DELTA)
    puts["score"] = np.abs(puts["delta"] - RR_PUT_TARGET_DELTA)

    call = calls.sort_values(["score", "dte", "k"]).iloc[0]
    put = puts.sort_values(["score", "dte", "k"]).iloc[0]

    delta_c = _safe_float(call["delta"])
    delta_p = _safe_float(put["delta"])
    vega_c = _safe_float(call["vega"])
    vega_p = _safe_float(put["vega"])

    if RR_OPTION_SCALING == "unit":
        w_call = 1.0
        w_put = -1.0
    elif RR_OPTION_SCALING == "vega_normalized":
        if (not np.isfinite(vega_c)) or (not np.isfinite(vega_p)) or (abs(vega_c) <= EPS) or (abs(vega_p) <= EPS):
            return None
        w_call = 1.0 / abs(vega_c)
        w_put = -1.0 / abs(vega_p)
    else:
        raise ValueError(f"unknown RR_OPTION_SCALING: {RR_OPTION_SCALING}")

    option_net_delta = w_call * delta_c + w_put * delta_p
    if not np.isfinite(option_net_delta):
        return None

    spot_t = pd.to_numeric(sub["spot"], errors="coerce").dropna()
    if len(spot_t) == 0:
        return None
    spot_t = float(spot_t.iloc[0])

    stock_hedge_weight = -option_net_delta if RR_INCLUDE_SPOT_HEDGE else 0.0

    legs = [
        {
            "book_type": "risk_reversal_25d_30d",
            "leg_name": "call_25d",
            "instrument_type": "option",
            "weight": float(w_call),
            "optionid_key": call["optionid_key"],
            "mid_t": _safe_float(call["mid"]),
            "delta_t": delta_c,
            "vega_t": vega_c,
            "iv_t": _safe_float(call["impl_volatility"]),
            "dte_t": _safe_float(call["dte"]),
            "strike_t": _safe_float(call["strike"]),
            "k_t": _safe_float(call["k"]),
            "cp_flag_t": "C",
            "exdate_t": pd.to_datetime(call["exdate"], errors="coerce"),
        },
        {
            "book_type": "risk_reversal_25d_30d",
            "leg_name": "put_25d",
            "instrument_type": "option",
            "weight": float(w_put),
            "optionid_key": put["optionid_key"],
            "mid_t": _safe_float(put["mid"]),
            "delta_t": delta_p,
            "vega_t": vega_p,
            "iv_t": _safe_float(put["impl_volatility"]),
            "dte_t": _safe_float(put["dte"]),
            "strike_t": _safe_float(put["strike"]),
            "k_t": _safe_float(put["k"]),
            "cp_flag_t": "P",
            "exdate_t": pd.to_datetime(put["exdate"], errors="coerce"),
        }
    ]

    if RR_INCLUDE_SPOT_HEDGE:
        legs.append({
            "book_type": "risk_reversal_25d_30d",
            "leg_name": "spot_hedge",
            "instrument_type": "spot",
            "weight": float(stock_hedge_weight),
            "optionid_key": pd.NA,
            "mid_t": float(spot_t),
            "delta_t": 1.0,
            "vega_t": 0.0,
            "iv_t": np.nan,
            "dte_t": 0.0,
            "strike_t": np.nan,
            "k_t": 0.0,
            "cp_flag_t": pd.NA,
            "exdate_t": pd.NaT,
        })

    return pd.DataFrame(legs)

def build_short_put_spread(day_df: pd.DataFrame, target_dte: int = TARGET_DTE) -> Optional[pd.DataFrame]:
    exdate = choose_target_expiry(day_df, target_dte=target_dte)
    if exdate is None:
        return None

    sub = day_df[day_df["exdate"] == exdate].copy()
    if len(sub) == 0:
        return None

    puts = sub[sub["cp_flag"] == "P"].dropna(subset=["delta", "mid"]).copy()
    if puts.empty:
        return None

    puts["short_score"] = np.abs(puts["delta"] - SHORT_PUT_SPREAD_SHORT_DELTA)
    short_put = puts.sort_values(["short_score", "dte", "k"]).iloc[0]

    long_candidates = puts[puts["optionid_key"] != short_put["optionid_key"]].copy()
    long_candidates = long_candidates[long_candidates["strike"] < short_put["strike"]].copy()
    if long_candidates.empty:
        return None

    long_candidates["long_score"] = np.abs(long_candidates["delta"] - SHORT_PUT_SPREAD_LONG_DELTA)
    long_put = long_candidates.sort_values(["long_score", "strike", "dte"]).iloc[0]

    delta_short = _safe_float(short_put["delta"])
    delta_long = _safe_float(long_put["delta"])
    vega_short = _safe_float(short_put["vega"])
    vega_long = _safe_float(long_put["vega"])

    w_short = -1.0
    w_long = 1.0
    option_net_delta = w_short * delta_short + w_long * delta_long
    if not np.isfinite(option_net_delta):
        return None

    spot_t = pd.to_numeric(sub["spot"], errors="coerce").dropna()
    if len(spot_t) == 0:
        return None
    spot_t = float(spot_t.iloc[0])

    stock_hedge_weight = -option_net_delta if SHORT_PUT_SPREAD_INCLUDE_SPOT_HEDGE else 0.0

    legs = [
        {
            "book_type": SHORT_PUT_SPREAD_BOOK_TYPE,
            "leg_name": "short_put_25delta",
            "instrument_type": "option",
            "weight": float(w_short),
            "optionid_key": short_put["optionid_key"],
            "mid_t": _safe_float(short_put["mid"]),
            "delta_t": delta_short,
            "vega_t": vega_short,
            "iv_t": _safe_float(short_put["impl_volatility"]),
            "dte_t": _safe_float(short_put["dte"]),
            "strike_t": _safe_float(short_put["strike"]),
            "k_t": _safe_float(short_put["k"]),
            "cp_flag_t": "P",
            "exdate_t": pd.to_datetime(short_put["exdate"], errors="coerce"),
        },
        {
            "book_type": SHORT_PUT_SPREAD_BOOK_TYPE,
            "leg_name": "long_put_10delta",
            "instrument_type": "option",
            "weight": float(w_long),
            "optionid_key": long_put["optionid_key"],
            "mid_t": _safe_float(long_put["mid"]),
            "delta_t": delta_long,
            "vega_t": vega_long,
            "iv_t": _safe_float(long_put["impl_volatility"]),
            "dte_t": _safe_float(long_put["dte"]),
            "strike_t": _safe_float(long_put["strike"]),
            "k_t": _safe_float(long_put["k"]),
            "cp_flag_t": "P",
            "exdate_t": pd.to_datetime(long_put["exdate"], errors="coerce"),
        },
    ]

    if SHORT_PUT_SPREAD_INCLUDE_SPOT_HEDGE:
        legs.append({
            "book_type": SHORT_PUT_SPREAD_BOOK_TYPE,
            "leg_name": "spot_hedge",
            "instrument_type": "spot",
            "weight": float(stock_hedge_weight),
            "optionid_key": pd.NA,
            "mid_t": float(spot_t),
            "delta_t": 1.0,
            "vega_t": 0.0,
            "iv_t": np.nan,
            "dte_t": 0.0,
            "strike_t": np.nan,
            "k_t": 0.0,
            "cp_flag_t": pd.NA,
            "exdate_t": pd.NaT,
        })

    return pd.DataFrame(legs)

def diagnose_book_selection_quality(legs: pd.DataFrame, book_type: str) -> dict:
    """Leg-level diagnostics for checking whether the selected book is economically close to target.

    This is deliberately separate from the construction rule. The main experiment keeps the
    original nearest-feasible construction. The strict robustness experiment filters using these
    diagnostics so reviewer objections about book identity can be answered directly.
    """
    out = {c: np.nan for c in BOOK_QUALITY_OUTPUT_COLS if c != "book_quality_mode"}
    if legs is None or len(legs) == 0:
        out["quality_pass_strict_economic"] = 0
        return out

    x = legs.copy()
    if "instrument_type" not in x.columns:
        x["instrument_type"] = "option"
    opt = x[x["instrument_type"].astype(str).eq("option")].copy()
    if len(opt) == 0:
        out["quality_pass_strict_economic"] = 0
        return out

    abs_k = pd.to_numeric(opt.get("k_t", np.nan), errors="coerce").abs().to_numpy(dtype=float)
    dte = pd.to_numeric(opt.get("dte_t", np.nan), errors="coerce").to_numpy(dtype=float)
    out["book_quality_max_abs_k_t"] = float(np.nanmax(abs_k)) if np.isfinite(abs_k).any() else np.nan
    out["book_quality_mean_abs_k_t"] = float(np.nanmean(abs_k)) if np.isfinite(abs_k).any() else np.nan
    dte_err = np.abs(dte - float(TARGET_DTE))
    out["book_quality_max_abs_dte_error_t"] = float(np.nanmax(dte_err)) if np.isfinite(dte_err).any() else np.nan

    def _get_leg(name: str) -> Optional[pd.Series]:
        g = x[x["leg_name"].astype(str).eq(name)] if "leg_name" in x.columns else pd.DataFrame()
        if len(g) == 0:
            return None
        return g.iloc[0]

    if book_type == "atm_straddle_30d":
        call = _get_leg("call")
        put = _get_leg("put")
        out["atm_call_abs_k_t"] = abs(_safe_float(call.get("k_t"))) if call is not None else np.nan
        out["atm_put_abs_k_t"] = abs(_safe_float(put.get("k_t"))) if put is not None else np.nan
        vals = [out["atm_call_abs_k_t"], out["atm_put_abs_k_t"]]
        vals = [v for v in vals if np.isfinite(v)]
        out["atm_max_abs_k_t"] = float(max(vals)) if vals else np.nan

    elif book_type == "risk_reversal_25d_30d":
        call = _get_leg("call_25d")
        put = _get_leg("put_25d")
        out["rr_call_delta_error_t"] = abs(_safe_float(call.get("delta_t")) - RR_CALL_TARGET_DELTA) if call is not None else np.nan
        out["rr_put_delta_error_t"] = abs(_safe_float(put.get("delta_t")) - RR_PUT_TARGET_DELTA) if put is not None else np.nan
        vals = [out["rr_call_delta_error_t"], out["rr_put_delta_error_t"]]
        vals = [v for v in vals if np.isfinite(v)]
        out["rr_max_delta_error_t"] = float(max(vals)) if vals else np.nan

    elif book_type == SHORT_PUT_SPREAD_BOOK_TYPE:
        short_put = _get_leg("short_put_25delta")
        long_put = _get_leg("long_put_10delta")
        out["putspread_short_delta_error_t"] = abs(_safe_float(short_put.get("delta_t")) - SHORT_PUT_SPREAD_SHORT_DELTA) if short_put is not None else np.nan
        out["putspread_long_delta_error_t"] = abs(_safe_float(long_put.get("delta_t")) - SHORT_PUT_SPREAD_LONG_DELTA) if long_put is not None else np.nan
        vals = [out["putspread_short_delta_error_t"], out["putspread_long_delta_error_t"]]
        vals = [v for v in vals if np.isfinite(v)]
        out["putspread_max_delta_error_t"] = float(max(vals)) if vals else np.nan

    out["quality_pass_strict_economic"] = int(is_book_selection_quality_ok(out, book_type, BOOK_QUALITY_MODE_STRICT))
    return out


def is_book_selection_quality_ok(quality: dict, book_type: str, book_quality_mode: str) -> bool:
    mode = str(book_quality_mode)
    if mode == BOOK_QUALITY_MODE_LOOSE:
        return True
    if mode != BOOK_QUALITY_MODE_STRICT:
        raise ValueError(f"unknown book_quality_mode: {book_quality_mode}")

    dte_ok = (
        np.isfinite(quality.get("book_quality_max_abs_dte_error_t", np.nan)) and
        quality.get("book_quality_max_abs_dte_error_t", np.inf) <= STRICT_MAX_ABS_DTE_ERROR
    )
    if not dte_ok:
        return False

    if book_type == "atm_straddle_30d":
        return (
            np.isfinite(quality.get("atm_max_abs_k_t", np.nan)) and
            quality.get("atm_max_abs_k_t", np.inf) <= STRICT_ATM_MAX_ABS_K
        )
    if book_type == "risk_reversal_25d_30d":
        return (
            np.isfinite(quality.get("rr_call_delta_error_t", np.nan)) and
            np.isfinite(quality.get("rr_put_delta_error_t", np.nan)) and
            quality.get("rr_call_delta_error_t", np.inf) <= STRICT_25D_DELTA_MAX_ERR and
            quality.get("rr_put_delta_error_t", np.inf) <= STRICT_25D_DELTA_MAX_ERR
        )
    if book_type == SHORT_PUT_SPREAD_BOOK_TYPE:
        return (
            np.isfinite(quality.get("putspread_short_delta_error_t", np.nan)) and
            np.isfinite(quality.get("putspread_long_delta_error_t", np.nan)) and
            quality.get("putspread_short_delta_error_t", np.inf) <= STRICT_25D_DELTA_MAX_ERR and
            quality.get("putspread_long_delta_error_t", np.inf) <= STRICT_10D_DELTA_MAX_ERR
        )
    return False


def build_standard_book(day_df: pd.DataFrame, book_type: str, book_quality_mode: str = PRIMARY_BOOK_QUALITY_MODE) -> Optional[pd.DataFrame]:
    if book_type == "atm_straddle_30d":
        legs = build_atm_straddle(day_df, target_dte=TARGET_DTE)
    elif book_type == "risk_reversal_25d_30d":
        legs = build_risk_reversal(day_df, target_dte=TARGET_DTE)
    elif book_type == SHORT_PUT_SPREAD_BOOK_TYPE:
        legs = build_short_put_spread(day_df, target_dte=TARGET_DTE)
    else:
        raise ValueError(f"unknown book_type: {book_type}")

    if legs is None or len(legs) == 0:
        return None

    quality = diagnose_book_selection_quality(legs, book_type)
    quality["book_quality_mode"] = str(book_quality_mode)
    if not is_book_selection_quality_ok(quality, book_type, book_quality_mode):
        return None
    legs.attrs["book_quality_info"] = quality
    return legs

def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _calendar_gap_days(day_t: pd.DataFrame, day_t1: pd.DataFrame) -> float:
    d0 = pd.to_datetime(day_t["date"].iloc[0], errors="coerce")
    d1 = pd.to_datetime(day_t1["date"].iloc[0], errors="coerce")
    if pd.isna(d0) or pd.isna(d1):
        return 1.0
    gap = float((d1 - d0).days)
    return max(gap, 1.0)

def _interp_mid_same_exdate(chain_sub: pd.DataFrame, strike_target: float, target_dte: float) -> float:
    if chain_sub.empty:
        return np.nan
    x = chain_sub.copy()
    x["dte_gap"] = np.abs(pd.to_numeric(x["dte"], errors="coerce") - target_dte)
    x = x.sort_values(["dte_gap", "strike"]).drop_duplicates(["strike"], keep="first")
    x = x.dropna(subset=["strike", "mid"]).sort_values("strike")
    if len(x) < 2:
        return np.nan
    strikes = x["strike"].to_numpy(dtype=float)
    mids = x["mid"].to_numpy(dtype=float)
    keep = np.isfinite(strikes) & np.isfinite(mids)
    strikes = strikes[keep]
    mids = mids[keep]
    if len(strikes) < 2:
        return np.nan
    order = np.argsort(strikes)
    strikes = strikes[order]
    mids = mids[order]
    uniq_mask = np.append([True], np.diff(strikes) > 0)
    strikes = strikes[uniq_mask]
    mids = mids[uniq_mask]
    if len(strikes) < 2:
        return np.nan
    if strike_target < strikes.min() - NEXT_DAY_STRIKE_ABS_TOL or strike_target > strikes.max() + NEXT_DAY_STRIKE_ABS_TOL:
        return np.nan
    return float(np.interp(strike_target, strikes, mids))

def lookup_next_option_mid(day_t1: pd.DataFrame, leg: pd.Series, gap_days: float, marking_mode: str = "robust_all") -> Tuple[float, str]:
    cp_flag = str(leg.get("cp_flag_t", leg.get("cp_flag", ""))).upper()
    exdate_t = pd.to_datetime(leg.get("exdate_t"), errors="coerce")
    strike_t = _safe_float(leg.get("strike_t"))
    dte_t = _safe_float(leg.get("dte_t"))
    optionid_key = leg.get("optionid_key", pd.NA)
    target_dte_next = max(dte_t - gap_days, 0.0) if np.isfinite(dte_t) else np.nan

    chain = day_t1.dropna(subset=["mid", "cp_flag", "strike", "dte", "exdate"]).copy()
    chain["cp_flag"] = chain["cp_flag"].astype(str).str.upper()
    chain = chain[chain["cp_flag"] == cp_flag].copy()
    if chain.empty:
        return np.nan, "missing_cp"

    if pd.notna(optionid_key) and "optionid_key" in chain.columns:
        exact = chain[chain["optionid_key"] == optionid_key].copy()
        if not exact.empty:
            exact = exact.sort_values(["dte", "strike"])
            return float(exact.iloc[0]["mid"]), "exact_optionid"

    if pd.notna(exdate_t):
        same_ex = chain[chain["exdate"] == exdate_t].copy()
    else:
        same_ex = pd.DataFrame()

    if not same_ex.empty and np.isfinite(strike_t):
        exact_contract = same_ex[np.isclose(same_ex["strike"].to_numpy(dtype=float), strike_t, atol=NEXT_DAY_STRIKE_ABS_TOL)].copy()
        if not exact_contract.empty:
            if np.isfinite(target_dte_next):
                exact_contract["dte_gap"] = np.abs(pd.to_numeric(exact_contract["dte"], errors="coerce") - target_dte_next)
                exact_contract = exact_contract.sort_values(["dte_gap", "mid"], ascending=[True, False])
            return float(exact_contract.iloc[0]["mid"]), "exact_contract"

        if marking_mode == "strict_exact_contract":
            return np.nan, "missing_after_exact_contract"

        interp_mid = _interp_mid_same_exdate(same_ex, strike_target=strike_t, target_dte=target_dte_next)
        if np.isfinite(interp_mid):
            return float(interp_mid), "interp_same_exdate"

        if np.isfinite(target_dte_next):
            same_ex = same_ex.copy()
            same_ex["dte_gap"] = np.abs(pd.to_numeric(same_ex["dte"], errors="coerce") - target_dte_next)
            same_ex["strike_gap"] = np.abs(pd.to_numeric(same_ex["strike"], errors="coerce") - strike_t) / max(abs(strike_t), 1.0)
            same_ex["score"] = same_ex["dte_gap"] + NEXT_DAY_NEAREST_STRIKE_WEIGHT * same_ex["strike_gap"]
            same_ex = same_ex.sort_values(["score", "dte_gap", "strike_gap"])
            best = same_ex.iloc[0]
            if np.isfinite(best["dte_gap"]) and float(best["dte_gap"]) <= NEXT_DAY_MAX_DTE_GAP:
                return float(best["mid"]), "nearest_same_exdate"

    if marking_mode == "strict_exact_contract":
        return np.nan, "missing_after_exact_contract"

    if np.isfinite(strike_t) and np.isfinite(target_dte_next):
        any_cp = chain.copy()
        any_cp["dte_gap"] = np.abs(pd.to_numeric(any_cp["dte"], errors="coerce") - target_dte_next)
        any_cp["strike_gap"] = np.abs(pd.to_numeric(any_cp["strike"], errors="coerce") - strike_t) / max(abs(strike_t), 1.0)
        any_cp["score"] = any_cp["dte_gap"] + NEXT_DAY_NEAREST_STRIKE_WEIGHT * any_cp["strike_gap"]
        any_cp = any_cp.sort_values(["score", "dte_gap", "strike_gap"])
        best = any_cp.iloc[0]
        if np.isfinite(best["dte_gap"]) and float(best["dte_gap"]) <= NEXT_DAY_MAX_DTE_GAP:
            return float(best["mid"]), "nearest_cp_any_exdate"

    return np.nan, "missing_option_mark"

def compute_loss_normalizer(legs: pd.DataFrame, option_legs: pd.DataFrame) -> Tuple[float, dict]:
    gross_option_premium = float(np.sum(np.abs(option_legs["weight"]) * option_legs["mid_t"]))
    net_option_premium = float(np.sum(option_legs["weight"] * option_legs["mid_t"]))
    gross_option_vega = float(np.nansum(np.abs(option_legs["weight"] * option_legs["vega_t"])))

    spot_legs = legs[legs["instrument_type"].astype(str).eq("spot")].copy()
    gross_spot_hedge_notional = 0.0
    if len(spot_legs) > 0:
        gross_spot_hedge_notional = float(np.nansum(np.abs(spot_legs["weight"] * spot_legs["mid_t"])))

    if RR_NORMALIZATION == "gross_option_premium":
        normalizer = gross_option_premium
    elif RR_NORMALIZATION == "gross_option_premium_plus_spot_notional":
        normalizer = gross_option_premium + gross_spot_hedge_notional
    else:
        raise ValueError(f"unknown RR_NORMALIZATION: {RR_NORMALIZATION}")

    meta = {
        "gross_option_premium_t": gross_option_premium,
        "net_option_premium_t": net_option_premium,
        "gross_option_vega_t": gross_option_vega,
        "gross_spot_hedge_notional_t": gross_spot_hedge_notional,
        "normalizer_t": float(normalizer) if np.isfinite(normalizer) else np.nan,
        "normalization_rule": RR_NORMALIZATION,
    }
    return normalizer, meta

# ============================================================
# One-step-ahead book loss
# ============================================================
def compute_book_loss_one_step(day_t: pd.DataFrame,
                               day_t1: pd.DataFrame,
                               book_type: str,
                               marking_mode: str = PRIMARY_MARKING_MODE,
                               book_quality_mode: str = PRIMARY_BOOK_QUALITY_MODE) -> Optional[dict]:
    legs = build_standard_book(day_t, book_type=book_type, book_quality_mode=book_quality_mode)
    if legs is None or len(legs) == 0:
        return None

    quality_info = dict(getattr(legs, "attrs", {}).get("book_quality_info", diagnose_book_selection_quality(legs, book_type)))
    quality_info["book_quality_mode"] = str(book_quality_mode)

    legs = legs.copy()
    if "instrument_type" not in legs.columns:
        legs["instrument_type"] = "option"

    gap_days = _calendar_gap_days(day_t, day_t1)

    spot_t1_series = pd.to_numeric(day_t1["spot"], errors="coerce").dropna()
    spot_t1 = float(spot_t1_series.iloc[0]) if len(spot_t1_series) > 0 else np.nan

    mids_next = []
    mark_sources = []
    n_exact = 0
    n_contract = 0
    n_interp = 0
    n_nearest = 0
    n_fallback = 0

    for _, leg in legs.iterrows():
        inst = str(leg.get("instrument_type", "option"))
        if inst == "option":
            mid_next, mark_source = lookup_next_option_mid(day_t1, leg, gap_days, marking_mode=marking_mode)
            if not np.isfinite(mid_next):
                return None
            mids_next.append(float(mid_next))
            mark_sources.append(mark_source)

            if mark_source == "exact_optionid":
                n_exact += 1
            elif mark_source == "exact_contract":
                n_contract += 1
                n_fallback += 1
            elif mark_source == "interp_same_exdate":
                n_interp += 1
                n_fallback += 1
            elif mark_source in {"nearest_same_exdate", "nearest_cp_any_exdate"}:
                n_nearest += 1
                n_fallback += 1
            else:
                return None
        elif inst == "spot":
            if not np.isfinite(spot_t1):
                return None
            mids_next.append(float(spot_t1))
            mark_sources.append("spot_mark")
        else:
            raise ValueError(f"unknown instrument_type: {inst}")

    legs["mid_t1"] = mids_next
    legs["mark_source_t1"] = mark_sources

    raw_loss = -float(np.sum(legs["weight"] * (legs["mid_t1"] - legs["mid_t"])))

    option_mask = legs["instrument_type"].astype(str).eq("option")
    option_legs = legs[option_mask].copy()
    if len(option_legs) == 0:
        return None

    normalizer, norm_meta = compute_loss_normalizer(legs, option_legs)
    if not np.isfinite(normalizer) or normalizer <= EPS:
        return None

    loss_norm = raw_loss / normalizer

    option_legs = option_legs.reset_index(drop=True)
    leg1 = option_legs.iloc[0]
    leg2 = option_legs.iloc[1] if len(option_legs) > 1 else option_legs.iloc[0]

    option_net_delta_pre_hedge = float(np.nansum(option_legs["weight"] * option_legs["delta_t"]))
    option_abs_delta_pre_hedge = float(np.nansum(np.abs(option_legs["weight"] * option_legs["delta_t"])))
    stock_hedge_weight = float(np.nansum(
        legs.loc[legs["instrument_type"].astype(str) == "spot", "weight"]
    )) if np.any(legs["instrument_type"].astype(str) == "spot") else 0.0

    out = {
        "book_type": book_type,
        **quality_info,
        "raw_loss": raw_loss,
        "loss_norm": loss_norm,
        **norm_meta,
        "option_net_delta_pre_hedge_t": option_net_delta_pre_hedge,
        "option_abs_delta_pre_hedge_t": option_abs_delta_pre_hedge,
        "stock_hedge_weight_t": stock_hedge_weight,
        "book_net_delta_t": float(np.nansum(legs["weight"] * legs["delta_t"])),
        "book_abs_delta_t": float(np.nansum(np.abs(legs["weight"] * legs["delta_t"]))),
        "book_net_vega_t": float(np.nansum(legs["weight"] * legs["vega_t"])),
        "book_abs_vega_t": float(np.nansum(np.abs(legs["weight"] * legs["vega_t"]))),
        "book_avg_dte_t": float(np.nanmean(option_legs["dte_t"])),
        "book_avg_abs_k_t": float(np.nanmean(np.abs(option_legs["k_t"]))),
        "n_legs": int(len(legs)),
        "n_option_legs": int(len(option_legs)),
        "n_option_mark_exact_t": int(n_exact),
        "n_option_mark_contract_t": int(n_contract),
        "n_option_mark_interp_t": int(n_interp),
        "n_option_mark_nearest_t": int(n_nearest),
        "n_option_mark_fallback_t": int(n_fallback),
        "leg1_mid_t": _safe_float(leg1["mid_t"]),
        "leg2_mid_t": _safe_float(leg2["mid_t"]),
        "leg1_iv_t": _safe_float(leg1["iv_t"]),
        "leg2_iv_t": _safe_float(leg2["iv_t"]),
        "leg1_delta_t": _safe_float(leg1["delta_t"]),
        "leg2_delta_t": _safe_float(leg2["delta_t"]),
        "leg1_vega_t": _safe_float(leg1["vega_t"]),
        "leg2_vega_t": _safe_float(leg2["vega_t"]),
        "leg1_abs_k_t": float(abs(_safe_float(leg1["k_t"], default=0.0))),
        "leg2_abs_k_t": float(abs(_safe_float(leg2["k_t"], default=0.0))),
        "book_iv_mean_t": float(np.nanmean(option_legs["iv_t"])),
        "book_iv_spread_t": float(np.nanmax(option_legs["iv_t"]) - np.nanmin(option_legs["iv_t"])),
    }
    return out

# ============================================================
# Ex-ante feature helpers
# ============================================================
def add_lagged_marking_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add marking-quality features that are known at the forecast time.

    The current columns n_option_mark_*_t summarize how the just-constructed
    date-t book was successfully marked on date t+1. Those columns are realized
    outcome diagnostics and are not allowed in X_t. This function converts them
    into lagged count and rolling-rate features. The shift(1) is essential.
    """
    panel = panel.sort_values("date").reset_index(drop=True).copy()

    required = list(REALIZED_CURRENT_MARKING_DIAGNOSTIC_COLS) + ["n_option_legs"]
    for c in required:
        if c not in panel.columns:
            panel[c] = np.nan

    exact = pd.to_numeric(panel["n_option_mark_exact_t"], errors="coerce")
    contract = pd.to_numeric(panel["n_option_mark_contract_t"], errors="coerce")
    interp = pd.to_numeric(panel["n_option_mark_interp_t"], errors="coerce")
    nearest = pd.to_numeric(panel["n_option_mark_nearest_t"], errors="coerce")
    fallback = pd.to_numeric(panel["n_option_mark_fallback_t"], errors="coerce")
    n_legs = pd.to_numeric(panel["n_option_legs"], errors="coerce")
    proxy = interp + nearest

    panel["mark_exact_count_lag1"] = exact.shift(1)
    panel["mark_contract_count_lag1"] = contract.shift(1)
    panel["mark_interp_count_lag1"] = interp.shift(1)
    panel["mark_nearest_count_lag1"] = nearest.shift(1)
    panel["mark_fallback_count_lag1"] = fallback.shift(1)
    panel["mark_proxy_count_lag1"] = proxy.shift(1)

    numerator_map = {
        "exact": exact,
        "contract": contract,
        "interp": interp,
        "nearest": nearest,
        "fallback": fallback,
        "proxy": proxy,
    }
    denom_shifted = n_legs.shift(1)
    for window in [5, 21]:
        minp = max(2, window // 2)
        denom_roll = denom_shifted.rolling(window, min_periods=minp).sum()
        for name, series in numerator_map.items():
            numer_roll = series.shift(1).rolling(window, min_periods=minp).sum()
            rate = numer_roll / denom_roll.replace(0.0, np.nan)
            panel[f"mark_{name}_rate_{window}"] = rate

    return panel


def assert_ex_ante_feature_set(feature_cols: List[str]) -> None:
    """Fail fast if realized t -> t+1 marking diagnostics re-enter X_t."""
    bad = [c for c in feature_cols if c in set(REALIZED_CURRENT_MARKING_DIAGNOSTIC_COLS)]
    if bad and STRICT_EX_ANTE_FEATURE_CHECK:
        raise ValueError(
            "Forecast-time feature leakage detected. Remove realized marking "
            f"diagnostic columns from get_feature_cols: {bad}"
        )


def write_feature_column_audit(feature_cols: List[str], out_dir: Path, state_cols: Optional[List[str]] = None) -> Path:
    """Save the model feature list used in this run for paper auditability."""
    state_set = set(state_cols or [])
    rows = []
    current_mark_set = set(REALIZED_CURRENT_MARKING_DIAGNOSTIC_COLS)
    lagged_mark_set = set(LAGGED_MARKING_FEATURE_COLS)
    for i, col in enumerate(feature_cols):
        if col in state_set:
            block = "state"
        elif col in lagged_mark_set:
            block = "lagged_marking_diagnostic"
        elif col.startswith("loss_"):
            block = "lagged_loss"
        elif col in current_mark_set:
            block = "forbidden_current_marking_diagnostic"
        else:
            block = "book_or_surface_descriptor"
        rows.append({
            "feature_index": i,
            "feature_col": col,
            "feature_block": block,
            "is_realized_current_marking_diagnostic": int(col in current_mark_set),
            "is_lagged_marking_feature": int(col in lagged_mark_set),
        })
    audit = pd.DataFrame(rows)
    path = Path(out_dir) / "book_var_feature_columns_no_leakage.csv"
    audit.to_csv(path, index=False)
    return path


# ============================================================
# Build the second-layer book panel
# ============================================================
def build_book_panel(feat_df: pd.DataFrame,
                     state_cols: List[str],
                     daily_chain_map: Dict[pd.Timestamp, pd.DataFrame],
                     book_type: str,
                     marking_mode: str = PRIMARY_MARKING_MODE,
                     book_quality_mode: str = PRIMARY_BOOK_QUALITY_MODE) -> pd.DataFrame:
    rows = []
    feat_df = feat_df.sort_values("date").reset_index(drop=True)

    for i in range(len(feat_df) - 1):
        date_t = pd.Timestamp(feat_df.loc[i, "date"])
        date_t1 = pd.Timestamp(feat_df.loc[i + 1, "date"])

        day_t = daily_chain_map.get(date_t, None)
        day_t1 = daily_chain_map.get(date_t1, None)
        if day_t is None or day_t1 is None or len(day_t) == 0 or len(day_t1) == 0:
            continue

        book_info = compute_book_loss_one_step(day_t, day_t1, book_type=book_type, marking_mode=marking_mode, book_quality_mode=book_quality_mode)
        if book_info is None:
            continue

        row = {
            "date": date_t,
            "next_date": date_t1,
            "book_type": book_type,
            "book_quality_mode": book_info.get("book_quality_mode", book_quality_mode),
            "quality_pass_strict_economic": int(book_info.get("quality_pass_strict_economic", 0)) if np.isfinite(book_info.get("quality_pass_strict_economic", np.nan)) else 0,
            "book_quality_max_abs_k_t": book_info.get("book_quality_max_abs_k_t", np.nan),
            "book_quality_mean_abs_k_t": book_info.get("book_quality_mean_abs_k_t", np.nan),
            "book_quality_max_abs_dte_error_t": book_info.get("book_quality_max_abs_dte_error_t", np.nan),
            "atm_call_abs_k_t": book_info.get("atm_call_abs_k_t", np.nan),
            "atm_put_abs_k_t": book_info.get("atm_put_abs_k_t", np.nan),
            "atm_max_abs_k_t": book_info.get("atm_max_abs_k_t", np.nan),
            "rr_call_delta_error_t": book_info.get("rr_call_delta_error_t", np.nan),
            "rr_put_delta_error_t": book_info.get("rr_put_delta_error_t", np.nan),
            "rr_max_delta_error_t": book_info.get("rr_max_delta_error_t", np.nan),
            "putspread_short_delta_error_t": book_info.get("putspread_short_delta_error_t", np.nan),
            "putspread_long_delta_error_t": book_info.get("putspread_long_delta_error_t", np.nan),
            "putspread_max_delta_error_t": book_info.get("putspread_max_delta_error_t", np.nan),
            "loss_norm_tp1": book_info["loss_norm"],
            "raw_loss_tp1": book_info["raw_loss"],
            "gross_premium_t": book_info["gross_option_premium_t"],
            "net_premium_t": book_info["net_option_premium_t"],
            "gross_option_vega_t": book_info["gross_option_vega_t"],
            "gross_spot_hedge_notional_t": book_info["gross_spot_hedge_notional_t"],
            "normalizer_t": book_info["normalizer_t"],
            "normalization_rule": book_info["normalization_rule"],
            "option_net_delta_pre_hedge_t": book_info["option_net_delta_pre_hedge_t"],
            "option_abs_delta_pre_hedge_t": book_info["option_abs_delta_pre_hedge_t"],
            "stock_hedge_weight_t": book_info["stock_hedge_weight_t"],
            "book_net_delta_t": book_info["book_net_delta_t"],
            "book_abs_delta_t": book_info["book_abs_delta_t"],
            "book_net_vega_t": book_info["book_net_vega_t"],
            "book_abs_vega_t": book_info["book_abs_vega_t"],
            "book_avg_dte_t": book_info["book_avg_dte_t"],
            "book_avg_abs_k_t": book_info["book_avg_abs_k_t"],
            "n_legs": book_info["n_legs"],
            "n_option_legs": book_info["n_option_legs"],
            "n_option_mark_exact_t": book_info["n_option_mark_exact_t"],
            "n_option_mark_contract_t": book_info["n_option_mark_contract_t"],
            "n_option_mark_interp_t": book_info["n_option_mark_interp_t"],
            "n_option_mark_nearest_t": book_info["n_option_mark_nearest_t"],
            "n_option_mark_fallback_t": book_info["n_option_mark_fallback_t"],
            "leg1_mid_t": book_info["leg1_mid_t"],
            "leg2_mid_t": book_info["leg2_mid_t"],
            "leg1_iv_t": book_info["leg1_iv_t"],
            "leg2_iv_t": book_info["leg2_iv_t"],
            "leg1_delta_t": book_info["leg1_delta_t"],
            "leg2_delta_t": book_info["leg2_delta_t"],
            "leg1_vega_t": book_info["leg1_vega_t"],
            "leg2_vega_t": book_info["leg2_vega_t"],
            "leg1_abs_k_t": book_info["leg1_abs_k_t"],
            "leg2_abs_k_t": book_info["leg2_abs_k_t"],
            "book_iv_mean_t": book_info["book_iv_mean_t"],
            "book_iv_spread_t": book_info["book_iv_spread_t"],
        }

        for c in state_cols:
            row[c] = feat_df.loc[i, c]

        rows.append(row)

    panel = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if len(panel) == 0:
        raise ValueError(f"{book_type}: no valid book-loss sample was constructed.")

    # Lagged loss features. The shift(1) prevents the current realized loss
    # Y_{t+1} from entering the date-t feature matrix.
    panel["loss_lag1"] = panel["loss_norm_tp1"].shift(1)
    panel["loss_mean_5"] = panel["loss_norm_tp1"].shift(1).rolling(5).mean()
    panel["loss_std_5"] = panel["loss_norm_tp1"].shift(1).rolling(5).std()
    panel["loss_mean_21"] = panel["loss_norm_tp1"].shift(1).rolling(21).mean()
    panel["loss_std_21"] = panel["loss_norm_tp1"].shift(1).rolling(21).std()

    # Lagged marking diagnostics. Current n_option_mark_*_t columns are
    # realized t -> t+1 diagnostics only and are not used by get_feature_cols.
    panel = add_lagged_marking_features(panel)

    panel["is_crisis"] = ((panel["date"] >= CRISIS_START) & (panel["date"] <= CRISIS_END)).astype(int)
    return panel

# ============================================================
# Base quantile models
# ============================================================
def train_base_var_model_v2(X_train: np.ndarray, y_train: np.ndarray, model_kind_requested: str):
    requested = str(model_kind_requested).lower()
    tried = []

    def fit_gbr():
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=QUANTILE_LEVEL,
            n_estimators=GBR_N_ESTIMATORS,
            learning_rate=GBR_LEARNING_RATE,
            max_depth=GBR_MAX_DEPTH,
            min_samples_leaf=GBR_MIN_SAMPLES_LEAF,
            subsample=GBR_SUBSAMPLE,
            random_state=42
        )
        model.fit(X_train, y_train)
        setattr(model, "_model_kind_used", "gbr")
        return model

    def fit_lgbm():
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")
        model = LGBMRegressor(
            objective="quantile",
            alpha=QUANTILE_LEVEL,
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            subsample=LGBM_SUBSAMPLE,
            colsample_bytree=LGBM_COLSAMPLE_BYTREE,
            reg_alpha=LGBM_REG_ALPHA,
            reg_lambda=LGBM_REG_LAMBDA,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_train, y_train)
        setattr(model, "_model_kind_used", "lightgbm")
        return model

    def fit_xgb():
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=QUANTILE_LEVEL,
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            reg_alpha=XGB_REG_ALPHA,
            reg_lambda=XGB_REG_LAMBDA,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        setattr(model, "_model_kind_used", "xgboost")
        return model

    order_map = {
        "lightgbm": [fit_lgbm, fit_gbr],
        "xgboost": [fit_xgb, fit_lgbm, fit_gbr],
        "gbr": [fit_gbr],
    }
    if requested not in order_map:
        raise ValueError(f"unknown BASE_MODEL_KIND: {model_kind_requested}")

    last_err = None
    for fitter in order_map[requested]:
        try:
            model = fitter()
            if tried:
                print(f"base learner fallback: requested={requested}, used={model._model_kind_used}, failed_before={tried}")
            return model
        except Exception as e:
            tried.append(fitter.__name__)
            last_err = e

    raise RuntimeError(f"all base quantile learners failed: requested={requested}, tried={tried}, last_err={last_err}")

# ============================================================
# One-sided sequential calibration buffer
# ============================================================
def get_var_buffer_time_decay_with_fallback(
    residual_history: List[dict],
    pred_idx: int,
    last_valid_buffer: float
):
    cal_items = [x for x in residual_history if (pred_idx - x["pred_idx"]) <= CALIB_WINDOW]

    b_t = np.nan
    b_source = "none"

    if len(cal_items) >= MIN_CAL_SCORES:
        residuals = np.array([x["residual"] for x in cal_items], dtype=float)
        weights = build_time_decay_weights(cal_items, pred_idx)

        b_t = weighted_quantile(residuals, weights, QUANTILE_LEVEL)
        b_source = "time_decay"

        if not np.isfinite(b_t):
            b_t = safe_nanquantile(residuals, QUANTILE_LEVEL)
            b_source = "time_decay_fallback_unweighted"

        if not np.isfinite(b_t):
            b_t = last_valid_buffer
            b_source = "time_decay_fallback_prev_buffer"

        if not np.isfinite(b_t):
            b_t = DEFAULT_BUFFER
            b_source = "time_decay_fallback_default"
    else:
        if len(residual_history) > 10:
            residuals = np.array([x["residual"] for x in residual_history], dtype=float)
            b_t = safe_nanquantile(residuals, QUANTILE_LEVEL)
            b_source = "warmup_unweighted"

            if not np.isfinite(b_t):
                b_t = last_valid_buffer
                b_source = "warmup_prev_buffer"

            if not np.isfinite(b_t):
                b_t = DEFAULT_BUFFER
                b_source = "warmup_default"
        else:
            b_t = DEFAULT_BUFFER
            b_source = "cold_start_default"

    if not np.isfinite(b_t):
        b_t = DEFAULT_BUFFER
        b_source = "forced_default"

    return float(b_t), b_source

# ============================================================
# rolling diagnostics
# ============================================================
def add_rolling_diagnostics(book_res: pd.DataFrame) -> pd.DataFrame:
    g = book_res.sort_values("date").copy()

    for w in ROLL_DIAG_WINDOWS:
        minp = max(20, w // 2)
        for name in ["base", "hist", "ewma", "caviar", "garch_t", "conf"]:
            exc_col = f"exceed_{name}"
            viol_col = f"violation_{name}"
            if exc_col in g.columns:
                g[f"roll{w}_{exc_col}"] = g[exc_col].rolling(w, min_periods=minp).mean()
            if viol_col in g.columns:
                g[f"roll{w}_{viol_col}"] = g[viol_col].rolling(w, min_periods=minp).mean()

    return g

# ============================================================
# V25 paper-ready second-layer experiment and robustness checks
# ============================================================
def get_feature_cols(state_cols: List[str]) -> List[str]:
    """Return forecast-time features for the rolling quantile learner.

    Important information-set rule:
    - Current n_option_mark_*_t columns are realized diagnostics from the
      t -> t+1 valuation step and must not enter X_t.
    - Lagged mark_* features below are shifted past diagnostics and are
      therefore known at the date-t forecast time.
    """
    feature_cols = state_cols + [
        "gross_premium_t",
        "net_premium_t",
        "gross_option_vega_t",
        "gross_spot_hedge_notional_t",
        "normalizer_t",
        "option_net_delta_pre_hedge_t",
        "option_abs_delta_pre_hedge_t",
        "stock_hedge_weight_t",
        "book_net_delta_t",
        "book_abs_delta_t",
        "book_net_vega_t",
        "book_abs_vega_t",
        "book_avg_dte_t",
        "book_avg_abs_k_t",
        "n_legs",
        "n_option_legs",
        "leg1_mid_t",
        "leg2_mid_t",
        "leg1_iv_t",
        "leg2_iv_t",
        "leg1_delta_t",
        "leg2_delta_t",
        "leg1_vega_t",
        "leg2_vega_t",
        "leg1_abs_k_t",
        "leg2_abs_k_t",
        "book_iv_mean_t",
        "book_iv_spread_t",
        *LAGGED_MARKING_FEATURE_COLS,
        "loss_lag1",
        "loss_mean_5",
        "loss_std_5",
        "loss_mean_21",
        "loss_std_21",
    ]
    assert_ex_ante_feature_set(feature_cols)
    return feature_cols


def build_experiment_specs() -> List[dict]:
    specs: List[dict] = []
    seen = set()

    def _label_floor(x: Optional[float]) -> str:
        return "none" if x is None else str(x).replace(".", "p")

    def _label_quality(q: str) -> str:
        return str(q).replace(" ", "_")

    def _add_spec(
        experiment_id: str,
        experiment_group: str,
        model_kind: str,
        marking_mode: str,
        var_floor_value: Optional[float],
        book_quality_mode: str = PRIMARY_BOOK_QUALITY_MODE,
    ):
        key = (
            str(model_kind).lower(),
            str(marking_mode),
            None if var_floor_value is None else float(var_floor_value),
            str(book_quality_mode),
        )
        if key in seen:
            return
        seen.add(key)
        specs.append({
            "experiment_id": experiment_id,
            "experiment_group": experiment_group,
            "model_kind": str(model_kind).lower(),
            "marking_mode": str(marking_mode),
            "var_floor_value": (None if var_floor_value is None else float(var_floor_value)),
            "book_quality_mode": str(book_quality_mode),
            "book_types": list(MAIN_BOOK_TYPES),
        })

    _add_spec(
        experiment_id=f"main_{PRIMARY_LEARNER}_{PRIMARY_MARKING_MODE}_floor_{_label_floor(PRIMARY_VAR_FLOOR)}_quality_{_label_quality(PRIMARY_BOOK_QUALITY_MODE)}",
        experiment_group="main",
        model_kind=PRIMARY_LEARNER,
        marking_mode=PRIMARY_MARKING_MODE,
        var_floor_value=PRIMARY_VAR_FLOOR,
        book_quality_mode=PRIMARY_BOOK_QUALITY_MODE,
    )

    if RUN_LEARNER_ROBUSTNESS:
        for mk in LEARNER_ROBUSTNESS_MODELS:
            _add_spec(
                experiment_id=f"learner_{mk}_{PRIMARY_MARKING_MODE}_floor_{_label_floor(PRIMARY_VAR_FLOOR)}_quality_{_label_quality(PRIMARY_BOOK_QUALITY_MODE)}",
                experiment_group="learner_robustness",
                model_kind=mk,
                marking_mode=PRIMARY_MARKING_MODE,
                var_floor_value=PRIMARY_VAR_FLOOR,
                book_quality_mode=PRIMARY_BOOK_QUALITY_MODE,
            )

    if RUN_MARKING_SENSITIVITY:
        for mm in MARKING_SENSITIVITY_MODES:
            _add_spec(
                experiment_id=f"marking_{PRIMARY_LEARNER}_{mm}_floor_{_label_floor(PRIMARY_VAR_FLOOR)}_quality_{_label_quality(PRIMARY_BOOK_QUALITY_MODE)}",
                experiment_group="marking_sensitivity",
                model_kind=PRIMARY_LEARNER,
                marking_mode=mm,
                var_floor_value=PRIMARY_VAR_FLOOR,
                book_quality_mode=PRIMARY_BOOK_QUALITY_MODE,
            )

    if RUN_VAR_FLOOR_SENSITIVITY:
        for floor_value in VAR_FLOOR_SENSITIVITY_VALUES:
            _add_spec(
                experiment_id=f"floor_{PRIMARY_LEARNER}_{PRIMARY_MARKING_MODE}_{_label_floor(floor_value)}_quality_{_label_quality(PRIMARY_BOOK_QUALITY_MODE)}",
                experiment_group="var_floor_sensitivity",
                model_kind=PRIMARY_LEARNER,
                marking_mode=PRIMARY_MARKING_MODE,
                var_floor_value=floor_value,
                book_quality_mode=PRIMARY_BOOK_QUALITY_MODE,
            )

    if RUN_BOOK_QUALITY_ROBUSTNESS:
        for qmode in BOOK_QUALITY_ROBUSTNESS_MODES:
            _add_spec(
                experiment_id=f"bookquality_{PRIMARY_LEARNER}_{PRIMARY_MARKING_MODE}_floor_{_label_floor(PRIMARY_VAR_FLOOR)}_quality_{_label_quality(qmode)}",
                experiment_group="book_quality_robustness",
                model_kind=PRIMARY_LEARNER,
                marking_mode=PRIMARY_MARKING_MODE,
                var_floor_value=PRIMARY_VAR_FLOOR,
                book_quality_mode=qmode,
            )

    return specs


def run_single_experiment(
    feat_df: pd.DataFrame,
    state_cols: List[str],
    daily_chain_map: Dict[pd.Timestamp, pd.DataFrame],
    spec: dict,
):
    experiment_id = spec["experiment_id"]
    experiment_group = spec["experiment_group"]
    model_kind_requested = spec["model_kind"]
    marking_mode = spec["marking_mode"]
    var_floor_value = spec["var_floor_value"]
    book_quality_mode = spec.get("book_quality_mode", PRIMARY_BOOK_QUALITY_MODE)
    book_types = spec["book_types"]

    print(f"\n==================== Experiment: {experiment_id} ====================")
    print(
        f"group={experiment_group}, learner={model_kind_requested}, "
        f"marking={marking_mode}, var_floor={var_floor_value}, "
        f"book_quality={book_quality_mode}"
    )
    compute_classical = should_compute_classical_baselines(experiment_group)
    if compute_classical:
        print("classical baselines enabled: EWMA Historical VaR, CAViaR, GARCH-t")

    result_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []
    yearly_rows: List[pd.DataFrame] = []
    crisis_rows: List[pd.DataFrame] = []

    feature_cols = get_feature_cols(state_cols)
    assert_ex_ante_feature_set(feature_cols)

    for book_type in book_types:
        print(f"\n---------- [{experiment_id}] Running book: {book_type} ----------")
        panel = build_book_panel(
            feat_df,
            state_cols,
            daily_chain_map,
            book_type=book_type,
            marking_mode=marking_mode,
            book_quality_mode=book_quality_mode,
        )

        panel = panel.reset_index(drop=True)
        X_all = panel[feature_cols].to_numpy(dtype=float)
        y_all = panel["loss_norm_tp1"].to_numpy(dtype=float)

        if len(panel) < TRAIN_WINDOW + 30:
            print(f"[{experiment_id}] {book_type}: sample too small, skipped.")
            continue

        rows_this_book: List[dict] = []
        model = None
        feat_trans = None
        model_kind_used = None
        residual_history: List[dict] = []
        last_valid_buffer = DEFAULT_BUFFER
        caviar_params = None
        garch_t_params = None
        caviar_fit_ok = 0
        garch_t_fit_ok = 0

        for pred_idx in range(TRAIN_WINDOW, len(panel)):
            current_date = pd.Timestamp(panel.loc[pred_idx, "date"])

            need_retrain = (model is None) or ((pred_idx - TRAIN_WINDOW) % RETRAIN_EVERY == 0)
            if need_retrain:
                train_start = max(0, pred_idx - TRAIN_WINDOW)
                X_train = X_all[train_start:pred_idx]
                y_train = y_all[train_start:pred_idx]
                y_train_classical = y_train[np.isfinite(y_train)]

                if compute_classical:
                    caviar_params = fit_caviar_sav(y_train_classical)
                    garch_t_params = fit_garch_t(y_train_classical)
                    caviar_fit_ok = int(caviar_params is not None)
                    garch_t_fit_ok = int(garch_t_params is not None)

                mask = np.isfinite(y_train)
                X_train = X_train[mask]
                y_train = y_train[mask]

                retrain_success = False
                if len(X_train) >= 80:
                    X_train_scaled, new_feat_trans = fit_feature_transform(X_train)
                    valid_mask = np.isfinite(X_train_scaled).all(axis=1) & np.isfinite(y_train)
                    X_train_scaled = X_train_scaled[valid_mask]
                    y_train_used = y_train[valid_mask]

                    if len(X_train_scaled) >= 80:
                        new_model = train_base_var_model_v2(
                            X_train_scaled,
                            y_train_used,
                            model_kind_requested=model_kind_requested,
                        )
                        model = new_model
                        feat_trans = new_feat_trans
                        model_kind_used = getattr(new_model, "_model_kind_used", model_kind_requested)
                        retrain_success = True
                        print(
                            f"[{experiment_id}] [{book_type}] [{current_date.date()}] retrain done, "
                            f"samples={len(X_train_scaled)}, used={model_kind_used}"
                        )

                if model is None or feat_trans is None:
                    print(f"[{experiment_id}] [{book_type}] [{current_date.date()}] first model failed, skip.")
                    continue

                if (not retrain_success) and (pred_idx > TRAIN_WINDOW):
                    print(f"[{experiment_id}] [{book_type}] [{current_date.date()}] retrain failed, keep old model.")

            train_start = max(0, pred_idx - TRAIN_WINDOW)
            y_hist_train = y_all[train_start:pred_idx]
            y_hist_train = y_hist_train[np.isfinite(y_hist_train)]
            q_hat_hist = safe_nanquantile(y_hist_train, QUANTILE_LEVEL)
            if not np.isfinite(q_hat_hist):
                q_hat_hist = 0.0

            if compute_classical:
                q_hat_ewma = ewma_historical_var(y_hist_train, QUANTILE_LEVEL, EWMA_LAMBDA)
                q_hat_caviar = predict_caviar_sav(y_hist_train, caviar_params)
                q_hat_garch_t = predict_garch_t_var(y_hist_train, garch_t_params)
            else:
                q_hat_ewma = np.nan
                q_hat_caviar = np.nan
                q_hat_garch_t = np.nan

            q_hat_ewma = apply_var_floor(q_hat_ewma, var_floor_value) if np.isfinite(q_hat_ewma) else np.nan
            q_hat_caviar = apply_var_floor(q_hat_caviar, var_floor_value) if np.isfinite(q_hat_caviar) else np.nan
            q_hat_garch_t = apply_var_floor(q_hat_garch_t, var_floor_value) if np.isfinite(q_hat_garch_t) else np.nan

            X_pred = X_all[pred_idx:pred_idx + 1]
            X_pred_scaled = apply_feature_transform(X_pred, feat_trans)
            q_hat_base_raw = float(model.predict(X_pred_scaled)[0])

            b_t, b_source = get_var_buffer_time_decay_with_fallback(
                residual_history=residual_history,
                pred_idx=pred_idx,
                last_valid_buffer=last_valid_buffer,
            )
            last_valid_buffer = b_t

            # Route A: theoretical threshold == backtested threshold
            q_hat_ref, q_hat_core_op, q_hat_rep = build_route_a_thresholds(
                q_hat_base_raw=q_hat_base_raw,
                b_t=b_t,
                floor_value=var_floor_value,
            )

            y_true = float(y_all[pred_idx])
            residual_t = y_true - q_hat_ref

            exceed_base = int(y_true > q_hat_ref)
            exceed_hist = int(y_true > q_hat_hist)
            exceed_ewma = int(y_true > q_hat_ewma) if np.isfinite(q_hat_ewma) else np.nan
            exceed_caviar = int(y_true > q_hat_caviar) if np.isfinite(q_hat_caviar) else np.nan
            exceed_garch_t = int(y_true > q_hat_garch_t) if np.isfinite(q_hat_garch_t) else np.nan
            exceed_conf = int(y_true > q_hat_rep)

            violation_base = max(y_true - q_hat_ref, 0.0)
            violation_hist = max(y_true - q_hat_hist, 0.0)
            violation_ewma = max(y_true - q_hat_ewma, 0.0) if np.isfinite(q_hat_ewma) else np.nan
            violation_caviar = max(y_true - q_hat_caviar, 0.0) if np.isfinite(q_hat_caviar) else np.nan
            violation_garch_t = max(y_true - q_hat_garch_t, 0.0) if np.isfinite(q_hat_garch_t) else np.nan
            violation_conf = max(y_true - q_hat_rep, 0.0)

            pinball_base = _pinball_loss_one(y_true, q_hat_ref, QUANTILE_LEVEL)
            pinball_hist = _pinball_loss_one(y_true, q_hat_hist, QUANTILE_LEVEL)
            pinball_ewma = _pinball_loss_one(y_true, q_hat_ewma, QUANTILE_LEVEL)
            pinball_caviar = _pinball_loss_one(y_true, q_hat_caviar, QUANTILE_LEVEL)
            pinball_garch_t = _pinball_loss_one(y_true, q_hat_garch_t, QUANTILE_LEVEL)
            pinball_conf = _pinball_loss_one(y_true, q_hat_rep, QUANTILE_LEVEL)

            rows_this_book.append({
                "experiment_id": experiment_id,
                "experiment_group": experiment_group,
                "book_type": book_type,
                "date": current_date,
                "next_date": pd.Timestamp(panel.loc[pred_idx, "next_date"]),
                "loss_norm_tp1": y_true,
                "raw_loss_tp1": float(panel.loc[pred_idx, "raw_loss_tp1"]),
                "q_hat_base_raw": q_hat_base_raw,
                "q_hat_ref": q_hat_ref,
                "q_hat_core_op": q_hat_core_op,
                "q_hat_rep": q_hat_rep,
                # backwards-compatible names used by existing tables/plots
                "q_hat_base": q_hat_ref,
                "q_hat_hist": q_hat_hist,
                "q_hat_ewma": q_hat_ewma,
                "q_hat_caviar": q_hat_caviar,
                "q_hat_garch_t": q_hat_garch_t,
                "buffer_t": b_t,
                "var_conf_raw": q_hat_core_op,
                "var_conf": q_hat_rep,
                "residual_t": residual_t,
                "exceed_base": exceed_base,
                "exceed_hist": exceed_hist,
                "exceed_ewma": exceed_ewma,
                "exceed_caviar": exceed_caviar,
                "exceed_garch_t": exceed_garch_t,
                "exceed_conf": exceed_conf,
                "violation_base": violation_base,
                "violation_hist": violation_hist,
                "violation_ewma": violation_ewma,
                "violation_caviar": violation_caviar,
                "violation_garch_t": violation_garch_t,
                "violation_conf": violation_conf,
                "pinball_base": pinball_base,
                "pinball_hist": pinball_hist,
                "pinball_ewma": pinball_ewma,
                "pinball_caviar": pinball_caviar,
                "pinball_garch_t": pinball_garch_t,
                "pinball_conf": pinball_conf,
                "book_quality_mode": str(panel.loc[pred_idx, "book_quality_mode"]),
                "quality_pass_strict_economic": int(panel.loc[pred_idx, "quality_pass_strict_economic"]),
                "book_quality_max_abs_k_t": float(panel.loc[pred_idx, "book_quality_max_abs_k_t"]),
                "book_quality_mean_abs_k_t": float(panel.loc[pred_idx, "book_quality_mean_abs_k_t"]),
                "book_quality_max_abs_dte_error_t": float(panel.loc[pred_idx, "book_quality_max_abs_dte_error_t"]),
                "atm_call_abs_k_t": float(panel.loc[pred_idx, "atm_call_abs_k_t"]),
                "atm_put_abs_k_t": float(panel.loc[pred_idx, "atm_put_abs_k_t"]),
                "atm_max_abs_k_t": float(panel.loc[pred_idx, "atm_max_abs_k_t"]),
                "rr_call_delta_error_t": float(panel.loc[pred_idx, "rr_call_delta_error_t"]),
                "rr_put_delta_error_t": float(panel.loc[pred_idx, "rr_put_delta_error_t"]),
                "rr_max_delta_error_t": float(panel.loc[pred_idx, "rr_max_delta_error_t"]),
                "putspread_short_delta_error_t": float(panel.loc[pred_idx, "putspread_short_delta_error_t"]),
                "putspread_long_delta_error_t": float(panel.loc[pred_idx, "putspread_long_delta_error_t"]),
                "putspread_max_delta_error_t": float(panel.loc[pred_idx, "putspread_max_delta_error_t"]),
                "gross_premium_t": float(panel.loc[pred_idx, "gross_premium_t"]),
                "net_premium_t": float(panel.loc[pred_idx, "net_premium_t"]),
                "gross_option_vega_t": float(panel.loc[pred_idx, "gross_option_vega_t"]),
                "gross_spot_hedge_notional_t": float(panel.loc[pred_idx, "gross_spot_hedge_notional_t"]),
                "normalizer_t": float(panel.loc[pred_idx, "normalizer_t"]),
                "normalization_rule": str(panel.loc[pred_idx, "normalization_rule"]),
                "option_net_delta_pre_hedge_t": float(panel.loc[pred_idx, "option_net_delta_pre_hedge_t"]),
                "option_abs_delta_pre_hedge_t": float(panel.loc[pred_idx, "option_abs_delta_pre_hedge_t"]),
                "stock_hedge_weight_t": float(panel.loc[pred_idx, "stock_hedge_weight_t"]),
                "book_net_delta_t": float(panel.loc[pred_idx, "book_net_delta_t"]),
                "book_abs_delta_t": float(panel.loc[pred_idx, "book_abs_delta_t"]),
                "book_net_vega_t": float(panel.loc[pred_idx, "book_net_vega_t"]),
                "book_abs_vega_t": float(panel.loc[pred_idx, "book_abs_vega_t"]),
                "book_avg_dte_t": float(panel.loc[pred_idx, "book_avg_dte_t"]),
                "book_avg_abs_k_t": float(panel.loc[pred_idx, "book_avg_abs_k_t"]),
                "n_legs": int(panel.loc[pred_idx, "n_legs"]),
                "n_option_legs": int(panel.loc[pred_idx, "n_option_legs"]),
                "n_option_mark_exact_t": int(panel.loc[pred_idx, "n_option_mark_exact_t"]),
                "n_option_mark_contract_t": int(panel.loc[pred_idx, "n_option_mark_contract_t"]),
                "n_option_mark_interp_t": int(panel.loc[pred_idx, "n_option_mark_interp_t"]),
                "n_option_mark_nearest_t": int(panel.loc[pred_idx, "n_option_mark_nearest_t"]),
                "n_option_mark_fallback_t": int(panel.loc[pred_idx, "n_option_mark_fallback_t"]),
                "leg1_mid_t": float(panel.loc[pred_idx, "leg1_mid_t"]),
                "leg2_mid_t": float(panel.loc[pred_idx, "leg2_mid_t"]),
                "leg1_iv_t": float(panel.loc[pred_idx, "leg1_iv_t"]),
                "leg2_iv_t": float(panel.loc[pred_idx, "leg2_iv_t"]),
                "leg1_delta_t": float(panel.loc[pred_idx, "leg1_delta_t"]),
                "leg2_delta_t": float(panel.loc[pred_idx, "leg2_delta_t"]),
                "leg1_vega_t": float(panel.loc[pred_idx, "leg1_vega_t"]),
                "leg2_vega_t": float(panel.loc[pred_idx, "leg2_vega_t"]),
                "leg1_abs_k_t": float(panel.loc[pred_idx, "leg1_abs_k_t"]),
                "leg2_abs_k_t": float(panel.loc[pred_idx, "leg2_abs_k_t"]),
                "book_iv_mean_t": float(panel.loc[pred_idx, "book_iv_mean_t"]),
                "book_iv_spread_t": float(panel.loc[pred_idx, "book_iv_spread_t"]),
                "is_crisis": int(panel.loc[pred_idx, "is_crisis"]),
                "buffer_source": b_source,
                "base_model_kind_requested": model_kind_requested,
                "base_model_kind_used": model_kind_used if model_kind_used is not None else getattr(model, "_model_kind_used", model_kind_requested),
                "classical_baselines_enabled": int(compute_classical),
                "caviar_fit_ok": int(caviar_fit_ok),
                "garch_t_fit_ok": int(garch_t_fit_ok),
                "marking_mode": marking_mode,
                "book_quality_mode_spec": str(book_quality_mode),
                "var_floor_value": np.nan if var_floor_value is None else float(var_floor_value),
                "q_hat_base_floor_applied": int(np.isfinite(q_hat_base_raw) and np.isfinite(q_hat_ref) and (q_hat_ref > q_hat_base_raw + EPS)),
                "var_conf_floor_applied": int(np.isfinite(q_hat_core_op) and np.isfinite(q_hat_rep) and (q_hat_rep > q_hat_core_op + EPS)),
                "route_a_changed": int(np.isfinite(q_hat_rep) and np.isfinite(q_hat_base_raw + b_t) and (abs(q_hat_rep - apply_var_floor(q_hat_base_raw + b_t, var_floor_value)) > EPS)),
            })

            residual_history.append({
                "pred_idx": pred_idx,
                "residual": float(residual_t),
            })

        book_res = pd.DataFrame(rows_this_book).sort_values("date").reset_index(drop=True)
        if len(book_res) == 0:
            continue

        book_res = add_rolling_diagnostics(book_res)
        result_frames.append(book_res)

        crisis_res = book_res[book_res["is_crisis"] == 1].copy()

        summary_rows.append({
            "experiment_id": experiment_id,
            "experiment_group": experiment_group,
            "book_type": book_type,
            "n_backtest_days": len(book_res),
            "target_exceedance_alpha": ALPHA,
            "empirical_exceedance_rate_base": book_res["exceed_base"].mean(),
            "empirical_exceedance_rate_hist": book_res["exceed_hist"].mean(),
            "empirical_exceedance_rate_ewma": book_res["exceed_ewma"].mean() if "exceed_ewma" in book_res else np.nan,
            "empirical_exceedance_rate_caviar": book_res["exceed_caviar"].mean() if "exceed_caviar" in book_res else np.nan,
            "empirical_exceedance_rate_garch_t": book_res["exceed_garch_t"].mean() if "exceed_garch_t" in book_res else np.nan,
            "empirical_exceedance_rate_conf": book_res["exceed_conf"].mean(),
            "avg_loss_norm": book_res["loss_norm_tp1"].mean(),
            "avg_q_hat_base_raw": book_res["q_hat_base_raw"].mean(),
            "avg_q_hat_ref": book_res["q_hat_ref"].mean(),
            "avg_q_hat_core_op": book_res["q_hat_core_op"].mean(),
            "avg_q_hat_rep": book_res["q_hat_rep"].mean(),
            "avg_q_hat_base": book_res["q_hat_base"].mean(),
            "avg_q_hat_hist": book_res["q_hat_hist"].mean(),
            "avg_q_hat_ewma": book_res["q_hat_ewma"].mean() if "q_hat_ewma" in book_res else np.nan,
            "avg_q_hat_caviar": book_res["q_hat_caviar"].mean() if "q_hat_caviar" in book_res else np.nan,
            "avg_q_hat_garch_t": book_res["q_hat_garch_t"].mean() if "q_hat_garch_t" in book_res else np.nan,
            "avg_buffer_t": book_res["buffer_t"].mean(),
            "avg_var_conf_raw": book_res["var_conf_raw"].mean(),
            "avg_var_conf": book_res["var_conf"].mean(),
            "avg_violation_base": book_res["violation_base"].mean(),
            "avg_violation_hist": book_res["violation_hist"].mean(),
            "avg_violation_ewma": book_res["violation_ewma"].mean() if "violation_ewma" in book_res else np.nan,
            "avg_violation_caviar": book_res["violation_caviar"].mean() if "violation_caviar" in book_res else np.nan,
            "avg_violation_garch_t": book_res["violation_garch_t"].mean() if "violation_garch_t" in book_res else np.nan,
            "avg_violation_conf": book_res["violation_conf"].mean(),
            "avg_pinball_base": book_res["pinball_base"].mean(),
            "avg_pinball_hist": book_res["pinball_hist"].mean(),
            "avg_pinball_ewma": book_res["pinball_ewma"].mean() if "pinball_ewma" in book_res else np.nan,
            "avg_pinball_caviar": book_res["pinball_caviar"].mean() if "pinball_caviar" in book_res else np.nan,
            "avg_pinball_garch_t": book_res["pinball_garch_t"].mean() if "pinball_garch_t" in book_res else np.nan,
            "avg_pinball_conf": book_res["pinball_conf"].mean(),
            "avg_threshold_base": book_res["q_hat_base"].mean(),
            "avg_threshold_hist": book_res["q_hat_hist"].mean(),
            "avg_threshold_conf": book_res["var_conf"].mean(),
            "quality_strict_pass_share": book_res["quality_pass_strict_economic"].mean() if "quality_pass_strict_economic" in book_res else np.nan,
            "p95_book_quality_max_abs_k_t": book_res["book_quality_max_abs_k_t"].quantile(0.95) if "book_quality_max_abs_k_t" in book_res else np.nan,
            "p95_book_quality_max_abs_dte_error_t": book_res["book_quality_max_abs_dte_error_t"].quantile(0.95) if "book_quality_max_abs_dte_error_t" in book_res else np.nan,
            "avg_gross_spot_hedge_notional_t": book_res["gross_spot_hedge_notional_t"].mean(),
            "avg_normalizer_t": book_res["normalizer_t"].mean(),
            "avg_option_net_delta_pre_hedge_t": book_res["option_net_delta_pre_hedge_t"].mean(),
            "avg_abs_option_net_delta_pre_hedge_t": book_res["option_net_delta_pre_hedge_t"].abs().mean(),
            "avg_abs_book_net_delta_t": book_res["book_net_delta_t"].abs().mean(),
            "avg_stock_hedge_weight_t": book_res["stock_hedge_weight_t"].mean(),
            "max_roll50_exceed_base": book_res["roll50_exceed_base"].max() if "roll50_exceed_base" in book_res else np.nan,
            "max_roll50_exceed_hist": book_res["roll50_exceed_hist"].max() if "roll50_exceed_hist" in book_res else np.nan,
            "max_roll50_exceed_ewma": book_res["roll50_exceed_ewma"].max() if "roll50_exceed_ewma" in book_res else np.nan,
            "max_roll50_exceed_caviar": book_res["roll50_exceed_caviar"].max() if "roll50_exceed_caviar" in book_res else np.nan,
            "max_roll50_exceed_garch_t": book_res["roll50_exceed_garch_t"].max() if "roll50_exceed_garch_t" in book_res else np.nan,
            "max_roll50_exceed_conf": book_res["roll50_exceed_conf"].max() if "roll50_exceed_conf" in book_res else np.nan,
            "avg_n_option_mark_fallback_t": book_res["n_option_mark_fallback_t"].mean(),
            "sum_option_mark_exact_t": int(book_res["n_option_mark_exact_t"].sum()),
            "sum_option_mark_contract_t": int(book_res["n_option_mark_contract_t"].sum()),
            "sum_option_mark_interp_t": int(book_res["n_option_mark_interp_t"].sum()),
            "sum_option_mark_nearest_t": int(book_res["n_option_mark_nearest_t"].sum()),
            "n_buffer_fallback_unweighted": int(book_res["buffer_source"].astype(str).str.contains("fallback_unweighted").sum()),
            "n_buffer_fallback_prev": int(book_res["buffer_source"].astype(str).str.contains("fallback_prev_buffer").sum()),
            "n_buffer_fallback_default": int(book_res["buffer_source"].astype(str).str.contains("default").sum()),
            "n_q_hat_base_raw_neg": int((book_res["q_hat_base_raw"] < 0).sum()),
            "n_q_hat_ref_neg": int((book_res["q_hat_ref"] < 0).sum()),
            "n_q_hat_core_op_neg": int((book_res["q_hat_core_op"] < 0).sum()),
            "n_q_hat_rep_neg": int((book_res["q_hat_rep"] < 0).sum()),
            "n_q_hat_base_neg": int((book_res["q_hat_base"] < 0).sum()),
            "n_var_conf_raw_neg": int((book_res["var_conf_raw"] < 0).sum()),
            "n_var_conf_neg": int((book_res["var_conf"] < 0).sum()),
            "n_q_hat_base_floor_applied": int(book_res["q_hat_base_floor_applied"].sum()),
            "n_var_conf_floor_applied": int(book_res["var_conf_floor_applied"].sum()),
            "n_route_a_changed": int(book_res["route_a_changed"].sum()),
            "base_model_kind_requested": model_kind_requested,
            "base_model_kind_used": book_res["base_model_kind_used"].astype(str).mode().iloc[0],
            "marking_mode": marking_mode,
            "book_quality_mode": str(book_quality_mode),
            "var_floor_value": np.nan if var_floor_value is None else float(var_floor_value),
            "normalization_rule": book_res["normalization_rule"].astype(str).mode().iloc[0],
            "crisis_n_days": len(crisis_res),
            "crisis_exceedance_rate_base": crisis_res["exceed_base"].mean() if len(crisis_res) > 0 else np.nan,
            "crisis_exceedance_rate_hist": crisis_res["exceed_hist"].mean() if len(crisis_res) > 0 else np.nan,
            "crisis_exceedance_rate_ewma": crisis_res["exceed_ewma"].mean() if len(crisis_res) > 0 and "exceed_ewma" in crisis_res else np.nan,
            "crisis_exceedance_rate_caviar": crisis_res["exceed_caviar"].mean() if len(crisis_res) > 0 and "exceed_caviar" in crisis_res else np.nan,
            "crisis_exceedance_rate_garch_t": crisis_res["exceed_garch_t"].mean() if len(crisis_res) > 0 and "exceed_garch_t" in crisis_res else np.nan,
            "crisis_exceedance_rate_conf": crisis_res["exceed_conf"].mean() if len(crisis_res) > 0 else np.nan,
            "crisis_avg_violation_conf": crisis_res["violation_conf"].mean() if len(crisis_res) > 0 else np.nan,
            "crisis_avg_var_conf": crisis_res["var_conf"].mean() if len(crisis_res) > 0 else np.nan,
        })

        yearly = (
            book_res.assign(year=pd.to_datetime(book_res["date"]).dt.year)
            .groupby("year", as_index=False)
            .agg(
                n_days=("date", "count"),
                empirical_exceedance_rate_base=("exceed_base", "mean"),
                empirical_exceedance_rate_hist=("exceed_hist", "mean"),
                empirical_exceedance_rate_ewma=("exceed_ewma", "mean"),
                empirical_exceedance_rate_caviar=("exceed_caviar", "mean"),
                empirical_exceedance_rate_garch_t=("exceed_garch_t", "mean"),
                empirical_exceedance_rate_conf=("exceed_conf", "mean"),
                avg_loss_norm=("loss_norm_tp1", "mean"),
                avg_q_hat_base_raw=("q_hat_base_raw", "mean"),
                avg_q_hat_ref=("q_hat_ref", "mean"),
                avg_q_hat_core_op=("q_hat_core_op", "mean"),
                avg_q_hat_rep=("q_hat_rep", "mean"),
                avg_q_hat_base=("q_hat_base", "mean"),
                avg_q_hat_hist=("q_hat_hist", "mean"),
                avg_q_hat_ewma=("q_hat_ewma", "mean"),
                avg_q_hat_caviar=("q_hat_caviar", "mean"),
                avg_q_hat_garch_t=("q_hat_garch_t", "mean"),
                avg_buffer_t=("buffer_t", "mean"),
                avg_var_conf_raw=("var_conf_raw", "mean"),
                avg_var_conf=("var_conf", "mean"),
                avg_violation_ewma=("violation_ewma", "mean"),
                avg_violation_caviar=("violation_caviar", "mean"),
                avg_violation_garch_t=("violation_garch_t", "mean"),
                avg_violation_conf=("violation_conf", "mean"),
                avg_pinball_base=("pinball_base", "mean"),
                avg_pinball_hist=("pinball_hist", "mean"),
                avg_pinball_ewma=("pinball_ewma", "mean"),
                avg_pinball_caviar=("pinball_caviar", "mean"),
                avg_pinball_garch_t=("pinball_garch_t", "mean"),
                avg_pinball_conf=("pinball_conf", "mean"),
                quality_strict_pass_share=("quality_pass_strict_economic", "mean"),
                p95_book_quality_max_abs_k_t=("book_quality_max_abs_k_t", lambda x: x.quantile(0.95)),
                p95_book_quality_max_abs_dte_error_t=("book_quality_max_abs_dte_error_t", lambda x: x.quantile(0.95)),
                n_route_a_changed=("route_a_changed", "sum"),
            )
        )
        yearly["experiment_id"] = experiment_id
        yearly["experiment_group"] = experiment_group
        yearly["book_type"] = book_type
        yearly["book_quality_mode"] = str(book_quality_mode)
        yearly_rows.append(yearly)

        if len(crisis_res) > 0:
            crisis_rows.append(pd.DataFrame([{
                "experiment_id": experiment_id,
                "experiment_group": experiment_group,
                "book_type": book_type,
                "crisis_n_days": len(crisis_res),
                "crisis_exceedance_rate_base": crisis_res["exceed_base"].mean(),
                "crisis_exceedance_rate_hist": crisis_res["exceed_hist"].mean(),
                "crisis_exceedance_rate_ewma": crisis_res["exceed_ewma"].mean() if "exceed_ewma" in crisis_res else np.nan,
                "crisis_exceedance_rate_caviar": crisis_res["exceed_caviar"].mean() if "exceed_caviar" in crisis_res else np.nan,
                "crisis_exceedance_rate_garch_t": crisis_res["exceed_garch_t"].mean() if "exceed_garch_t" in crisis_res else np.nan,
                "crisis_exceedance_rate_conf": crisis_res["exceed_conf"].mean(),
                "crisis_avg_loss_norm": crisis_res["loss_norm_tp1"].mean(),
                "crisis_avg_q_hat_base_raw": crisis_res["q_hat_base_raw"].mean(),
                "crisis_avg_q_hat_ref": crisis_res["q_hat_ref"].mean(),
                "crisis_avg_q_hat_core_op": crisis_res["q_hat_core_op"].mean(),
                "crisis_avg_q_hat_rep": crisis_res["q_hat_rep"].mean(),
                "crisis_avg_q_hat_base": crisis_res["q_hat_base"].mean(),
                "crisis_avg_q_hat_hist": crisis_res["q_hat_hist"].mean(),
                "crisis_avg_q_hat_ewma": crisis_res["q_hat_ewma"].mean() if "q_hat_ewma" in crisis_res else np.nan,
                "crisis_avg_q_hat_caviar": crisis_res["q_hat_caviar"].mean() if "q_hat_caviar" in crisis_res else np.nan,
                "crisis_avg_q_hat_garch_t": crisis_res["q_hat_garch_t"].mean() if "q_hat_garch_t" in crisis_res else np.nan,
                "crisis_avg_buffer_t": crisis_res["buffer_t"].mean(),
                "crisis_avg_var_conf_raw": crisis_res["var_conf_raw"].mean(),
                "crisis_avg_var_conf": crisis_res["var_conf"].mean(),
                "crisis_avg_violation_conf": crisis_res["violation_conf"].mean(),
                "marking_mode": marking_mode,
                "book_quality_mode_spec": str(book_quality_mode),
                "var_floor_value": np.nan if var_floor_value is None else float(var_floor_value),
                "base_model_kind_requested": model_kind_requested,
            }]))

    results_df = pd.concat(result_frames, axis=0, ignore_index=True) if len(result_frames) > 0 else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["experiment_group", "experiment_id", "book_type"]).reset_index(drop=True)
    yearly_df = pd.concat(yearly_rows, axis=0, ignore_index=True) if len(yearly_rows) > 0 else pd.DataFrame()
    crisis_df = pd.concat(crisis_rows, axis=0, ignore_index=True) if len(crisis_rows) > 0 else pd.DataFrame()

    rolling_cols = ["experiment_id", "experiment_group", "book_type", "date"] + [c for c in results_df.columns if c.startswith("roll")]
    rolling_df = results_df[rolling_cols].copy() if len(results_df) > 0 else pd.DataFrame()

    for _df in [results_df, summary_df, yearly_df, crisis_df, rolling_df]:
        if len(_df) > 0:
            if "asset_ticker" not in _df.columns:
                _df.insert(0, "asset_ticker", TICKER)
            if "asset_secid" not in _df.columns:
                _df.insert(1, "asset_secid", ASSET_SECID if ASSET_SECID is not None else np.nan)

    return results_df, summary_df, yearly_df, crisis_df, rolling_df


def build_marking_intersection_tables(results_df: pd.DataFrame,
                                     learner: str = PRIMARY_LEARNER,
                                     floor_value: Optional[float] = PRIMARY_VAR_FLOOR,
                                     book_quality_mode: str = PRIMARY_BOOK_QUALITY_MODE):
    """Same-date strict-vs-robust marking comparison.

    This table is meant to answer the reviewer question: does robust marking create a
    systematic loss or calibration distortion, or does it mainly recover feasible dates?
    """
    if len(results_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["next_date"] = pd.to_datetime(df["next_date"])

    floor_cmp = -999999.0 if floor_value is None else float(floor_value)
    df["var_floor_cmp"] = pd.to_numeric(df["var_floor_value"], errors="coerce").fillna(-999999.0)
    if "book_quality_mode" not in df.columns:
        df["book_quality_mode"] = PRIMARY_BOOK_QUALITY_MODE

    base_mask = (
        (df["base_model_kind_requested"] == learner) &
        (df["var_floor_cmp"] == floor_cmp) &
        (df["book_quality_mode"].astype(str) == str(book_quality_mode))
    )
    strict_mask = (
        df["experiment_group"].astype(str).eq("marking_sensitivity") &
        df["marking_mode"].astype(str).eq("strict_exact_contract")
    )
    robust_mask = (
        df["experiment_group"].astype(str).isin(["main", "marking_sensitivity"]) &
        df["marking_mode"].astype(str).eq("robust_all")
    )
    use = df[base_mask & (strict_mask | robust_mask)].copy()

    if len(use) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # If both main and marking_sensitivity contain the same robust specification, keep one copy.
    use["_source_priority"] = np.where(use["experiment_group"].astype(str).eq("main"), 0, 1)
    use = use.sort_values(["_source_priority", "book_type", "date", "next_date"]).drop_duplicates(
        ["book_type", "date", "next_date", "marking_mode"], keep="first"
    )

    key = ["book_type", "date", "next_date"]
    value_cols = [
        "loss_norm_tp1", "raw_loss_tp1",
        "q_hat_base_raw", "q_hat_ref", "q_hat_core_op", "q_hat_rep",
        "q_hat_base", "q_hat_hist", "q_hat_ewma", "q_hat_caviar", "q_hat_garch_t",
        "buffer_t", "var_conf_raw", "var_conf",
        "exceed_base", "exceed_hist", "exceed_ewma", "exceed_caviar", "exceed_garch_t", "exceed_conf",
        "violation_base", "violation_hist", "violation_ewma", "violation_caviar", "violation_garch_t", "violation_conf",
        "pinball_base", "pinball_hist", "pinball_ewma", "pinball_caviar", "pinball_garch_t", "pinball_conf",
        "n_option_legs", "n_option_mark_exact_t", "n_option_mark_contract_t", "n_option_mark_interp_t", "n_option_mark_nearest_t", "n_option_mark_fallback_t",
        "quality_pass_strict_economic", "book_quality_max_abs_k_t", "book_quality_max_abs_dte_error_t",
        "route_a_changed",
    ]
    value_cols = [c for c in value_cols if c in use.columns]

    strict = (
        use[use["marking_mode"] == "strict_exact_contract"][key + value_cols]
        .rename(columns={c: f"{c}_strict" for c in value_cols})
    )
    robust = (
        use[use["marking_mode"] == "robust_all"][key + value_cols]
        .rename(columns={c: f"{c}_robust" for c in value_cols})
    )

    inter = strict.merge(robust, on=key, how="inner")
    if len(inter) == 0:
        return pd.DataFrame(), pd.DataFrame()

    inter["loss_diff"] = inter["loss_norm_tp1_robust"] - inter["loss_norm_tp1_strict"]
    inter["abs_loss_diff"] = np.abs(inter["loss_diff"])
    inter["conf_diff"] = inter["q_hat_rep_robust"] - inter["q_hat_rep_strict"]
    inter["abs_conf_diff"] = np.abs(inter["conf_diff"])
    if "pinball_conf_robust" in inter.columns and "pinball_conf_strict" in inter.columns:
        inter["pinball_conf_diff"] = inter["pinball_conf_robust"] - inter["pinball_conf_strict"]
    else:
        inter["pinball_conf_diff"] = np.nan
    inter["conf_exceed_switch"] = (inter["exceed_conf_robust"] != inter["exceed_conf_strict"]).astype(int)
    inter["base_exceed_switch"] = (inter["exceed_base_robust"] != inter["exceed_base_strict"]).astype(int)

    def _q(s, prob):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return float(s.quantile(prob)) if len(s) else np.nan

    def _summarize(g: pd.DataFrame, label: str) -> dict:
        denom_marks = pd.to_numeric(g.get("n_option_legs_robust", pd.Series(dtype=float)), errors="coerce").sum()
        approx_marks = (
            pd.to_numeric(g.get("n_option_mark_interp_t_robust", pd.Series(dtype=float)), errors="coerce").sum() +
            pd.to_numeric(g.get("n_option_mark_nearest_t_robust", pd.Series(dtype=float)), errors="coerce").sum()
        )
        loss_corr = np.nan
        if len(g) > 2:
            a = pd.to_numeric(g["loss_norm_tp1_strict"], errors="coerce")
            b = pd.to_numeric(g["loss_norm_tp1_robust"], errors="coerce")
            if a.notna().sum() > 2 and b.notna().sum() > 2:
                loss_corr = float(a.corr(b))
        return {
            "book_type": label,
            "n_intersection": int(len(g)),
            "strict_base_exc": float(g["exceed_base_strict"].mean()),
            "robust_base_exc": float(g["exceed_base_robust"].mean()),
            "strict_conf_exc": float(g["exceed_conf_strict"].mean()),
            "robust_conf_exc": float(g["exceed_conf_robust"].mean()),
            "robust_minus_strict_conf_exc": float(g["exceed_conf_robust"].mean() - g["exceed_conf_strict"].mean()),
            "strict_conf_viol": float(g["violation_conf_strict"].mean()),
            "robust_conf_viol": float(g["violation_conf_robust"].mean()),
            "strict_conf_pinball": float(g["pinball_conf_strict"].mean()) if "pinball_conf_strict" in g else np.nan,
            "robust_conf_pinball": float(g["pinball_conf_robust"].mean()) if "pinball_conf_robust" in g else np.nan,
            "mean_loss_diff": float(g["loss_diff"].mean()),
            "median_loss_diff": _q(g["loss_diff"], 0.50),
            "q05_loss_diff": _q(g["loss_diff"], 0.05),
            "q95_loss_diff": _q(g["loss_diff"], 0.95),
            "mean_abs_loss_diff": float(g["abs_loss_diff"].mean()),
            "p90_abs_loss_diff": _q(g["abs_loss_diff"], 0.90),
            "p95_abs_loss_diff": _q(g["abs_loss_diff"], 0.95),
            "max_abs_loss_diff": float(g["abs_loss_diff"].max()),
            "mean_abs_conf_diff": float(g["abs_conf_diff"].mean()),
            "p95_abs_conf_diff": _q(g["abs_conf_diff"], 0.95),
            "max_abs_conf_diff": float(g["abs_conf_diff"].max()),
            "mean_pinball_conf_diff": float(g["pinball_conf_diff"].mean()) if "pinball_conf_diff" in g else np.nan,
            "conf_exceed_switch_rate": float(g["conf_exceed_switch"].mean()),
            "base_exceed_switch_rate": float(g["base_exceed_switch"].mean()),
            "loss_corr_strict_robust": loss_corr,
            "robust_approx_mark_share_intersection": float(approx_marks / denom_marks) if denom_marks > 0 else np.nan,
            "book_quality_mode": str(book_quality_mode),
        }

    rows = [_summarize(g, book) for book, g in inter.groupby("book_type", sort=False)]
    rows.append(_summarize(inter, "pooled"))
    summary = pd.DataFrame(rows)
    return inter, summary



def _var_method_specs() -> List[Tuple[str, str, str]]:
    return [
        ("Historical VaR", "hist", "q_hat_hist"),
        ("EWMA Historical VaR", "ewma", "q_hat_ewma"),
        ("CAViaR", "caviar", "q_hat_caviar"),
        ("GARCH-t VaR", "garch_t", "q_hat_garch_t"),
        ("LightGBM Quantile", "base", "q_hat_base"),
        ("LightGBM + Calibration", "conf", "var_conf"),
    ]


def build_baseline_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format table comparing VaR methods on coverage, violation, pinball loss, and threshold efficiency."""
    if results_df is None or len(results_df) == 0:
        return pd.DataFrame()

    df = results_df.copy()
    df = df[df["experiment_group"].astype(str).eq("main")].copy()
    if len(df) == 0:
        return pd.DataFrame()
    if "book_quality_mode" not in df.columns:
        df["book_quality_mode"] = PRIMARY_BOOK_QUALITY_MODE

    rows = []
    methods = _var_method_specs()

    def _add_rows(g: pd.DataFrame, book_label: str):
        for method_name, suffix, q_col in methods:
            exc_col = f"exceed_{suffix}"
            viol_col = f"violation_{suffix}"
            pinball_col = f"pinball_{suffix}"
            roll_col = f"roll50_exceed_{suffix}"
            if exc_col not in g.columns:
                continue
            valid = np.isfinite(g[exc_col].to_numpy(dtype=float))
            if valid.sum() == 0:
                continue
            gv = g.loc[valid].copy()
            crisis = gv.loc[gv.get("is_crisis", 0) == 1]
            avg_violation = float(np.nanmean(gv[viol_col])) if viol_col in gv.columns else np.nan
            exc_rate = float(np.nanmean(gv[exc_col]))
            rows.append({
                "book_type": book_label,
                "method": method_name,
                "method_suffix": suffix,
                "n_backtest_days": int(valid.sum()),
                "target_exceedance_alpha": ALPHA,
                "empirical_exceedance_rate": exc_rate,
                "exceedance_gap": exc_rate - ALPHA,
                "abs_exceedance_gap": abs(exc_rate - ALPHA),
                "avg_violation": avg_violation,
                "avg_violation_per_exceedance": float(avg_violation / exc_rate) if exc_rate > 0 and np.isfinite(avg_violation) else np.nan,
                "avg_pinball_loss": float(np.nanmean(gv[pinball_col])) if pinball_col in gv.columns else np.nan,
                "max_roll50_exceedance": float(np.nanmax(g[roll_col])) if roll_col in g.columns and np.isfinite(g[roll_col]).any() else np.nan,
                "crisis_n_days": int(len(crisis)),
                "crisis_exceedance_rate": float(np.nanmean(crisis[exc_col])) if len(crisis) else np.nan,
                "avg_threshold": float(np.nanmean(gv[q_col])) if q_col in gv.columns and np.isfinite(gv[q_col]).any() else np.nan,
                "std_threshold": float(np.nanstd(gv[q_col])) if q_col in gv.columns and np.isfinite(gv[q_col]).any() else np.nan,
                "p05_threshold": float(np.nanquantile(gv[q_col], 0.05)) if q_col in gv.columns and np.isfinite(gv[q_col]).any() else np.nan,
                "p95_threshold": float(np.nanquantile(gv[q_col], 0.95)) if q_col in gv.columns and np.isfinite(gv[q_col]).any() else np.nan,
                "book_quality_mode": gv["book_quality_mode"].astype(str).mode().iloc[0] if "book_quality_mode" in gv else PRIMARY_BOOK_QUALITY_MODE,
            })

    for book, g_book in df.groupby("book_type", sort=False):
        _add_rows(g_book, book)
    _add_rows(df, "pooled")

    out = pd.DataFrame(rows)
    if len(out) > 0:
        method_order = {m[0]: i for i, m in enumerate(methods)}
        book_order = {b: i for i, b in enumerate(list(MAIN_BOOK_TYPES) + ["pooled"])}
        out["_book_order"] = out["book_type"].map(book_order).fillna(999)
        out["_method_order"] = out["method"].map(method_order).fillna(999)
        out = out.sort_values(["_book_order", "_method_order"]).drop(columns=["_book_order", "_method_order"]).reset_index(drop=True)
    return out


def _safe_loglik_bernoulli(successes: int, failures: int, p: float) -> float:
    if successes < 0 or failures < 0:
        return np.nan
    if p <= 0.0:
        return 0.0 if successes == 0 else -np.inf
    if p >= 1.0:
        return 0.0 if failures == 0 else -np.inf
    return float(successes * np.log(p) + failures * np.log(1.0 - p))


def kupiec_christoffersen_tests(exceedances, alpha: float = ALPHA) -> dict:
    """Kupiec unconditional coverage and Christoffersen independence/conditional coverage tests."""
    e = pd.to_numeric(pd.Series(exceedances), errors="coerce").dropna().astype(int).to_numpy()
    e = e[(e == 0) | (e == 1)]
    n = int(len(e))
    x = int(e.sum())
    out = {
        "n": n,
        "n_exceed": x,
        "exceedance_rate": float(x / n) if n > 0 else np.nan,
        "lr_uc": np.nan,
        "p_uc": np.nan,
        "n00": np.nan, "n01": np.nan, "n10": np.nan, "n11": np.nan,
        "lr_ind": np.nan,
        "p_ind": np.nan,
        "lr_cc": np.nan,
        "p_cc": np.nan,
    }
    if n == 0:
        return out

    p_hat = x / n
    ll_null = _safe_loglik_bernoulli(x, n - x, alpha)
    ll_alt = _safe_loglik_bernoulli(x, n - x, p_hat)
    if np.isfinite(ll_null) and np.isfinite(ll_alt):
        lr_uc = max(0.0, 2.0 * (ll_alt - ll_null))
        out["lr_uc"] = float(lr_uc)
        out["p_uc"] = float(1.0 - chi2.cdf(lr_uc, df=1))

    if n >= 2:
        prev = e[:-1]
        curr = e[1:]
        n00 = int(((prev == 0) & (curr == 0)).sum())
        n01 = int(((prev == 0) & (curr == 1)).sum())
        n10 = int(((prev == 1) & (curr == 0)).sum())
        n11 = int(((prev == 1) & (curr == 1)).sum())
        out.update({"n00": n00, "n01": n01, "n10": n10, "n11": n11})
        denom0 = n00 + n01
        denom1 = n10 + n11
        denom = denom0 + denom1
        pi = (n01 + n11) / denom if denom > 0 else np.nan
        pi0 = n01 / denom0 if denom0 > 0 else np.nan
        pi1 = n11 / denom1 if denom1 > 0 else np.nan

        ll_markov_null = _safe_loglik_bernoulli(n01 + n11, n00 + n10, pi) if np.isfinite(pi) else np.nan
        ll_markov_alt = 0.0
        valid_alt = True
        if denom0 > 0 and np.isfinite(pi0):
            ll_markov_alt += _safe_loglik_bernoulli(n01, n00, pi0)
        elif denom0 > 0:
            valid_alt = False
        if denom1 > 0 and np.isfinite(pi1):
            ll_markov_alt += _safe_loglik_bernoulli(n11, n10, pi1)
        elif denom1 > 0:
            valid_alt = False

        if valid_alt and np.isfinite(ll_markov_null) and np.isfinite(ll_markov_alt):
            lr_ind = max(0.0, 2.0 * (ll_markov_alt - ll_markov_null))
            out["lr_ind"] = float(lr_ind)
            out["p_ind"] = float(1.0 - chi2.cdf(lr_ind, df=1))
            if np.isfinite(out["lr_uc"]):
                lr_cc = float(out["lr_uc"] + lr_ind)
                out["lr_cc"] = lr_cc
                out["p_cc"] = float(1.0 - chi2.cdf(lr_cc, df=2))
    return out


def build_backtest_test_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Kupiec and Christoffersen test table for every experiment, book, and VaR method."""
    if results_df is None or len(results_df) == 0:
        return pd.DataFrame()
    df = results_df.copy()
    if "book_quality_mode" not in df.columns:
        df["book_quality_mode"] = PRIMARY_BOOK_QUALITY_MODE

    rows = []
    methods = _var_method_specs()
    group_cols = ["experiment_group", "experiment_id", "book_quality_mode", "book_type"]

    def _append_for_group(g: pd.DataFrame, book_label: str):
        if "date" in g.columns:
            g = g.sort_values("date").copy()
        meta = {c: g[c].iloc[0] for c in ["experiment_group", "experiment_id", "book_quality_mode"] if c in g.columns}
        for method_name, suffix, _ in methods:
            exc_col = f"exceed_{suffix}"
            if exc_col not in g.columns:
                continue
            valid = pd.to_numeric(g[exc_col], errors="coerce").dropna()
            if len(valid) == 0:
                continue
            test = kupiec_christoffersen_tests(valid, alpha=ALPHA)
            independence_applicable = int(book_label != "pooled")
            if not independence_applicable:
                # Pooled rows mix different books, so transition-based independence is not meaningful.
                # Keep Kupiec unconditional coverage and suppress Christoffersen transition tests.
                for k in ["n00", "n01", "n10", "n11", "lr_ind", "p_ind", "lr_cc", "p_cc"]:
                    test[k] = np.nan
            rows.append({
                **meta,
                "book_type": book_label,
                "method": method_name,
                "method_suffix": suffix,
                "target_exceedance_alpha": ALPHA,
                **test,
                "independence_test_applicable": independence_applicable,
                "reject_uc_5pct": int(np.isfinite(test.get("p_uc", np.nan)) and test["p_uc"] < 0.05),
                "reject_ind_5pct": int(np.isfinite(test.get("p_ind", np.nan)) and test["p_ind"] < 0.05),
                "reject_cc_5pct": int(np.isfinite(test.get("p_cc", np.nan)) and test["p_cc"] < 0.05),
            })

    for _, g in df.groupby(group_cols, sort=False):
        _append_for_group(g, g["book_type"].iloc[0])

    pooled_cols = ["experiment_group", "experiment_id", "book_quality_mode"]
    for _, g in df.groupby(pooled_cols, sort=False):
        _append_for_group(g, "pooled")

    out = pd.DataFrame(rows)
    if len(out) > 0:
        method_order = {m[0]: i for i, m in enumerate(methods)}
        book_order = {b: i for i, b in enumerate(list(MAIN_BOOK_TYPES) + ["pooled"])}
        out["_book_order"] = out["book_type"].map(book_order).fillna(999)
        out["_method_order"] = out["method"].map(method_order).fillna(999)
        out = out.sort_values(["experiment_group", "experiment_id", "_book_order", "_method_order"]).drop(columns=["_book_order", "_method_order"]).reset_index(drop=True)
    return out


def build_book_quality_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table for whether the selected books are actually close to the intended economic targets."""
    if results_df is None or len(results_df) == 0:
        return pd.DataFrame()
    df = results_df.copy()
    if "book_quality_mode" not in df.columns:
        df["book_quality_mode"] = PRIMARY_BOOK_QUALITY_MODE

    def _q(g, col, p):
        if col not in g.columns:
            return np.nan
        s = pd.to_numeric(g[col], errors="coerce").dropna()
        return float(s.quantile(p)) if len(s) else np.nan

    rows = []
    for keys, g in df.groupby(["experiment_group", "experiment_id", "book_quality_mode", "book_type"], sort=False):
        experiment_group, experiment_id, qmode, book = keys
        row = {
            "experiment_group": experiment_group,
            "experiment_id": experiment_id,
            "book_quality_mode": qmode,
            "book_type": book,
            "n_backtest_days": int(len(g)),
            "strict_quality_pass_share": float(g["quality_pass_strict_economic"].mean()) if "quality_pass_strict_economic" in g else np.nan,
            "median_max_abs_k": _q(g, "book_quality_max_abs_k_t", 0.50),
            "p90_max_abs_k": _q(g, "book_quality_max_abs_k_t", 0.90),
            "p95_max_abs_k": _q(g, "book_quality_max_abs_k_t", 0.95),
            "median_max_abs_dte_error": _q(g, "book_quality_max_abs_dte_error_t", 0.50),
            "p95_max_abs_dte_error": _q(g, "book_quality_max_abs_dte_error_t", 0.95),
            "median_atm_max_abs_k": _q(g, "atm_max_abs_k_t", 0.50),
            "p95_atm_max_abs_k": _q(g, "atm_max_abs_k_t", 0.95),
            "median_rr_max_delta_error": _q(g, "rr_max_delta_error_t", 0.50),
            "p95_rr_max_delta_error": _q(g, "rr_max_delta_error_t", 0.95),
            "median_putspread_max_delta_error": _q(g, "putspread_max_delta_error_t", 0.50),
            "p95_putspread_max_delta_error": _q(g, "putspread_max_delta_error_t", 0.95),
            "empirical_exceedance_rate_conf": float(g["exceed_conf"].mean()) if "exceed_conf" in g else np.nan,
            "avg_pinball_conf": float(g["pinball_conf"].mean()) if "pinball_conf" in g else np.nan,
            "avg_threshold_conf": float(g["var_conf"].mean()) if "var_conf" in g else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)



def ensure_result_diagnostic_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    """Make older result CSVs usable for the new diagnostic tables."""
    df = results_df.copy()
    if "book_quality_mode" not in df.columns:
        df["book_quality_mode"] = PRIMARY_BOOK_QUALITY_MODE
    if "quality_pass_strict_economic" not in df.columns:
        df["quality_pass_strict_economic"] = np.nan
    method_to_q = {
        "hist": "q_hat_hist",
        "ewma": "q_hat_ewma",
        "caviar": "q_hat_caviar",
        "garch_t": "q_hat_garch_t",
        "base": "q_hat_base",
        "conf": "var_conf",
    }
    if "loss_norm_tp1" in df.columns:
        y = pd.to_numeric(df["loss_norm_tp1"], errors="coerce")
        for suffix, q_col in method_to_q.items():
            pin_col = f"pinball_{suffix}"
            if pin_col not in df.columns and q_col in df.columns:
                q = pd.to_numeric(df[q_col], errors="coerce")
                u = y - q
                df[pin_col] = np.where(u >= 0.0, QUANTILE_LEVEL * u, (QUANTILE_LEVEL - 1.0) * u)
            viol_col = f"violation_{suffix}"
            if viol_col not in df.columns and q_col in df.columns:
                q = pd.to_numeric(df[q_col], errors="coerce")
                df[viol_col] = np.maximum(y - q, 0.0)
            exc_col = f"exceed_{suffix}"
            if exc_col not in df.columns and q_col in df.columns:
                q = pd.to_numeric(df[q_col], errors="coerce")
                valid_pair = np.isfinite(y) & np.isfinite(q)
                df[exc_col] = np.where(valid_pair, (y > q).astype(float), np.nan)
    return df


def build_diagnostic_tables_from_existing_results(results_path: Optional[Path] = None) -> pd.DataFrame:
    """Fast path for already-generated result CSVs.

    This creates Kupiec/Christoffersen, pinball/threshold, and marking-sensitivity tables
    without rebuilding option books. It cannot create the strict economic-book robustness
    sample because that requires rebuilding the books from raw option chains.
    """
    if results_path is None:
        results_path = OUT_DIR / "book_var_results_v25_paper_ready.csv"
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"results CSV not found: {results_path}")
    results_df = pd.read_csv(results_path, low_memory=False)
    results_df = ensure_result_diagnostic_columns(results_df)

    baseline_comparison_df = build_baseline_comparison_table(results_df)
    backtest_tests_df = build_backtest_test_table(results_df)
    book_quality_summary_df = build_book_quality_summary(results_df)
    inter_df, inter_summary_df = build_marking_intersection_tables(results_df)

    baseline_path = OUT_DIR / "book_var_baseline_comparison_v25.csv"
    tests_path = OUT_DIR / "book_var_backtest_tests_v25.csv"
    quality_path = OUT_DIR / "book_var_book_quality_summary_v25.csv"
    inter_path = OUT_DIR / "book_var_marking_intersection_dates_v25.csv"
    inter_summary_path = OUT_DIR / "book_var_marking_intersection_summary_v25.csv"

    baseline_comparison_df.to_csv(baseline_path, index=False)
    backtest_tests_df.to_csv(tests_path, index=False)
    book_quality_summary_df.to_csv(quality_path, index=False)
    if len(inter_df) > 0:
        inter_df.to_csv(inter_path, index=False)
        inter_summary_df.to_csv(inter_summary_path, index=False)

    print(f"Saved to: {baseline_path}")
    print(f"Saved to: {tests_path}")
    print(f"Saved to: {quality_path}")
    if len(inter_df) > 0:
        print(f"Saved to: {inter_path}")
        print(f"Saved to: {inter_summary_path}")
    return results_df


def run_second_layer_book_var_v25_paper_ready():
    print(f"Running option-book VaR pipeline for {TICKER}, secid={ASSET_SECID}, base_dir={BASE_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Settings: settlement_policy={SETTLEMENT_POLICY}, require_european_style={REQUIRE_EUROPEAN_STYLE}, use_vendor_forward_price={USE_VENDOR_FORWARD_PRICE}")
    feat_df, state_cols, daily_chain_map = collect_state_and_daily_chains()
    feature_audit_path = write_feature_column_audit(get_feature_cols(state_cols), OUT_DIR, state_cols=state_cols)
    print(f"Saved feature-column audit to: {feature_audit_path}")
    specs = build_experiment_specs()

    all_results = []
    all_summaries = []
    all_yearlies = []
    all_crises = []
    all_rollings = []

    for spec in specs:
        results_df, summary_df, yearly_df, crisis_df, rolling_df = run_single_experiment(feat_df, state_cols, daily_chain_map, spec)
        if len(results_df) > 0:
            all_results.append(results_df)
        if len(summary_df) > 0:
            all_summaries.append(summary_df)
        if len(yearly_df) > 0:
            all_yearlies.append(yearly_df)
        if len(crisis_df) > 0:
            all_crises.append(crisis_df)
        if len(rolling_df) > 0:
            all_rollings.append(rolling_df)

    if len(all_results) == 0:
        raise ValueError("V25 generated no valid results. Check the data, book construction, or next-day matching.")

    results_df = pd.concat(all_results, axis=0, ignore_index=True).sort_values(["experiment_group", "experiment_id", "book_type", "date"]).reset_index(drop=True)
    summary_df = pd.concat(all_summaries, axis=0, ignore_index=True).sort_values(["experiment_group", "experiment_id", "book_type"]).reset_index(drop=True)
    yearly_df = pd.concat(all_yearlies, axis=0, ignore_index=True).sort_values(["experiment_group", "experiment_id", "book_type", "year"]).reset_index(drop=True) if len(all_yearlies) > 0 else pd.DataFrame()
    crisis_df = pd.concat(all_crises, axis=0, ignore_index=True).sort_values(["experiment_group", "experiment_id", "book_type"]).reset_index(drop=True) if len(all_crises) > 0 else pd.DataFrame()
    rolling_df = pd.concat(all_rollings, axis=0, ignore_index=True).sort_values(["experiment_group", "experiment_id", "book_type", "date"]).reset_index(drop=True) if len(all_rollings) > 0 else pd.DataFrame()

    results_path = OUT_DIR / "book_var_results_v25_paper_ready.csv"
    summary_path = OUT_DIR / "book_var_summary_v25_paper_ready.csv"
    yearly_path = OUT_DIR / "book_var_yearly_v25_paper_ready.csv"
    crisis_path = OUT_DIR / "book_var_crisis_v25_paper_ready.csv"
    rolling_path = OUT_DIR / "book_var_rolling_v25_paper_ready.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    crisis_df.to_csv(crisis_path, index=False)
    rolling_df.to_csv(rolling_path, index=False)

    baseline_comparison_df = build_baseline_comparison_table(results_df)
    baseline_comparison_path = OUT_DIR / "book_var_baseline_comparison_v25.csv"
    baseline_comparison_df.to_csv(baseline_comparison_path, index=False)
    print(f"Saved to: {baseline_comparison_path}")

    backtest_tests_df = build_backtest_test_table(results_df)
    backtest_tests_path = OUT_DIR / "book_var_backtest_tests_v25.csv"
    backtest_tests_df.to_csv(backtest_tests_path, index=False)
    print(f"Saved to: {backtest_tests_path}")

    book_quality_summary_df = build_book_quality_summary(results_df)
    book_quality_summary_path = OUT_DIR / "book_var_book_quality_summary_v25.csv"
    book_quality_summary_df.to_csv(book_quality_summary_path, index=False)
    print(f"Saved to: {book_quality_summary_path}")

    # Apples-to-apples intersection sample for strict vs robust marking
    inter_df, inter_summary_df = build_marking_intersection_tables(results_df)
    if len(inter_df) > 0:
        inter_path = OUT_DIR / "book_var_marking_intersection_dates_v25.csv"
        inter_summary_path = OUT_DIR / "book_var_marking_intersection_summary_v25.csv"
        inter_df.to_csv(inter_path, index=False)
        inter_summary_df.to_csv(inter_summary_path, index=False)
        print(f"Saved to: {inter_path}")
        print(f"Saved to: {inter_summary_path}")

    # Diagnostics for late-sample thinning
    build_yearly_filter_attrition_report()
    build_book_marking_diagnostics(daily_chain_map)

    print("\n========== V25 paper-ready second-layer experiment completed ==========")
    print(summary_df)
    print(f"\nSaved to: {results_path}")
    print(f"Saved to: {summary_path}")
    print(f"Saved to: {yearly_path}")
    print(f"Saved to: {crisis_path}")
    print(f"Saved to: {rolling_path}")
    print(f"Saved to: {backtest_tests_path}")
    print(f"Saved to: {book_quality_summary_path}")
    print(f"Saved feature-column audit to: {feature_audit_path}")


if __name__ == "__main__":
    run_second_layer_book_var_v25_paper_ready()
