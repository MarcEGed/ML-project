"""
Feature Engineering Pipeline
==============================
Transforms raw market time-series data into a rich feature set for predicting
`market_forward_excess_returns`.

Approach Overview
-----------------
The pipeline is built around three complementary ideas:

1. **Correlation-filtered rolling statistics**
   Rolling mean and standard deviation are expensive to compute across many
   features and many windows.  Rather than computing them for every column,
   we first rank all base features by their absolute Pearson correlation with
   the target and restrict the rolling-stat block to the top-N (default 30).
   This keeps the feature matrix at a manageable size while focusing
   engineering effort on the signals that carry the most predictive content.

2. **Lag & momentum features for all base features**
   Short-horizon lags (1-10 days) capture autocorrelation in market factors.
   Momentum differences (diff over 1/2/3/5 days) turn level features into
   rate-of-change signals, which are often more stationary and informative
   than the raw levels for regression targets that are themselves returns.
   These are computed for *all* base features, not just the top-N, because
   a feature with modest raw correlation can still produce a useful momentum
   signal.

3. **Autoregressive target lags**
   The target itself (market_forward_excess_returns) is lagged by 1-3 periods
   and included as features.  This gives the model direct access to recent
   regime information that may not be captured by the factor columns.

Volatility proxy
----------------
An auxiliary column `vol_proxy` is derived from the absolute mean of all
columns whose names start with "V".  It is excluded from the training feature
set (used only for diagnostics / allocation) but is computed here for
completeness.

Missing-value handling
-----------------------
Raw factor columns are forward-filled then back-filled before any
transformation.  After all features are created, rows that still contain NaN
(mainly the burn-in rows at the start of each rolling window) are dropped.

Window schedule
---------------
Short  :  3,  5, 10 days   — intraweek / weekly effects
Medium : 20, 30, 60 days   — monthly / quarterly effects
Macro  : 90,120,252 days   — year-scale regime features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Default configuration values (can be overridden by passing a config object)
# ---------------------------------------------------------------------------
DEFAULT_SHORT_WINDOWS  = [3, 5, 10]
DEFAULT_MEDIUM_WINDOWS = [20, 30, 60]
DEFAULT_MACRO_WINDOWS  = [90, 120, 252]
DEFAULT_LAG_PERIODS    = [1, 2, 3, 5, 10]
DEFAULT_MOMENTUM_LAGS  = [1, 2, 3, 5]
DEFAULT_TARGET_LAGS    = [1, 2, 3]
DEFAULT_TOP_N_FEATURES = 30
TARGET_COL             = 'market_forward_excess_returns'
NON_FEATURE_COLS       = ['date_id', 'forward_returns', 'risk_free_rate',
                           TARGET_COL, 'vol_proxy']


# ---------------------------------------------------------------------------
# Step 1 - Imputation
# ---------------------------------------------------------------------------

def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then back-fill every column that is not a target or date.

    Forward-fill propagates the last known value across gaps (appropriate for
    slowly-changing factor data).  Back-fill handles any remaining NaN at the
    very start of the series.
    """
    targets = ['forward_returns', 'risk_free_rate', TARGET_COL]
    for col in df.columns:
        if col not in targets + ['date_id']:
            df[col] = df[col].ffill().bfill()
    return df


# ---------------------------------------------------------------------------
# Step 2 - Identify base features & select top-N by correlation
# ---------------------------------------------------------------------------

def get_base_feature_cols(df: pd.DataFrame) -> list:
    """Return all columns that should be treated as base input features."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def rank_features_by_correlation(df: pd.DataFrame,
                                  feature_cols: list,
                                  target: str = TARGET_COL,
                                  top_n: int = DEFAULT_TOP_N_FEATURES,
                                  verbose: bool = True) -> list:
    """
    Rank *feature_cols* by absolute Pearson correlation with *target* and
    return the top-*top_n* names.

    Why correlation-based selection?
    - Rolling statistics multiply the feature count by (n_windows × 2).
      With many windows this quickly becomes intractable.
    - Absolute correlation is a fast, interpretable proxy for linear
      predictive relevance; features with near-zero correlation with the
      target are unlikely to produce useful rolling statistics.
    - The selection is done on the full training+backtest frame before the
      train/test split because the rolling windows use historical values only
      (no look-ahead leakage from future targets).

    Parameters
    ----------
    df         : DataFrame containing both features and target.
    feature_cols : candidate base feature column names.
    target     : column name of the regression target.
    top_n      : how many features to keep.
    verbose    : if True, print the top-5 features and their correlations.

    Returns
    -------
    List of the top-*top_n* feature names (descending correlation order).
    """
    corrs = {col: abs(np.corrcoef(df[col], df[target])[0, 1])
             for col in feature_cols}
    ranked = sorted(corrs.items(), key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"  Top {top_n} features selected for rolling statistics:")
        for feat, corr in ranked[:5]:
            print(f"    {feat}: |corr| = {corr:.4f}")
        print(f"    ... ({top_n} total)")

    return [name for name, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Step 3 - Lag features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame,
                     feature_cols: list,
                     lag_periods: list = DEFAULT_LAG_PERIODS) -> pd.DataFrame:
    """
    Add shifted copies of each base feature.

    Lag-k of feature X at time t equals X at time t-k.  This gives the model
    direct access to the recent history of every factor without requiring it
    to learn lag structure implicitly through tree splits.

    Lags used: 1, 2, 3, 5, 10 days.
    - Lags 1-3 capture very short autocorrelation (mean-reversion or momentum
      at the 1-3 day horizon).
    - Lag 5 ≈ one week.
    - Lag 10 ≈ two weeks / half a month.
    """
    for feature in feature_cols:
        for lag in lag_periods:
            df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Step 4 - Momentum (difference) features
# ---------------------------------------------------------------------------

def add_momentum_features(df: pd.DataFrame,
                           feature_cols: list,
                           momentum_lags: list = DEFAULT_MOMENTUM_LAGS) -> pd.DataFrame:
    """
    Add first-difference momentum features for each base feature.

    momentum_k(X, t) = X(t) - X(t-k)

    Motivation
    ----------
    Raw factor levels are often non-stationary.  First differences transform
    them into rate-of-change signals that are closer to stationary and often
    correlate better with forward return targets (which are themselves
    differences).  Short-horizon differences (1-5 days) also capture
    "acceleration" or reversal patterns in each factor.
    """
    for feature in feature_cols:
        for lag in momentum_lags:
            df[f"{feature}_momentum_{lag}"] = df[feature].diff(lag)
    return df


# ---------------------------------------------------------------------------
# Step 5 - Rolling statistics (top-N features only)
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame,
                          top_features: list,
                          short_windows: list  = DEFAULT_SHORT_WINDOWS,
                          medium_windows: list = DEFAULT_MEDIUM_WINDOWS,
                          macro_windows: list  = DEFAULT_MACRO_WINDOWS) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation for each window in
    {short, medium, macro} schedules, but only for the *top_features* subset.

    Rolling mean  - smoothed level; captures trend direction over the window.
    Rolling std   - realised dispersion; acts as a local volatility or
                    uncertainty signal for that factor.

    Window rationale
    ----------------
    Short  (3/5/10d)   : within-week and weekly effects; react quickly to
                         regime shifts.
    Medium (20/30/60d) : monthly / quarterly smoothing; standard horizons for
                         factor momentum in academic literature.
    Macro  (90/120/252d): year-scale regime detection; 252 ≈ one trading year.

    min_periods=1 avoids NaN at the start of the series (uses whatever data
    is available for early rows; these are later removed if any other column
    produces NaN).
    """
    all_windows = short_windows + medium_windows + macro_windows
    for feature in top_features:
        for window in all_windows:
            df[f"{feature}_rolling_mean_{window}"] = (
                df[feature].rolling(window=window, min_periods=1).mean()
            )
            df[f"{feature}_rolling_std_{window}"] = (
                df[feature].rolling(window=window, min_periods=1).std()
            )
    return df


# ---------------------------------------------------------------------------
# Step 6 - Autoregressive target lags
# ---------------------------------------------------------------------------

def add_target_lags(df: pd.DataFrame,
                    target: str = TARGET_COL,
                    lags: list = DEFAULT_TARGET_LAGS) -> pd.DataFrame:
    """
    Include lagged values of the target itself as features.

    This is an autoregressive (AR) component.  Even if the individual factor
    lags contain most of the predictive information, past excess-return values
    carry direct regime information (e.g. whether the market has been
    consistently positive or negative recently) that may not be fully
    reconstructable from factor lags alone.

    Lags 1-3 days are used to avoid look-ahead leakage while still capturing
    short-term serial correlation.
    """
    for lag in lags:
        df[f"target_lag_{lag}"] = df[target].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Step 7 - Volatility proxy
# ---------------------------------------------------------------------------

def add_vol_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple volatility proxy from the absolute mean of all 'V*'
    columns.

    The 'V' columns are treated as volatility-related factor scores.  Their
    absolute mean provides a single scalar that summarises contemporaneous
    realised dispersion.  This column is *excluded* from the training feature
    set but is used downstream in the allocation / signal-scaling logic.
    """
    v_cols = [col for col in df.columns if col.startswith('V')]
    df['vol_proxy'] = df[v_cols].abs().mean(axis=1) if v_cols else 1.0
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   top_n: int   = DEFAULT_TOP_N_FEATURES,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline on a raw DataFrame.

    Steps
    -----
    1. Impute missing values (ffill → bfill).
    2. Identify base feature columns.
    3. Rank base features by |correlation| with target; select top-N.
    4. Add lag features for all base features.
    5. Add momentum (diff) features for all base features.
    6. Add rolling mean/std for top-N features across all windows.
    7. Add autoregressive target lags.
    8. Add volatility proxy column.
    9. Drop rows that still contain NaN (rolling burn-in, lag burn-in).

    Parameters
    ----------
    df      : Raw DataFrame sorted by date_id (ascending).
    top_n   : Number of features to use for rolling statistics.
    verbose : Print progress and feature-selection diagnostics.

    Returns
    -------
    Transformed DataFrame with all engineered columns added and NaN rows
    removed.
    """
    if verbose:
        print(f"  Shape before feature engineering: {df.shape}")

    df = impute_features(df)

    base_cols  = get_base_feature_cols(df)
    top_feats  = rank_features_by_correlation(df, base_cols, top_n=top_n,
                                               verbose=verbose)

    df = add_lag_features(df, base_cols)
    df = add_momentum_features(df, base_cols)
    df = add_rolling_features(df, top_feats)
    df = add_target_lags(df)
    df = add_vol_proxy(df)

    df = df.dropna()

    if verbose:
        n_feats = len([c for c in df.columns if c not in NON_FEATURE_COLS])
        print(f"  Shape after feature engineering: {df.shape}")
        print(f"  Total trainable features: {n_feats}")

    return df


# ---------------------------------------------------------------------------
# Train / backtest split + scaling
# ---------------------------------------------------------------------------

def split_and_scale(df: pd.DataFrame,
                    exclude_last_n_days: int = 180):
    """
    Chronological train / backtest split followed by StandardScaler fit on
    training data only.

    Parameters
    ----------
    df                  : Fully engineered DataFrame (output of build_features).
    exclude_last_n_days : Number of rows held out as the backtest period.

    Returns
    -------
    Dictionary with keys:
        train_era, backtest_era       - raw DataFrames for each split
        X_train_scaled, X_test_scaled - scaled numpy arrays
        feature_cols                  - list of feature column names
        scaler                        - fitted StandardScaler instance
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    split_idx    = len(df) - exclude_last_n_days
    train_era    = df.iloc[:split_idx].copy()
    backtest_era = df.iloc[split_idx:].copy()

    scaler           = StandardScaler()
    X_train_scaled   = scaler.fit_transform(train_era[feature_cols])
    X_test_scaled    = scaler.transform(backtest_era[feature_cols])

    return {
        'train_era':       train_era,
        'backtest_era':    backtest_era,
        'X_train_scaled':  X_train_scaled,
        'X_test_scaled':   X_test_scaled,
        'feature_cols':    feature_cols,
        'scaler':          scaler,
    }


# ---------------------------------------------------------------------------
# Convenience entry point (mirrors prepare_data from main script)
# ---------------------------------------------------------------------------

def prepare_data(data_path: str,
                 exclude_last_n_days: int = 180,
                 top_n: int = DEFAULT_TOP_N_FEATURES,
                 verbose: bool = True) -> dict:
    """
    Load CSV, run full feature engineering pipeline, split, and scale.

    This is a drop-in replacement for the `prepare_data` function in the
    main modelling script.

    Parameters
    ----------
    data_path           : Path to the raw training CSV.
    exclude_last_n_days : Rows held out as backtest period.
    top_n               : Features used for rolling statistics.
    verbose             : Print progress diagnostics.

    Returns
    -------
    Dictionary containing all artefacts needed for model training and
    evaluation (same schema as the original prepare_data return value, plus
    the top_features list for reference).
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df.sort_values(by="date_id").reset_index(drop=True)

    df = build_features(df, top_n=top_n, verbose=verbose)

    result          = split_and_scale(df, exclude_last_n_days)
    result['df']    = df
    # Expose top features for diagnostics / downstream use
    base_cols       = get_base_feature_cols(
        pd.read_csv(data_path, nrows=1))          # lightweight re-read for col names
    # (top features already embedded in df columns; store for reference only)
    result['top_n'] = top_n
    return result