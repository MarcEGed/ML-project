"""
Phase 0: Volatility Constrained - IMPROVED
Replicate 163rd place core implementation
Target: Achieve 2.1+ validation score

Key improvements:
- Use top 30 features by correlation for rolling stats (instead of first 25)
- Ridge with alpha=100 (instead of 500)
- Same volatility multiplier and signal conversion (163rd place)

Feature engineering is handled by feature_engineering.py.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import prepare_data


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    DATA_PATH           = "train.csv"
    RANDOM_STATE        = 42
    EXCLUDE_LAST_N_DAYS = 180
    TARGET              = 'market_forward_excess_returns'

    # Number of top features to use for rolling stats (see feature_engineering.py)
    TOP_N_FEATURES      = 30

    LGBM_PARAMS = {
        'objective':        'regression',
        'metric':           'rmse',
        'boosting_type':    'gbdt',
        'num_leaves':       31,
        'max_depth':        -1,
        'learning_rate':    0.01,
        'n_estimators':     5000,
        'min_child_samples': 20,
        'reg_alpha':        0.0,
        'reg_lambda':       0.0,
        'random_state':     42,
        'verbose':          -1,
        'n_jobs':           -1,
    }

    EARLY_STOPPING_ROUNDS = 100
    VALIDATION_FRACTION   = 0.10

    # Signal conversion parameters (163rd place - DO NOT MODIFY)
    TANH_BOUND  = 0.006
    TANH_SCALE  = 3.0
    VOL_WINDOW  = 20

    # Volatility constraint
    MAX_VOL_RATIO = 1.20

    # Ridge parameters
    RIDGE_ALPHA   = 100.0


config = Config()


# ---------------------------------------------------------------------------
# Signal conversion & allocation  (163rd place proven implementation)
# ---------------------------------------------------------------------------

def convert_ret_to_signal(x, bound=0.006, scale=3.0):
    """place proven implementation."""
    x_clipped = np.clip(x, -bound, bound)
    x_norm    = x_clipped / bound
    x_scaled  = x_norm * scale
    return np.tanh(x_scaled) + 1.0


def calculate_volatility_multiplier(predictions, window=20):
    """
    Adaptive volatility multiplier.
    multiplier = overall_std / rolling_std
    Boosts allocation when prediction vol is low; penalises when high.
    """
    pred_series  = pd.Series(predictions)
    rolling_std  = pred_series.rolling(window=window, min_periods=1).std()
    rolling_std  = rolling_std.bfill().fillna(1.0)
    overall_std  = np.std(predictions)

    if overall_std < 1e-10:
        return pd.Series([1.0] * len(predictions))

    return overall_std / (rolling_std + 1e-10)


def calculate_allocation_vol_constrained(predictions, config):
    """
    Scale predictions by the volatility multiplier, convert to a [0, 1]
    signal via tanh, then clip to [0, MAX_VOL_RATIO].
    """
    vol_multiplier    = calculate_volatility_multiplier(predictions,
                                                        window=config.VOL_WINDOW)
    scaled_predictions = predictions * vol_multiplier.values
    signals            = convert_ret_to_signal(scaled_predictions,
                                               config.TANH_BOUND,
                                               config.TANH_SCALE)
    return np.clip(signals, 0.0, config.MAX_VOL_RATIO)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgbm(X_train, y_train, config):
    print("\nTraining LightGBM...")
    val_size = int(len(X_train) * config.VALIDATION_FRACTION)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    train_ds = lgb.Dataset(X_tr, label=y_tr)
    val_ds   = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    model = lgb.train(
        config.LGBM_PARAMS, train_ds,
        num_boost_round=config.LGBM_PARAMS['n_estimators'],
        valid_sets=[val_ds], valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100),
        ]
    )

    yp = model.predict(X_val)
    print(f"Val MSE: {mean_squared_error(y_val, yp):.6f}, "
          f"IC: {np.corrcoef(y_val, yp)[0,1]:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(backtest_era, predictions, config, name="Strat"):
    allocation = calculate_allocation_vol_constrained(predictions, config)

    mkt_ret   = backtest_era['forward_returns'].values
    strat_ret = allocation * mkt_ret
    actual    = backtest_era[config.TARGET].values

    ic        = np.corrcoef(predictions, actual)[0, 1]

    ann_mkt   = np.mean(mkt_ret)   * 252 * 100
    ann_strat = np.mean(strat_ret) * 252 * 100
    vol_mkt   = np.std(mkt_ret)    * np.sqrt(252) * 100
    vol_strat = np.std(strat_ret)  * np.sqrt(252) * 100

    vol_ratio = vol_strat / vol_mkt if vol_mkt > 0 else 0
    vol_pass  = vol_ratio <= config.MAX_VOL_RATIO

    mkt_sharpe  = ann_mkt   / vol_mkt   if vol_mkt   > 0 else 0
    strat_sharpe = ann_strat / vol_strat if vol_strat > 0 else 0
    alpha       = ann_strat - ann_mkt

    score = strat_sharpe * (1.0 - max(0, vol_ratio - config.MAX_VOL_RATIO) / 2.0)

    return {
        'name': name, 'ic': ic,
        'ann_mkt': ann_mkt, 'ann_strat': ann_strat,
        'vol_mkt': vol_mkt, 'vol_strat': vol_strat,
        'vol_ratio': vol_ratio, 'vol_pass': vol_pass,
        'mkt_sharpe': mkt_sharpe, 'strat_sharpe': strat_sharpe,
        'alpha': alpha, 'score': score, 'allocation': allocation,
    }


def print_results(r):
    print(f"\n{r['name']}")
    print(f"  IC: {r['ic']:+.4f}, Score: {r['score']:.4f}")
    print(f"  Mkt:   {r['ann_mkt']:.2f}%  vol={r['vol_mkt']:.2f}%  "
          f"Sharpe={r['mkt_sharpe']:.3f}")
    print(f"  Strat: {r['ann_strat']:.2f}%  vol={r['vol_strat']:.2f}%  "
          f"Sharpe={r['strat_sharpe']:.3f}")
    print(f"  Vol ratio: {r['vol_ratio']:.2f}x "
          f"{'PASS' if r['vol_pass'] else 'FAIL'}")
    print(f"  Alpha: {r['alpha']:+.2f}%")
    print(f"  Alloc: [{np.min(r['allocation']):.3f}, "
          f"{np.max(r['allocation']):.3f}]  "
          f"Mean: {np.mean(r['allocation']):.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("="*70)
    print("PHASE 0: VOLATILITY CONSTRAINED - IMPROVED")
    print("163rd place core implementation")
    print(f"Top {config.TOP_N_FEATURES} features for rolling stats, "
          f"Ridge alpha={config.RIDGE_ALPHA}")
    print(f"Allocation clipped to [0, {config.MAX_VOL_RATIO}]")
    print("Target Score: 2.16")
    print("="*70)

    # Feature engineering is fully delegated to feature_engineering.py
    data = prepare_data(
        data_path=config.DATA_PATH,
        exclude_last_n_days=config.EXCLUDE_LAST_N_DAYS,
        top_n=config.TOP_N_FEATURES,
    )

    X_train      = data['X_train_scaled']
    y_train      = data['train_era'][config.TARGET].values
    X_test       = data['X_test_scaled']
    backtest_era = data['backtest_era']

    print(f"\nTraining data:  {X_train.shape[0]} samples, "
          f"{X_train.shape[1]} features")
    print(f"Backtest data:  {X_test.shape[0]} samples")

    # --- LightGBM ---
    lgbm_model  = train_lgbm(X_train, y_train, config)
    lgbm_pred   = lgbm_model.predict(X_test)
    lgbm_result = evaluate(backtest_era, lgbm_pred, config, "Phase 0 (LightGBM)")
    print_results(lgbm_result)

    # --- Ridge (key improvement over 163rd place baseline) ---
    ridge       = Ridge(alpha=config.RIDGE_ALPHA,
                        random_state=config.RANDOM_STATE)
    ridge.fit(X_train, y_train)
    ridge_pred   = ridge.predict(X_test)
    ridge_result = evaluate(backtest_era, ridge_pred, config, "Ridge (alpha=100)")
    print_results(ridge_result)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"Phase 0 Score (LightGBM): {lgbm_result['score']:.4f}  (Target: 2.16)")
    print(f"Ridge Score:              {ridge_result['score']:.4f}  (Target: 2.16)")
    print(f"Vol check (LightGBM):     {'PASS' if lgbm_result['vol_pass'] else 'FAIL'}")
    print(f"Vol check (Ridge):        {'PASS' if ridge_result['vol_pass'] else 'FAIL'}")
    print("="*70)

    if ridge_result['score'] >= 2.16:
        print("\nSUCCESS: Ridge achieves target score of 2.16!")
    else:
        print(f"\nGap to target: {2.16 - ridge_result['score']:.4f}")


if __name__ == "__main__":
    main()