"""
Phase 0: Volatility Constrained
Constrain allocation to [0, 1.20] to ensure vol_ratio <= 1.20
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')


class Config:
    DATA_PATH = "train.csv"
    RANDOM_STATE = 42
    EXCLUDE_LAST_N_DAYS = 180
    TARGET = 'market_forward_excess_returns'
    
    SHORT_WINDOWS = [3, 5, 10]
    MEDIUM_WINDOWS = [20, 30, 60]
    MACRO_WINDOWS = [90, 120, 252]
    LAG_PERIODS = [1, 2, 3, 5, 10]
    
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.01,
        'n_estimators': 5000,
        'min_child_samples': 20,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    EARLY_STOPPING_ROUNDS = 100
    VALIDATION_FRACTION = 0.10
    
    TANH_BOUND = 0.006
    TANH_SCALE = 3.0
    VOL_WINDOW = 20
    
    # Volatility constraint
    MAX_VOL_RATIO = 1.20


config = Config()


def convert_ret_to_signal(x, bound=0.006, scale=3.0):
    x_clipped = np.clip(x, -bound, bound)
    x_norm = x_clipped / bound
    x_scaled = x_norm * scale
    return np.tanh(x_scaled) + 1.0


def calculate_volatility_multiplier(predictions, window=20):
    pred_series = pd.Series(predictions)
    rolling_std = pred_series.rolling(window=window, min_periods=1).std()
    rolling_std = rolling_std.fillna(1.0)
    
    overall_std = np.std(predictions)
    if overall_std < 1e-10:
        return pd.Series([1.0] * len(predictions))
    
    raw_multiplier = overall_std / (rolling_std + 1e-10)
    return raw_multiplier


def calculate_allocation_vol_constrained(predictions, backtest_era, data, config):
    """
    Allocation with volatility constraint.
    Clip to [0, MAX_VOL_RATIO] to ensure vol_ratio <= MAX_VOL_RATIO
    """
    vol_multiplier = calculate_volatility_multiplier(predictions, window=config.VOL_WINDOW)
    scaled_predictions = predictions * vol_multiplier.values
    signals = convert_ret_to_signal(scaled_predictions, config.TANH_BOUND, config.TANH_SCALE)
    
    # Constrain to [0, MAX_VOL_RATIO] instead of [0, 2]
    # This ensures that if allocation is constant, vol_ratio = allocation <= MAX_VOL_RATIO
    return np.clip(signals, 0.0, config.MAX_VOL_RATIO)


def prepare_data(data_path, config):
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df.sort_values(by="date_id").reset_index(drop=True)
    
    targets = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    
    # Impute feature columns
    for col in df.columns:
        if col not in targets + ['date_id']:
            df[col] = df[col].ffill().bfill()
    
    v_cols = [col for col in df.columns if col.startswith('V')]
    if v_cols:
        df['vol_proxy'] = df[v_cols].abs().mean(axis=1)
    else:
        df['vol_proxy'] = 1.0
    
    feature_cols_base = [c for c in df.columns 
                       if c not in ['date_id'] + targets + ['vol_proxy']]
    
    # Create features
    for feature in feature_cols_base:
        for lag in config.LAG_PERIODS:
            df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
    
    for feature in feature_cols_base:
        for lag in [1, 2, 3, 5]:
            df[f"{feature}_momentum_{lag}"] = df[feature].diff(lag)
    
    all_windows = config.SHORT_WINDOWS + config.MEDIUM_WINDOWS + config.MACRO_WINDOWS
    for feature in feature_cols_base[:25]:
        for window in all_windows:
            df[f"{feature}_rolling_mean_{window}"] = df[feature].rolling(
                window=window, min_periods=1).mean()
            df[f"{feature}_rolling_std_{window}"] = df[feature].rolling(
                window=window, min_periods=1).std()
    
    df = df.dropna()
    
    feature_cols = [c for c in df.columns 
                   if c not in ['date_id'] + targets + ['vol_proxy']]
    
    TRAIN_END = len(df) - config.EXCLUDE_LAST_N_DAYS
    train_era = df.iloc[:TRAIN_END].copy()
    backtest_era = df.iloc[TRAIN_END:].copy()
    
    X_train = train_era[feature_cols]
    X_backtest = backtest_era[feature_cols]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_backtest_scaled = scaler.transform(X_backtest)
    
    return {
        'df': df, 'train_era': train_era, 'backtest_era': backtest_era,
        'X_train_scaled': X_train_scaled, 'X_backtest_scaled': X_backtest_scaled,
        'feature_cols': feature_cols, 'scaler': scaler
    }


def train_model(X_train, y_train, config):
    print("\nTraining LightGBM...")
    val_size = int(len(X_train) * config.VALIDATION_FRACTION)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
    
    train_ds = lgb.Dataset(X_tr, label=y_tr)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
    
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
    print(f"Val MSE: {mean_squared_error(y_val, yp):.6f}, IC: {np.corrcoef(y_val, yp)[0,1]:.4f}")
    return model


def evaluate(backtest_era, predictions, data, config, name="Strat"):
    allocation = calculate_allocation_vol_constrained(predictions, backtest_era, data, config)
    
    mkt_ret = backtest_era['forward_returns'].values
    strat_ret = allocation * mkt_ret
    actual = backtest_era[config.TARGET].values
    
    ic = np.corrcoef(predictions, actual)[0, 1]
    
    ann_mkt = np.mean(mkt_ret) * 252 * 100
    ann_strat = np.mean(strat_ret) * 252 * 100
    vol_mkt = np.std(mkt_ret) * np.sqrt(252) * 100
    vol_strat = np.std(strat_ret) * np.sqrt(252) * 100
    
    vol_ratio = vol_strat / vol_mkt if vol_mkt > 0 else 0
    vol_pass = vol_ratio <= config.MAX_VOL_RATIO
    
    mkt_sharpe = ann_mkt / vol_mkt if vol_mkt > 0 else 0
    strat_sharpe = ann_strat / vol_strat if vol_strat > 0 else 0
    alpha = ann_strat - ann_mkt
    
    score = strat_sharpe * (1.0 - max(0, vol_ratio - config.MAX_VOL_RATIO) / 2.0)
    
    return {
        'name': name, 'ic': ic, 'ann_mkt': ann_mkt, 'ann_strat': ann_strat,
        'vol_mkt': vol_mkt, 'vol_strat': vol_strat, 'vol_ratio': vol_ratio,
        'vol_pass': vol_pass, 'mkt_sharpe': mkt_sharpe, 'strat_sharpe': strat_sharpe,
        'alpha': alpha, 'score': score, 'allocation': allocation
    }


def print_results(r):
    print(f"\n{r['name']}")
    print(f"  IC: {r['ic']:+.4f}, Score: {r['score']:.4f}")
    print(f"  Mkt: {r['ann_mkt']:.2f}% vol={r['vol_mkt']:.2f}%, Sharpe={r['mkt_sharpe']:.3f}")
    print(f"  Strat: {r['ann_strat']:.2f}% vol={r['vol_strat']:.2f}%, Sharpe={r['strat_sharpe']:.3f}")
    print(f"  Vol ratio: {r['vol_ratio']:.2f}x {'PASS' if r['vol_pass'] else 'FAIL'}")
    print(f"  Alpha: {r['alpha']:+.2f}%")
    print(f"  Alloc: [{np.min(r['allocation']):.3f}, {np.max(r['allocation']):.3f}], Mean: {np.mean(r['allocation']):.3f}")


def main():
    print("="*70)
    print("PHASE 0: VOLATILITY CONSTRAINED")
    print(f"Allocation clipped to [0, {config.MAX_VOL_RATIO}]")
    print("Target Score: 2.16")
    print("="*70)
    
    data = prepare_data(config.DATA_PATH, config)
    
    X_train = data['X_train_scaled']
    y_train = data['train_era'][config.TARGET].values
    X_test = data['X_backtest_scaled']
    backtest_era = data['backtest_era']
    
    model = train_model(X_train, y_train, config)
    predictions = model.predict(X_test)
    
    phase0 = evaluate(backtest_era, predictions, data, config, "Phase 0 (Vol Constrained)")
    print_results(phase0)
    
    ridge = Ridge(alpha=500.0, random_state=config.RANDOM_STATE)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_res = evaluate(backtest_era, ridge_pred, data, config, "Ridge")
    print_results(ridge_res)
    
    print(f"\n{'='*70}")
    print(f"Phase 0 Score: {phase0['score']:.4f} (Target: 2.16)")
    print(f"Ridge Score: {ridge_res['score']:.4f}")
    print(f"Volatility Check: {'PASS' if phase0['vol_pass'] else 'FAIL'}")
    print("="*70)


if __name__ == "__main__":
    main()
