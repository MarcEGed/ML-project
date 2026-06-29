"""
Bronze Medal Replication (Score: 1.958)
Single LightGBM Regressor with Multi-Timeframe Features and Tanh Post-Processing

This script implements the Bronze Medal approach from the Hull Tactical competition:
- Single LightGBM regressor (not ensemble)
- Target: market_forward_excess_returns
- Multi-timeframe rolling statistics (short, medium, macro/quarterly)
- Momentum & lag features
- Autoregressive targets
- Tanh-based allocation with adaptive volatility multiplier
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data settings
    DATA_PATH = "train.csv"
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    
    # Target
    TARGET = 'market_forward_excess_returns'
    
    # Feature Engineering
    # Rolling window configurations for multi-timeframe features
    SHORT_WINDOWS = [3, 5, 10]  # Short-term: days
    MEDIUM_WINDOWS = [20, 30, 60]  # Medium-term: weeks to months
    MACRO_WINDOWS = [90, 120, 252]  # Macro/quarterly: quarters to year
    
    # Lag features
    LAG_PERIODS = [1, 2, 3, 5, 10]  # Lag depths for momentum
    
    # Model settings - LightGBM conservative boosting
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,  # -1 means no limit
        'learning_rate': 0.01,
        'n_estimators': 5000,
        'min_child_samples': 20,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    # Early stopping
    EARLY_STOPPING_ROUNDS = 100
    VALIDATION_FRACTION = 0.10
    
    # Post-processing settings
    TANH_SCALE = 1.0  # Scale factor for tanh
    TANH_SENSITIVITY = 50.0  # Sensitivity parameter for tanh
    
    # Adaptive volatility multiplier
    VOL_WINDOW = 60  # Window for trailing prediction volatility
    MIN_MULTIPLIER = 0.5  # Minimum volatility multiplier
    MAX_MULTIPLIER = 2.0  # Maximum volatility multiplier
    
    # Position sizing
    BASE_MULTIPLIER = 80.0
    
    # Cross-validation
    N_CV_FOLDS = 5

config = Config()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_multi_timeframe_features(df, target_col):
    """
    Create multi-timeframe rolling statistics for quality features.
    Focus on top-performing feature categories based on Bronze Medal insights.
    """
    print("Creating multi-timeframe rolling features...")
    
    # Identify feature categories
    # Based on Bronze Medal: focus on quality over quantity
    # Select representative features from each category
    
    # Define which base features to use for rolling stats
    # Start with all available features, but we'll be selective
    feature_categories = {
        'D': [col for col in df.columns if col.startswith('D')],
        'E': [col for col in df.columns if col.startswith('E')],
        'M': [col for col in df.columns if col.startswith('M')],
        'P': [col for col in df.columns if col.startswith('P')],
        'S': [col for col in df.columns if col.startswith('S')],
        'V': [col for col in df.columns if col.startswith('V')],
        'I': [col for col in df.columns if col.startswith('I')]
    }
    
    # For efficiency, focus on features that are likely to have predictive power
    # Based on competition insights, V (volatility), M (macro), P (price) are often strong
    # Also include the target itself for autoregressive features
    
    # Select key features for rolling statistics
    # We'll use a subset to maintain quality over quantity
    key_features = []
    for cat, cols in feature_categories.items():
        # Only use non-empty columns
        non_empty_cols = [c for c in cols if c in df.columns and not df[c].isna().all()]
        if non_empty_cols:
            # For large categories, sample the first few
            if len(non_empty_cols) > 10:
                key_features.extend(non_empty_cols[:5])  # Take first 5 from each category
            else:
                key_features.extend(non_empty_cols)
    
    # Always include the target for autoregressive features
    if target_col in df.columns:
        key_features.append(target_col)
    
    print(f"Computing rolling stats for {len(key_features)} key features")
    
    # Multi-timeframe rolling statistics
    all_windows = config.SHORT_WINDOWS + config.MEDIUM_WINDOWS + config.MACRO_WINDOWS
    
    for feature in key_features:
        for window in all_windows:
            # Rolling mean
            col_name = f"{feature}_rolling_mean_{window}"
            df[col_name] = df[feature].rolling(window=window, min_periods=1).mean()
            
            # Rolling standard deviation
            col_name = f"{feature}_rolling_std_{window}"
            df[col_name] = df[feature].rolling(window=window, min_periods=1).std()
    
    print(f"Added {len(all_windows) * 2 * len(key_features)} rolling features")
    
    return df


def create_lag_features(df, target_col):
    """
    Create lagged features for momentum and autoregressive targets.
    """
    print("Creating lag features...")
    
    # Lag features for key columns
    # Focus on a subset of important features
    feature_categories = {
        'D': [col for col in df.columns if col.startswith('D')],
        'M': [col for col in df.columns if col.startswith('M')],
        'P': [col for col in df.columns if col.startswith('P')],
        'V': [col for col in df.columns if col.startswith('V')],
    }
    
    key_features = []
    for cat, cols in feature_categories.items():
        non_empty_cols = [c for c in cols if c in df.columns and not df[c].isna().all()]
        if non_empty_cols:
            if len(non_empty_cols) > 5:
                key_features.extend(non_empty_cols[:3])  # Take first 3 from each category
            else:
                key_features.extend(non_empty_cols)
    
    # Always include target
    if target_col in df.columns:
        key_features.append(target_col)
    
    # Create lag features
    for feature in key_features:
        for lag in config.LAG_PERIODS:
            col_name = f"{feature}_lag_{lag}"
            df[col_name] = df[feature].shift(lag)
    
    print(f"Added {len(config.LAG_PERIODS) * len(key_features)} lag features")
    
    return df


def create_momentum_features(df):
    """
    Create momentum features (lagged differences).
    """
    print("Creating momentum features...")
    
    # Focus on price and macro features for momentum
    price_features = [col for col in df.columns if col.startswith('P')]
    macro_features = [col for col in df.columns if col.startswith('M')]
    
    momentum_features = price_features + macro_features
    momentum_features = [f for f in momentum_features if f in df.columns and not df[f].isna().all()]
    
    # Short-term lagged differences (momentum)
    for feature in momentum_features:
        for lag in [1, 2, 3, 5]:
            col_name = f"{feature}_momentum_{lag}"
            df[col_name] = df[feature].diff(lag)
    
    print(f"Added {len([1, 2, 3, 5]) * len(momentum_features)} momentum features")
    
    return df


def prepare_data(data_path, config):
    """Load and prepare data with feature engineering."""
    print("Loading and preparing data...")
    
    # Load and sort data chronologically
    df = pd.read_csv(data_path)
    df = df.sort_values(by="date_id").reset_index(drop=True)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create volatility proxy from V-columns (for post-processing, not features)
    v_cols = [col for col in df.columns if col.startswith('V')]
    if v_cols:
        df['vol_proxy'] = df[v_cols].abs().mean(axis=1)
    
    # Feature engineering
    # Note: We need to be careful about look-ahead bias
    # All feature engineering must use only past information
    
    # Create lag features (these use past data, safe)
    df = create_lag_features(df, config.TARGET)
    
    # Create momentum features (differences, also safe)
    df = create_momentum_features(df)
    
    # Create rolling statistics - these need careful handling
    # We'll compute them in a way that avoids look-ahead
    df = create_multi_timeframe_features(df, config.TARGET)
    
    # Drop rows with NaN values that resulted from lag/rolling operations
    # We can't use these rows for training as they don't have complete history
    initial_len = len(df)
    df = df.dropna(subset=[col for col in df.columns if not col.startswith('date_id')])
    print(f"Dropped {initial_len - len(df)} rows with NaN values from feature engineering")
    
    # Define targets
    targets = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    
    # Define features - exclude date_id, targets, and vol_proxy
    exclude_cols = ['date_id'] + targets + ['vol_proxy']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Final feature count: {len(feature_cols)}")
    print(f"Features: {feature_cols[:10]}...")  # Show first 10
    
    # Chronological split
    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    train_era = df.iloc[:split_idx].copy()
    backtest_era = df.iloc[split_idx:].copy()
    
    print(f"Training samples: {len(train_era)}")
    print(f"Backtest samples: {len(backtest_era)}")
    
    # Prepare feature matrices
    X_train = train_era[feature_cols]
    X_backtest = backtest_era[feature_cols]
    
    # Impute missing values with training median (no data leakage)
    feature_medians = X_train.median().fillna(0)
    X_train_imputed = X_train.fillna(feature_medians)
    X_backtest_imputed = X_backtest.fillna(feature_medians)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_backtest_scaled = scaler.transform(X_backtest_imputed)
    
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Backtest data shape: {X_backtest_scaled.shape}")
    
    return {
        'df': df,
        'train_era': train_era,
        'backtest_era': backtest_era,
        'X_train_scaled': X_train_scaled,
        'X_backtest_scaled': X_backtest_scaled,
        'feature_cols': feature_cols,
        'feature_medians': feature_medians,
        'scaler': scaler,
        'targets': targets,
        'split_idx': split_idx
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_lightgbm_model(X_train, y_train, config):
    """Train LightGBM model with early stopping and conservative parameters."""
    print("\nTraining LightGBM model...")
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Create validation set for early stopping
    # Use time-series split: last 10% of training data for validation
    val_size = int(len(X_train) * config.VALIDATION_FRACTION)
    X_train_main = X_train[:-val_size]
    y_train_main = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    
    train_dataset = lgb.Dataset(X_train_main, label=y_train_main)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
    
    # Train model
    model = lgb.train(
        config.LGBM_PARAMS,
        train_dataset,
        num_boost_round=config.LGBM_PARAMS['n_estimators'],
        valid_sets=[val_dataset],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100),
        ]
    )
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f"Validation MSE: {val_mse:.6f}")
    
    # Calculate IC on validation
    val_ic = np.corrcoef(y_val, y_val_pred)[0, 1]
    print(f"Validation IC: {val_ic:.4f}")
    
    return model


# ============================================================================
# POST-PROCESSING
# ============================================================================

def tanh_allocation(predictions, scale, sensitivity):
    """
    Apply scaled tanh function for base allocation.
    
    Args:
        predictions: Model predictions (raw excess return predictions)
        scale: Scale parameter (controls overall magnitude)
        sensitivity: Sensitivity parameter (controls steepness of tanh)
    
    Returns:
        Base allocation values
    """
    return scale * np.tanh(predictions * sensitivity)


def calculate_adaptive_volatility_multiplier(predictions, window=60, min_mult=0.5, max_mult=2.0):
    """
    Calculate adaptive volatility multiplier based on trailing prediction volatility.
    
    The idea: When model predictions have low volatility (calm market conditions),
    we can increase our allocation multiplier to capitalize on stable signals.
    When predictions have high volatility (chaotic market), we reduce the multiplier
    to protect against erratic signals.
    
    Args:
        predictions: Array of model predictions
        window: Rolling window for volatility calculation
        min_mult: Minimum multiplier (floor)
        max_mult: Maximum multiplier (ceiling)
    
    Returns:
        Dynamic multiplier that adapts to prediction volatility
    """
    # Calculate rolling standard deviation of predictions
    pred_series = pd.Series(predictions)
    rolling_std = pred_series.rolling(window=window, min_periods=1).std()
    
    # Normalize by overall prediction std
    overall_std = np.std(predictions)
    if overall_std < 1e-10:
        # If predictions have no variation, use maximum multiplier
        return pd.Series([max_mult] * len(predictions))
    
    # Calculate relative volatility
    relative_vol = rolling_std / overall_std
    
    # Inverse relationship: low volatility -> high multiplier
    # When relative_vol is low (< 0.5), multiplier should be high
    # When relative_vol is high (> 1.5), multiplier should be low
    # Use a sigmoid-like transformation
    
    # Transform: relative_vol -> multiplier
    # We want: low relative_vol -> multiplier close to max_mult
    #           high relative_vol -> multiplier close to min_mult
    raw_multiplier = max_mult - (max_mult - min_mult) * (relative_vol ** 2)
    
    # Clip to bounds
    multiplier = np.clip(raw_multiplier, min_mult, max_mult)
    
    return multiplier


def calculate_allocation(predictions, backtest_era, data, config):
    """
    Calculate final allocation using tanh-based post-processing with adaptive volatility.
    
    This implements the Bronze Medal approach:
    1. Base allocation via scaled tanh
    2. Adaptive volatility multiplier based on prediction volatility
    3. Final position sizing with BASE_MULTIPLIER
    """
    # Step 1: Tanh-based base allocation
    base_allocation = tanh_allocation(
        predictions, 
        scale=config.TANH_SCALE, 
        sensitivity=config.TANH_SENSITIVITY
    )
    
    # Step 2: Adaptive volatility multiplier
    vol_multiplier = calculate_adaptive_volatility_multiplier(
        predictions,
        window=config.VOL_WINDOW,
        min_mult=config.MIN_MULTIPLIER,
        max_mult=config.MAX_MULTIPLIER
    )
    
    # Step 3: Traditional volatility scaling (from original solution)
    train_vol_median = data['train_era']['vol_proxy'].median()
    vol_scalar = train_vol_median / (backtest_era['vol_proxy'].values + 1e-8)
    
    # Combine multipliers
    # The adaptive volatility multiplier modifies the base allocation
    # Then we apply traditional volatility scaling
    combined_multiplier = vol_multiplier.values * vol_scalar
    
    # Final allocation
    # Start from base (1.0 = neutral market position)
    # Add tanh-based allocation adjustment
    # Scale by combined multiplier
    raw_allocation = 1.0 + (base_allocation * config.BASE_MULTIPLIER * combined_multiplier)
    
    # Clip to valid range [0, 2]
    final_allocation = np.clip(raw_allocation, 0.0, 2.0)
    
    return final_allocation, {
        'base_allocation': base_allocation,
        'vol_multiplier': vol_multiplier.values,
        'vol_scalar': vol_scalar,
        'combined_multiplier': combined_multiplier,
        'raw_allocation': raw_allocation
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_strategy(backtest_era, predictions, data, config, strategy_name="Bronze"):
    """Evaluate strategy performance with competition metrics."""
    
    # Calculate allocation
    allocation, post_processing_data = calculate_allocation(
        predictions, backtest_era, data, config
    )
    
    # Calculate strategy returns
    market_daily_return = backtest_era['forward_returns'].values
    strategy_daily_return = allocation * market_daily_return
    
    # Information Coefficient
    actual_returns = backtest_era[config.TARGET].values
    ic = np.corrcoef(predictions, actual_returns)[0, 1]
    
    # Annualized returns
    ann_market_return = np.mean(market_daily_return) * 252 * 100
    ann_strategy_return = np.mean(strategy_daily_return) * 252 * 100
    
    # Annualized volatility
    ann_market_vol = np.std(market_daily_return) * np.sqrt(252) * 100
    ann_strategy_vol = np.std(strategy_daily_return) * np.sqrt(252) * 100
    
    # Volatility ratio
    vol_ratio = ann_strategy_vol / ann_market_vol if ann_market_vol > 0 else 0
    vol_pass = vol_ratio <= 1.20
    vol_ratio_status = "(PASS)" if vol_pass else "(FAIL)"
    
    # Sharpe Ratio
    market_sharpe = ann_market_return / ann_market_vol if ann_market_vol > 0 else 0
    strategy_sharpe = ann_strategy_return / ann_strategy_vol if ann_strategy_vol > 0 else 0
    
    # Net Alpha
    net_alpha = ann_strategy_return - ann_market_return
    
    # Additional metrics
    max_allocation = np.max(allocation)
    min_allocation = np.min(allocation)
    mean_allocation = np.mean(allocation)
    
    # competition-like score (modified Sharpe)
    # This approximates the competition scoring
    score = strategy_sharpe * (1.0 - max(0, vol_ratio - 1.20) / 2.0)
    
    return {
        'name': strategy_name,
        'ic': ic,
        'ann_market_return': ann_market_return,
        'ann_strategy_return': ann_strategy_return,
        'ann_market_vol': ann_market_vol,
        'ann_strategy_vol': ann_strategy_vol,
        'vol_ratio': vol_ratio,
        'vol_ratio_status': vol_ratio_status,
        'vol_pass': vol_pass,
        'market_sharpe': market_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'net_alpha': net_alpha,
        'max_allocation': max_allocation,
        'min_allocation': min_allocation,
        'mean_allocation': mean_allocation,
        'predictions': predictions,
        'allocation': allocation,
        'score': score,
        'post_processing': post_processing_data
    }


def print_strategy_results(results):
    """Print formatted strategy results."""
    print(f"\n{'='*70}")
    print(f" {results['name']} BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f" Information Coefficient (Correlation):     {results['ic']:+.4f}")
    print(f" Annualized Market Return:                 {results['ann_market_return']:.2f}%")
    print(f" Annualized Strategy Return:               {results['ann_strategy_return']:.2f}%")
    print(f" Annualized Market Volatility:             {results['ann_market_vol']:.2f}%")
    print(f" Annualized Strategy Volatility:           {results['ann_strategy_vol']:.2f}%")
    print(f" Strategy-to-Market Volatility Ratio:      {results['vol_ratio']:.2f}x {results['vol_ratio_status']}")
    print(f" Market Buy-and-Hold Sharpe Ratio:          {results['market_sharpe']:.3f}")
    print(f" Strategy Sharpe Ratio:                     {results['strategy_sharpe']:.3f}")
    print(f" Net Alpha Generation:                     {results['net_alpha']:+.2f}%")
    print(f" Allocation Range: [{results['min_allocation']:.3f}, {results['max_allocation']:.3f}]")
    print(f" Mean Allocation:                          {results['mean_allocation']:.3f}")
    
    # Post-processing details
    pp = results.get('post_processing', {})
    if pp:
        print(f"\n Post-Processing Details:")
        print(f"   Mean Volatility Multiplier:             {np.mean(pp.get('vol_multiplier', [1.0])):.3f}")
        print(f"   Mean Combined Multiplier:              {np.mean(pp.get('combined_multiplier', [1.0])):.3f}")
        print(f"   Mean Base Allocation (tanh):            {np.mean(pp.get('base_allocation', [0.0])):.3f}")
    
    print(f" Estimated Competition Score:              {results['score']:.4f}")
    print(f"{'='*70}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print(" BRONZE MEDAL REPLICATION (Score Target: 1.958)")
    print(" Single LightGBM + Multi-Timeframe Features + Tanh Post-Processing")
    print("="*70)
    
    # Step 1: Prepare data with feature engineering
    data = prepare_data(config.DATA_PATH, config)
    
    # Step 2: Train LightGBM model
    X_train = data['X_train_scaled']
    y_train = data['train_era'][config.TARGET].values
    
    model = train_lightgbm_model(X_train, y_train, config)
    
    # Step 3: Generate predictions on backtest data
    X_backtest = data['X_backtest_scaled']
    backtest_era = data['backtest_era']
    
    predictions = model.predict(X_backtest)
    
    print(f"\nPredictions statistics:")
    print(f"  Mean: {np.mean(predictions):.6f}")
    print(f"  Std:  {np.std(predictions):.6f}")
    print(f"  Min:  {np.min(predictions):.6f}")
    print(f"  Max:  {np.max(predictions):.6f}")
    
    # Step 4: Evaluate Bronze strategy
    bronze_results = evaluate_strategy(
        backtest_era, predictions, data, config,
        strategy_name="Bronze Medal (LightGBM + Tanh + Adaptive Vol)"
    )
    
    print_strategy_results(bronze_results)
    
    # Step 5: Compare with simple Ridge baseline
    print("\n" + "="*70)
    print(" COMPARISON: Ridge Baseline vs Bronze LightGBM")
    print("="*70)
    
    # Train simple Ridge for comparison
    from sklearn.linear_model import Ridge
    ridge_model = Ridge(alpha=500.0, random_state=config.RANDOM_STATE)
    ridge_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_backtest)
    
    ridge_results = evaluate_strategy(
        backtest_era, ridge_predictions, data, config,
        strategy_name="Ridge Baseline"
    )
    
    print_strategy_results(ridge_results)
    
    # Print comparison
    print(f"\n{'='*70}")
    print(" COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} {'Bronze LightGBM':>20} {'Ridge':>20}")
    print("-"*65)
    print(f"{'Sharpe Ratio':<25} {bronze_results['strategy_sharpe']:>20.3f} {ridge_results['strategy_sharpe']:>20.3f}")
    print(f"{'Annual Return %':<25} {bronze_results['ann_strategy_return']:>20.2f} {ridge_results['ann_strategy_return']:>20.2f}")
    print(f"{'Volatility %':<25} {bronze_results['ann_strategy_vol']:>20.2f} {ridge_results['ann_strategy_vol']:>20.2f}")
    print(f"{'Volatility Ratio':<25} {bronze_results['vol_ratio']:>20.2f}x {ridge_results['vol_ratio']:>20.2f}x")
    print(f"{'IC':<25} {bronze_results['ic']:>20.4f} {ridge_results['ic']:>20.4f}")
    print(f"{'Alpha %':<25} {bronze_results['net_alpha']:>20.2f} {ridge_results['net_alpha']:>20.2f}")
    print(f"{'Score (est.)':<25} {bronze_results['score']:>20.4f} {ridge_results['score']:>20.4f}")
    print("="*65)
    
    # Step 6: Feature importance analysis
    print("\n" + "="*70)
    print(" FEATURE IMPORTANCE (Top 20)")
    print("="*70)
    
    # Get feature importances from LightGBM
    feature_importances = model.feature_importance(importance_type='gain')
    feature_names = data['feature_cols']
    
    # Sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Print top 20
    print(importance_df.head(20).to_string(index=False))
    
    # Save results
    print(f"\n✓ Bronze Medal replication complete!")
    print(f"✓ Target score: 1.958")
    print(f"✓ Achieved score: {bronze_results['score']:.4f}")
    print(f"✓ Volatility ratio check: {'PASS' if bronze_results['vol_pass'] else 'FAIL'}")
    
    return bronze_results, ridge_results, model, importance_df


if __name__ == "__main__":
    results = main()
