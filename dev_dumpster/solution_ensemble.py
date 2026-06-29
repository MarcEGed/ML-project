"""
Enhanced Hull Tactical Market Prediction Solution
Multi-Model Comparison & Ensemble Approach

This script:
1. Compares multiple ML models for predicting market excess returns
2. Implements ensemble methods (weighted averaging, stacking)
3. Evaluates each model and ensemble using the competition metrics
4. Maintains chronological validation and risk-managed allocation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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
    
    # Model settings
    MODELS_TO_TEST = [
        'Ridge',
        'Lasso', 
        'ElasticNet',
        'BayesianRidge',
        'RandomForest',
        'GradientBoosting',
        # 'SVR',  # Commented out as it can be slow
    ]
    
    # Ensemble settings
    ENSEMBLE_METHODS = ['weighted_average', 'stacking', 'voting']
    
    # Prediction settings
    TARGET = 'market_forward_excess_returns'  # Primary target
    PREDICT_ALL_TARGETS = False  # Set to True to predict all targets
    
    # Position sizing
    BASE_MULTIPLIER = 80.0
    
    # Cross-validation
    N_CV_FOLDS = 5

config = Config()

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(data_path):
    """Load and prepare data chronologically."""
    print("Loading and preparing historical data...")
    
    # Load and sort data
    df = pd.read_csv(data_path)
    df = df.sort_values(by="date_id").reset_index(drop=True)
    
    # Create volatility proxy from V-columns
    v_cols = [col for col in df.columns if col.startswith('V')]
    df['vol_proxy'] = df[v_cols].abs().mean(axis=1)
    
    # Define all possible targets
    targets = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    
    # Define features (exclude date_id, targets, and vol_proxy)
    exclude_cols = ['date_id'] + targets + ['vol_proxy']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Chronological split
    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    train_era = df.iloc[:split_idx].copy()
    backtest_era = df.iloc[split_idx:].copy()
    
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
    
    print(f"Training data: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
    print(f"Backtest data: {X_backtest_scaled.shape[0]} samples")
    
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
# MODEL DEFINITIONS
# ============================================================================

def get_models():
    """Initialize all models with optimized hyperparameters."""
    models = {}
    
    # Linear Models
    models['Ridge'] = Ridge(alpha=500.0, random_state=config.RANDOM_STATE)
    models['Lasso'] = Lasso(alpha=0.001, random_state=config.RANDOM_STATE, max_iter=10000)
    models['ElasticNet'] = ElasticNet(
        alpha=0.001, l1_ratio=0.5, 
        random_state=config.RANDOM_STATE, max_iter=10000
    )
    models['BayesianRidge'] = BayesianRidge()
    models['LinearRegression'] = LinearRegression()
    
    # Tree-based Models
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=200, max_depth=10, 
        min_samples_leaf=5, random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    models['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, 
        max_depth=5, min_samples_leaf=5,
        random_state=config.RANDOM_STATE
    )
    
    # SVM (can be slow on large datasets)
    models['SVR'] = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    
    return models

def get_ensemble_models(base_models, X_train, y_train):
    """Create ensemble models from base models."""
    ensembles = {}
    
    # Weighted Average Ensemble (weights based on validation performance)
    # We'll compute weights after training
    
    # Stacking Ensemble
    estimators = [
        ('ridge', base_models['Ridge']),
        ('lasso', base_models['Lasso']),
        ('gb', base_models['GradientBoosting'])
    ]
    
    # Use Ridge as final estimator for stacking
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=100.0),
        cv=3,
        n_jobs=-1
    )
    stacking_model.fit(X_train, y_train)
    ensembles['stacking'] = stacking_model
    
    # Voting Ensemble (for regressors, this is averaging)
    voting_models = [
        ('ridge', base_models['Ridge']),
        ('lasso', base_models['Lasso']),
        ('elasticnet', base_models['ElasticNet']),
        ('bayesian', base_models['BayesianRidge'])
    ]
    voting_model = VotingRegressor(estimators=voting_models)
    voting_model.fit(X_train, y_train)
    ensembles['voting'] = voting_model
    
    return ensembles

# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate_models(data, models):
    """Train all models and evaluate on validation set."""
    train_era = data['train_era']
    X_train = data['X_train_scaled']
    y_train = train_era[config.TARGET].values
    
    results = {}
    
    print("\n" + "="*60)
    print(" TRAINING AND EVALUATING INDIVIDUAL MODELS")
    print("="*60)
    
    # Train each model
    for name in config.MODELS_TO_TEST:
        if name not in models:
            continue
            
        print(f"\nTraining {name}...")
        try:
            model = models[name]
            model.fit(X_train, y_train)
            
            # Cross-validation with time series split
            tscv = TimeSeriesSplit(n_splits=min(config.N_CV_FOLDS, 5))
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                
                # Use negative MSE as score (higher is better)
                score = -mean_squared_error(y_val, y_pred)
                cv_scores.append(score)
            
            # Refit on full training data
            model.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_mean_score': np.mean(cv_scores),
                'cv_std_score': np.std(cv_scores),
                'cv_scores': cv_scores
            }
            
            print(f"  CV Score (neg MSE): {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
            results[name] = None
    
    # Compute weights based on CV performance
    valid_models = {k: v for k, v in results.items() if v is not None}
    if valid_models:
        # Weight by inverse of MSE (higher score = more weight)
        scores = {k: v['cv_mean_score'] for k, v in valid_models.items()}
        total_score = sum(scores.values())
        weights = {k: (s / total_score) if total_score > 0 else 1.0/len(scores) 
                   for k, s in scores.items()}
        results['weights'] = weights
        print(f"\nModel weights for ensemble:")
        for k, w in weights.items():
            print(f"  {k}: {w:.4f}")
    
    return results

def weighted_average_predict(models_results, X, weights=None):
    """Create weighted average predictions from multiple models."""
    if weights is None:
        weights = models_results.get('weights', {})
    
    predictions = []
    model_weights = []
    
    for name, result in models_results.items():
        if name in ['weights', 'weighted_average', 'stacking', 'voting']:
            continue
        if result is None or 'model' not in result:
            continue
        
        weight = weights.get(name, 1.0/len(models_results))
        pred = result['model'].predict(X)
        predictions.append(pred)
        model_weights.append(weight)
    
    if not predictions:
        return np.zeros(len(X))
    
    # Normalize weights to sum to 1
    total_weight = sum(model_weights)
    if total_weight > 0:
        model_weights = [w/total_weight for w in model_weights]
    else:
        model_weights = [1.0/len(model_weights)] * len(model_weights)
    
    # Weighted average
    final_pred = np.zeros(len(X))
    for pred, weight in zip(predictions, model_weights):
        final_pred += pred * weight
    
    return final_pred

# ============================================================================
# BACKTEST EVALUATION
# ============================================================================

def calculate_allocation(predictions, backtest_era, data, multiplier=None):
    """Calculate allocation based on predictions and volatility scaling."""
    if multiplier is None:
        multiplier = config.BASE_MULTIPLIER
    
    # Calculate volatility scalar
    train_vol_median = data['train_era']['vol_proxy'].median()
    vol_scalar = train_vol_median / (backtest_era['vol_proxy'] + 1e-8)
    
    # Position sizing
    dynamic_multiplier = multiplier * vol_scalar
    allocation = 1.0 + (predictions * dynamic_multiplier)
    allocation = np.clip(allocation, 0.0, 2.0)
    
    return allocation, dynamic_multiplier, vol_scalar

def evaluate_strategy(backtest_era, predictions, data, strategy_name, 
                     multiplier=None, target=None):
    """Evaluate a strategy with given predictions."""
    if target is None:
        target = config.TARGET
    
    allocation, dynamic_multiplier, vol_scalar = calculate_allocation(
        predictions, backtest_era, data, multiplier
    )
    
    # Calculate strategy returns
    market_daily_return = backtest_era['forward_returns'].values
    strategy_daily_return = allocation * market_daily_return
    
    # Store results in backtest_era for consistency with original
    temp_era = backtest_era.copy()
    temp_era['predicted_return'] = predictions
    temp_era['allocation'] = allocation
    temp_era['market_daily_return'] = market_daily_return
    temp_era['strategy_daily_return'] = strategy_daily_return
    
    # Information Coefficient
    actual_returns = backtest_era[target].values
    ic = np.corrcoef(predictions, actual_returns)[0, 1]
    
    # Annualized returns
    ann_market_return = np.mean(market_daily_return) * 252 * 100
    ann_strategy_return = np.mean(strategy_daily_return) * 252 * 100
    
    # Annualized volatility
    ann_market_vol = np.std(market_daily_return) * np.sqrt(252) * 100
    ann_strategy_vol = np.std(strategy_daily_return) * np.sqrt(252) * 100
    
    # Volatility ratio
    vol_ratio = ann_strategy_vol / ann_market_vol if ann_market_vol > 0 else 0
    vol_ratio_status = "(PASS)" if vol_ratio <= 1.20 else "(FAIL)"
    
    # Sharpe Ratio
    market_sharpe = (ann_market_return / ann_market_vol) if ann_market_vol > 0 else 0
    strategy_sharpe = (ann_strategy_return / ann_strategy_vol) if ann_strategy_vol > 0 else 0
    
    # Net Alpha
    net_alpha = ann_strategy_return - ann_market_return
    
    # Additional metrics
    max_allocation = np.max(allocation)
    min_allocation = np.min(allocation)
    mean_allocation = np.mean(allocation)
    
    return {
        'name': strategy_name,
        'ic': ic,
        'ann_market_return': ann_market_return,
        'ann_strategy_return': ann_strategy_return,
        'ann_market_vol': ann_market_vol,
        'ann_strategy_vol': ann_strategy_vol,
        'vol_ratio': vol_ratio,
        'vol_ratio_status': vol_ratio_status,
        'market_sharpe': market_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'net_alpha': net_alpha,
        'max_allocation': max_allocation,
        'min_allocation': min_allocation,
        'mean_allocation': mean_allocation,
        'predictions': predictions,
        'allocation': allocation
    }

def print_strategy_results(results):
    """Print formatted strategy results."""
    print(f"\n{results['name']} Backtest Scorecard")
    print("-" * 60)
    print(f"Information Coefficient (Correlation):    {results['ic']:+.4f}")
    print(f"Annualized Market Return:                {results['ann_market_return']:.2f}%")
    print(f"Annualized Strategy Return:              {results['ann_strategy_return']:.2f}%")
    print(f"Annualized Market Volatility:            {results['ann_market_vol']:.2f}%")
    print(f"Annualized Strategy Volatility:          {results['ann_strategy_vol']:.2f}%")
    print(f"Strategy-to-Market Volatility Ratio:     {results['vol_ratio']:.2f}x {results['vol_ratio_status']}")
    print(f"Market Buy-and-Hold Sharpe Ratio:         {results['market_sharpe']:.3f}")
    print(f"Strategy Sharpe Ratio:                    {results['strategy_sharpe']:.3f}")
    print(f"Net Alpha Generation:                    {results['net_alpha']:+.2f}%")
    print(f"Allocation Range: [{results['min_allocation']:.3f}, {results['max_allocation']:.3f}]")
    print(f"Mean Allocation:                         {results['mean_allocation']:.3f}")
    print("=" * 60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print(" ENHANCED HULL TACTICAL MARKET PREDICTION")
    print(" Multi-Model Comparison & Ensemble")
    print("="*60)
    
    # Step 1: Prepare data
    data = prepare_data(config.DATA_PATH)
    
    # Step 2: Initialize models
    models = get_models()
    
    # Step 3: Train and evaluate individual models
    models_results = train_and_evaluate_models(data, models)
    
    # Step 4: Backtest each model
    print("\n" + "="*60)
    print(" BACKTESTING INDIVIDUAL MODELS")
    print("="*60)
    
    backtest_era = data['backtest_era']
    X_backtest = data['X_backtest_scaled']
    
    all_results = []
    
    for name in config.MODELS_TO_TEST:
        if name not in models_results or models_results[name] is None:
            continue
        
        predictions = models_results[name]['model'].predict(X_backtest)
        result = evaluate_strategy(
            backtest_era, predictions, data, 
            strategy_name=name
        )
        all_results.append(result)
        print_strategy_results(result)
    
    # Step 5: Create and evaluate ensemble models
    print("\n" + "="*60)
    print(" BACKTESTING ENSEMBLE MODELS")
    print("="*60)
    
    # Weighted Average Ensemble
    wa_predictions = weighted_average_predict(models_results, X_backtest)
    wa_result = evaluate_strategy(
        backtest_era, wa_predictions, data,
        strategy_name="Weighted Average Ensemble"
    )
    all_results.append(wa_result)
    print_strategy_results(wa_result)
    
    # Create and test stacking ensemble
    base_models_for_ensemble = {k: v['model'] for k, v in models_results.items() 
                                 if v is not None and 'model' in v}
    if len(base_models_for_ensemble) >= 3:
        ensembles = get_ensemble_models(
            base_models_for_ensemble, 
            data['X_train_scaled'], 
            data['train_era'][config.TARGET].values
        )
        
        for name, ensemble_model in ensembles.items():
            predictions = ensemble_model.predict(X_backtest)
            result = evaluate_strategy(
                backtest_era, predictions, data,
                strategy_name=name.capitalize() + " Ensemble"
            )
            all_results.append(result)
            print_strategy_results(result)
    
    # Step 6: Summary comparison
    print("\n" + "="*60)
    print(" MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Sort by Sharpe ratio
    sorted_results = sorted(all_results, key=lambda x: x['strategy_sharpe'], reverse=True)
    
    print(f"\n{'Model':<30} {'Sharpe':>10} {'Return %':>12} {'Vol %':>10} {'IC':>8} {'Alpha %':>10}")
    print("-" * 80)
    
    for result in sorted_results:
        print(f"{result['name']:<30} {result['strategy_sharpe']:>10.3f} "
              f"{result['ann_strategy_return']:>12.2f} {result['ann_strategy_vol']:>10.2f} "
              f"{result['ic']:>+8.4f} {result['net_alpha']:>+10.2f}")
    
    print("-" * 80)
    
    # Identify best model
    best = sorted_results[0]
    print(f"\n🏆 Best Model: {best['name']}")
    print(f"   Sharpe Ratio: {best['strategy_sharpe']:.3f}")
    print(f"   Annual Return: {best['ann_strategy_return']:.2f}%")
    print(f"   Volatility: {best['ann_strategy_vol']:.2f}%")
    print(f"   Alpha: {best['net_alpha']:.2f}%")
    
    # Step 7: Save best model predictions for potential submission
    best_model_name = best['name']
    print(f"\n✓ Analysis complete. Best model: {best_model_name}")
    
    return all_results, models_results, best

# ============================================================================
# ADDITIONAL FUNCTIONS FOR MULTI-TARGET PREDICTION
# ============================================================================

def train_multi_target_models(data, models):
    """Train models to predict all target variables."""
    print("\n" + "="*60)
    print(" TRAINING MULTI-TARGET MODELS")
    print("="*60)
    
    train_era = data['train_era']
    X_train = data['X_train_scaled']
    targets = data['targets']
    
    multi_target_results = {}
    
    for target in targets:
        print(f"\nTraining models for target: {target}")
        y_train = train_era[target].values
        
        target_results = {}
        for name in config.MODELS_TO_TEST:
            if name not in models:
                continue
            
            try:
                model = models[name]
                model.fit(X_train, y_train)
                target_results[name] = model
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        
        multi_target_results[target] = target_results
    
    return multi_target_results

def ensemble_multi_target_predictions(multi_target_models, X, targets, method='weighted'):
    """Create ensemble predictions for all targets."""
    predictions = {}
    
    for target in targets:
        target_models = multi_target_models[target]
        
        if method == 'weighted':
            # Simple average for multi-target
            preds = []
            for name, model in target_models.items():
                preds.append(model.predict(X))
            predictions[target] = np.mean(preds, axis=0)
        elif method == 'best':
            # Use the best model for each target
            # This would need validation to determine
            best_name = list(target_models.keys())[0]  # Simplified
            predictions[target] = target_models[best_name].predict(X)
    
    return predictions

# ============================================================================
# IF RUNNING AS SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Run the main analysis
    all_results, models_results, best = main()
    
    # Optional: Multi-target prediction
    if config.PREDICT_ALL_TARGETS:
        print("\n" + "="*60)
        print(" RUNNING MULTI-TARGET PREDICTION")
        print("="*60)
        
        data = prepare_data(config.DATA_PATH)
        models = get_models()
        multi_target_models = train_multi_target_models(data, models)
        
        X_backtest = data['X_backtest_scaled']
        targets = data['targets']
        
        ensemble_preds = ensemble_multi_target_predictions(
            multi_target_models, X_backtest, targets
        )
        
        print("\nMulti-target ensemble predictions:")
        for target, preds in ensemble_preds.items():
            print(f"  {target}: mean={np.mean(preds):.6f}, std={np.std(preds):.6f}")
