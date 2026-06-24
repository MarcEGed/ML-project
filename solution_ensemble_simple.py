"""
Simplified Multi-Model Ensemble Solution for Hull Tactical Market Prediction

This script extends the original solution by:
1. Testing multiple regression models
2. Creating weighted ensemble predictions
3. Using stacking ensemble
4. Maintaining the same risk-managed allocation approach
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

print("="*70)
print(" ENHANCED HULL TACTICAL - MULTI-MODEL ENSEMBLE")
print("="*70)

# ============================================================================
# DATA PREPARATION (Same as original)
# ============================================================================

print("\n[1] Loading and preparing historical data...")

# Load data chronologically
df = pd.read_csv("train.csv")
df = df.sort_values(by="date_id").reset_index(drop=True)

# Create volatility proxy from V-columns
v_cols = [col for col in df.columns if col.startswith('V')]
df['vol_proxy'] = df[v_cols].abs().mean(axis=1)

# Chronological train/backtest split (80/20)
split_idx = int(len(df) * 0.80)
train_era = df.iloc[:split_idx].copy()
backtest_era = df.iloc[split_idx:].copy()

# Define features
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns', 'vol_proxy']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X_train = train_era[feature_cols]
y_train = train_era['market_forward_excess_returns']

# Impute missing values with training median (no data leakage)
feature_medians = X_train.median().fillna(0)
X_train_imputed = X_train.fillna(feature_medians)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

print("\n[2] Initializing models...")

# Linear models
models = {
    'Ridge': Ridge(alpha=500.0, random_state=42),
    'Lasso': Lasso(alpha=0.001, random_state=42, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000),
    'BayesianRidge': BayesianRidge(),
}

# Tree-based models
models['RandomForest'] = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=5, 
    random_state=42, n_jobs=-1
)
models['GradientBoosting'] = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5,
    min_samples_leaf=5, random_state=42
)

print(f"Testing {len(models)} models...")

# ============================================================================
# MODEL TRAINING WITH TIME SERIES CROSS-VALIDATION
# ============================================================================

print("\n[3] Training models with time-series cross-validation...")

model_results = {}

tscv = TimeSeriesSplit(n_splits=5)

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = -mean_squared_error(y_val, y_pred)  # Negative MSE
        cv_scores.append(score)
    
    # Refit on full training data
    model.fit(X_train_scaled, y_train)
    
    avg_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    model_results[name] = {
        'model': model,
        'cv_mean': avg_score,
        'cv_std': std_score
    }
    print(f"    CV Score (neg MSE): {avg_score:.6f} ± {std_score:.6f}")

# ============================================================================
# CALCULATE ENSEMBLE WEIGHTS
# ============================================================================

print("\n[4] Calculating ensemble weights based on CV performance...")

# Get scores for valid models
valid_scores = {name: result['cv_mean'] for name, result in model_results.items()}
total_score = sum(valid_scores.values())

# Weight by performance (higher score = more weight)
weights = {}
for name, score in valid_scores.items():
    weights[name] = score / total_score if total_score > 0 else 1.0 / len(valid_scores)

print("\n  Model weights:")
for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"    {name}: {weight:.4f}")

# ============================================================================
# CREATE ENSEMBLE MODELS
# ============================================================================

print("\n[5] Creating ensemble models...")

# Stacking Ensemble
excess_estimators = [
    ('ridge', model_results['Ridge']['model']),
    ('lasso', model_results['Lasso']['model']),
    ('gb', model_results['GradientBoosting']['model'])
]

stacking_model = StackingRegressor(
    estimators=excess_estimators,
    final_estimator=Ridge(alpha=100.0),
    cv=3,
    n_jobs=-1
)
stacking_model.fit(X_train_scaled, y_train)
model_results['Stacking'] = {'model': stacking_model}

print("  ✓ Stacking ensemble created")

# ============================================================================
# PREPARE BACKTEST DATA
# ============================================================================

print("\n[6] Preparing backtest data...")

X_backtest = backtest_era[feature_cols].fillna(feature_medians)
X_backtest_scaled = scaler.transform(X_backtest)

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(predictions, model_name, BASE_MULTIPLIER=80.0):
    """Evaluate a model's predictions using competition metrics."""
    # Calculate volatility scalar
    train_vol_median = train_era['vol_proxy'].median()
    backtest_era_copy = backtest_era.copy()
    backtest_era_copy['vol_scalar'] = train_vol_median / (backtest_era_copy['vol_proxy'] + 1e-8)
    
    # Position sizing
    backtest_era_copy['dynamic_multiplier'] = BASE_MULTIPLIER * backtest_era_copy['vol_scalar']
    backtest_era_copy['allocation'] = 1.0 + (predictions * backtest_era_copy['dynamic_multiplier'])
    backtest_era_copy['allocation'] = np.clip(backtest_era_copy['allocation'], 0.0, 2.0)
    
    # Calculate returns
    backtest_era_copy['market_daily_return'] = backtest_era_copy['forward_returns']
    backtest_era_copy['strategy_daily_return'] = backtest_era_copy['allocation'] * backtest_era_copy['market_daily_return']
    
    # Metrics
    ic = np.corrcoef(predictions, backtest_era_copy['market_forward_excess_returns'])[0, 1]
    
    ann_market_return = backtest_era_copy['market_daily_return'].mean() * 252 * 100
    ann_strategy_return = backtest_era_copy['strategy_daily_return'].mean() * 252 * 100
    
    ann_market_vol = backtest_era_copy['market_daily_return'].std() * np.sqrt(252) * 100
    ann_strategy_vol = backtest_era_copy['strategy_daily_return'].std() * np.sqrt(252) * 100
    
    vol_ratio = ann_strategy_vol / ann_market_vol if ann_market_vol > 0 else 0
    vol_status = "(PASS)" if vol_ratio <= 1.20 else "(FAIL)"
    
    market_sharpe = (ann_market_return / ann_market_vol) if ann_market_vol > 0 else 0
    strategy_sharpe = (ann_strategy_return / ann_strategy_vol) if ann_strategy_vol > 0 else 0
    
    net_alpha = ann_strategy_return - ann_market_return
    
    return {
        'name': model_name,
        'ic': ic,
        'ann_return': ann_strategy_return,
        'ann_vol': ann_strategy_vol,
        'vol_ratio': vol_ratio,
        'vol_status': vol_status,
        'sharpe': strategy_sharpe,
        'market_sharpe': market_sharpe,
        'alpha': net_alpha,
        'allocation_mean': backtest_era_copy['allocation'].mean(),
        'allocation_min': backtest_era_copy['allocation'].min(),
        'allocation_max': backtest_era_copy['allocation'].max()
    }

# ============================================================================
# EVALUATE ALL MODELS
# ============================================================================

print("\n" + "="*70)
print(" BACKTEST RESULTS")
print("="*70)

all_results = []

# Evaluate individual models
print("\n--- Individual Models ---")
for name in models.keys():
    predictions = model_results[name]['model'].predict(X_backtest_scaled)
    results = evaluate_model(predictions, name)
    all_results.append(results)
    
    print(f"\n{name}:")
    print(f"  Sharpe: {results['sharpe']:.3f} | Return: {results['ann_return']:.2f}% | Vol: {results['ann_vol']:.2f}%")
    print(f"  IC: {results['ic']:+.4f} | Alpha: {results['alpha']:+.2f}% | Vol Ratio: {results['vol_ratio']:.2f}x {results['vol_status']}")
    print(f"  Allocation: [{results['allocation_min']:.3f}, {results['allocation_max']:.3f}]")

# Evaluate Stacking Ensemble
predictions = stacking_model.predict(X_backtest_scaled)
results = evaluate_model(predictions, "Stacking Ensemble")
all_results.append(results)
print(f"\nStacking Ensemble:")
print(f"  Sharpe: {results['sharpe']:.3f} | Return: {results['ann_return']:.2f}% | Vol: {results['ann_vol']:.2f}%")
print(f"  IC: {results['ic']:+.4f} | Alpha: {results['alpha']:+.2f}% | Vol Ratio: {results['vol_ratio']:.2f}x {results['vol_status']}")

# Evaluate Weighted Average Ensemble
weighted_preds = np.zeros(len(X_backtest_scaled))
for name, weight in weights.items():
    weighted_preds += model_results[name]['model'].predict(X_backtest_scaled) * weight

results = evaluate_model(weighted_preds, "Weighted Average Ensemble")
all_results.append(results)
print(f"\nWeighted Average Ensemble:")
print(f"  Sharpe: {results['sharpe']:.3f} | Return: {results['ann_return']:.2f}% | Vol: {results['ann_vol']:.2f}%")
print(f"  IC: {results['ic']:+.4f} | Alpha: {results['alpha']:+.2f}% | Vol Ratio: {results['vol_ratio']:.2f}x {results['vol_status']}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" MODEL COMPARISON SUMMARY")
print("="*70)

# Sort by Sharpe ratio
sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Model':<25} {'Sharpe':>8} {'Return %':>10} {'Vol %':>8} {'IC':>8} {'Alpha %':>8}")
print("-" * 70)

for r in sorted_results:
    print(f"{r['name']:<25} {r['sharpe']:>8.3f} {r['ann_return']:>10.2f} {r['ann_vol']:>8.2f} {r['ic']:>+8.4f} {r['alpha']:>+8.2f}")

print("-" * 70)

best = sorted_results[0]
print(f"\n🏆 Best Model: {best['name']}")
print(f"   Sharpe Ratio: {best['sharpe']:.3f}")
print(f"   Annual Return: {best['ann_return']:.2f}%")
print(f"   Volatility: {best['ann_vol']:.2f}%")
print(f"   Information Coefficient: {best['ic']:+.4f}")
print(f"   Alpha: {best['alpha']:+.2f}%")
print(f"   Allocation Range: [{best['allocation_min']:.3f}, {best['allocation_max']:.3f}]")

print("\n✅ Analysis complete!")
