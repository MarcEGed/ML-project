import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. LOAD DATA & DATA PREPARATION
# ==============================================================================
print("Loading and preparing historical data...")

# Load the dataset and sort chronologically to prevent future data leakage (look-ahead bias)
df = pd.read_csv("train.csv")
df = df.sort_values(by="date_id").reset_index(drop=True)

# Feature Engineering: Create an overall volatility proxy.
# We calculate the mean absolute value across all 'V' (volatility-related) columns
# to measure the average market turbulence for a given period.
v_cols = [col for col in df.columns if col.startswith('V')]
df['vol_proxy'] = df[v_cols].abs().mean(axis=1)

# Chronological Train / Backtest Split (80/20)
# The first 80% of time-series data is used for training; the remaining 20% evaluates out-of-sample performance.
split_idx = int(len(df) * 0.80)
train_era = df.iloc[:split_idx].copy()
backtest_era = df.iloc[split_idx:].copy()

# Define structural features (Ignore meta-tags and targets)
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns', 'vol_proxy']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X_train = train_era[feature_cols]
y_train = train_era['market_forward_excess_returns']

# Impute missing values using the median of the training set.
# We calculate medians STRICTLY on the training set to avoid data leakage from the backtest era.
feature_medians = X_train.median().fillna(0)
X_train_imputed = X_train.fillna(feature_medians)

# Standardize features (mean=0, variance=1) so the Ridge model penalizes all features equally
# and gradient descent converges effectively.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# ==============================================================================
# 2. TRAIN OPTIMIZED REGULARIZED ENGINE
# ==============================================================================
print("Training Optimized Ridge Regressor...")

# L2 Regularization (Ridge) helps prevent overfitting to historical financial noise.
# alpha=500.0 mathematically balances feature signal vs noise attenuation; 
# higher values drive coefficients closer to zero, smoothing out the signal.
model = Ridge(alpha=500.0, random_state=42)
model.fit(X_train_scaled, y_train)

# ==============================================================================
# 3. DYNAMIC VOLATILITY-SCALED INFERENCE BACKTEST
# ==============================================================================
print("Executing risk-managed strategy backtest...")

# Prepare backtest data using the SAME medians and scaler fitted on the training data.
X_backtest = backtest_era[feature_cols].fillna(feature_medians)
X_backtest_scaled = scaler.transform(X_backtest)

# Generate raw predicted excess returns using our trained Ridge model
backtest_era['predicted_return'] = model.predict(X_backtest_scaled)

# Calculate the running volatility scalar (Inverse Volatility Scaling).
# If current volatility is higher than historical norms, we scale down exposure (< 1).
# If current volatility is lower, we scale up exposure (> 1).
# The 1e-8 term prevents division by zero in zero-volatility environments.
train_vol_median = train_era['vol_proxy'].median()
backtest_era['vol_scalar'] = train_vol_median / (backtest_era['vol_proxy'] + 1e-8)

# Bounded multiplier designed to maximize alpha inside the 1.20x volatility cap
# BASE_MULTIPLIER (80.0) converts small percentage predictions into actionable leverage weights.
BASE_MULTIPLIER = 80.0
backtest_era['dynamic_multiplier'] = BASE_MULTIPLIER * backtest_era['vol_scalar']

# Calculate the final target allocation.
# 1.0 is the base allocation (100% long the market). We scale up or down based on our predictions.
# We clip allocations strictly between [0.0, 2.0] meaning:
#   0.0 = completely out of the market (cash)
#   2.0 = maximum allowed leverage (200% long)
backtest_era['allocation'] = 1.0 + (backtest_era['predicted_return'] * backtest_era['dynamic_multiplier'])
backtest_era['allocation'] = np.clip(backtest_era['allocation'], 0.0, 2.0)

# Performance accounting calculations:
# Apply our daily allocation weights to the actual market returns to get the strategy returns.
backtest_era['market_daily_return'] = backtest_era['forward_returns']
backtest_era['strategy_daily_return'] = backtest_era['allocation'] * backtest_era['market_daily_return']

# ==============================================================================
# 4. COMPUTE FINANCIAL PERFORMANCE SCORECARD
# ==============================================================================
print("\n=============================================")
print(" TUNED RIDGE BACKTEST SCORECARD")
print("=============================================")

# Information Coefficient (IC) measures the correlation between predicted and actual excess returns.
ic = np.corrcoef(backtest_era['predicted_return'], backtest_era['market_forward_excess_returns'])[0, 1]
print(f"Information Coefficient (Correlation): {ic:+.4f}")

# Annualize average daily returns (assuming 252 trading days in a standard financial year)
ann_market_return = backtest_era['market_daily_return'].mean() * 252 * 100
ann_strategy_return = backtest_era['strategy_daily_return'].mean() * 252 * 100
print(f"Annualized Market Return: {ann_market_return:.2f}%")
print(f"Annualized Strategy Return: {ann_strategy_return:.2f}%")

# Annualize daily standard deviation to measure long-term volatility
ann_market_vol = backtest_era['market_daily_return'].std() * np.sqrt(252) * 100
ann_strategy_vol = backtest_era['strategy_daily_return'].std() * np.sqrt(252) * 100
print(f"Annualized Market Volatility:{ann_market_vol:.2f}%")
print(f"Annualized Strategy Volatility: {ann_strategy_vol:.2f}%")

# Volatility Ratio checks if the strategy respects the imposed 1.20x risk limit
vol_ratio = ann_strategy_vol / ann_market_vol if ann_market_vol > 0 else 0
print(f"Strategy-to-Market Volatility Ratio:{vol_ratio:.2f}x " + ("(PASS)" if vol_ratio <= 1.20 else "(FAIL)"))

# Sharpe Ratio assesses risk-adjusted returns (Return generated per unit of Volatility)
# Note: This is a simplified calculation that assumes a 0% risk-free rate
market_sharpe = (ann_market_return / ann_market_vol) if ann_market_vol > 0 else 0
strategy_sharpe = (ann_strategy_return / ann_strategy_vol) if ann_strategy_vol > 0 else 0
print(f"Market Buy-and-Hold Sharpe Ratio:  {market_sharpe:.3f}")
print(f"Your Upgraded Strategy Sharpe Ratio: {strategy_sharpe:.3f}")

# Net Alpha compares the raw annualized strategy return directly against the baseline market return
print(f"Net Alpha Generation: {ann_strategy_return - ann_market_return:+.2f}%")
print("=============================================")