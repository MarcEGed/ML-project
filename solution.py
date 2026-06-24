import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print("Loading and preparing historical data...")

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

# Impute missing values with training median
feature_medians = X_train.median().fillna(0)
X_train_imputed = X_train.fillna(feature_medians)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

print("Training Optimized Ridge Regressor...")

# Train Ridge model
model = Ridge(alpha=500.0, random_state=42)
model.fit(X_train_scaled, y_train)

print("Executing risk-managed strategy backtest...")

# Prepare backtest data
X_backtest = backtest_era[feature_cols].fillna(feature_medians)
X_backtest_scaled = scaler.transform(X_backtest)

# Predict excess returns
backtest_era['predicted_return'] = model.predict(X_backtest_scaled)

# Calculate volatility scalar (inverse volatility scaling)
train_vol_median = train_era['vol_proxy'].median()
backtest_era['vol_scalar'] = train_vol_median / (backtest_era['vol_proxy'] + 1e-8)

# Position sizing
BASE_MULTIPLIER = 80.0
backtest_era['dynamic_multiplier'] = BASE_MULTIPLIER * backtest_era['vol_scalar']
backtest_era['allocation'] = 1.0 + (backtest_era['predicted_return'] * backtest_era['dynamic_multiplier'])
backtest_era['allocation'] = np.clip(backtest_era['allocation'], 0.0, 2.0)

# Calculate strategy returns
backtest_era['market_daily_return'] = backtest_era['forward_returns']
backtest_era['strategy_daily_return'] = backtest_era['allocation'] * backtest_era['market_daily_return']

print("\n=============================================")
print(" TUNED RIDGE BACKTEST SCORECARD")
print("=============================================")

# Information Coefficient
ic = np.corrcoef(backtest_era['predicted_return'], backtest_era['market_forward_excess_returns'])[0, 1]
print(f"Information Coefficient (Correlation): {ic:+.4f}")

# Annualized returns
ann_market_return = backtest_era['market_daily_return'].mean() * 252 * 100
ann_strategy_return = backtest_era['strategy_daily_return'].mean() * 252 * 100
print(f"Annualized Market Return: {ann_market_return:.2f}%")
print(f"Annualized Strategy Return: {ann_strategy_return:.2f}%")

# Annualized volatility
ann_market_vol = backtest_era['market_daily_return'].std() * np.sqrt(252) * 100
ann_strategy_vol = backtest_era['strategy_daily_return'].std() * np.sqrt(252) * 100
print(f"Annualized Market Volatility: {ann_market_vol:.2f}%")
print(f"Annualized Strategy Volatility: {ann_strategy_vol:.2f}%")

# Volatility ratio check
vol_ratio = ann_strategy_vol / ann_market_vol if ann_market_vol > 0 else 0
print(f"Strategy-to-Market Volatility Ratio: {vol_ratio:.2f}x " + ("(PASS)" if vol_ratio <= 1.20 else "(FAIL)"))

# Sharpe Ratio
market_sharpe = (ann_market_return / ann_market_vol) if ann_market_vol > 0 else 0
strategy_sharpe = (ann_strategy_return / ann_strategy_vol) if ann_strategy_vol > 0 else 0
print(f"Market Buy-and-Hold Sharpe Ratio:  {market_sharpe:.3f}")
print(f"Your Upgraded Strategy Sharpe Ratio: {strategy_sharpe:.3f}")

# Net Alpha
print(f"Net Alpha Generation: {ann_strategy_return - ann_market_return:+.2f}%")
print("=============================================")