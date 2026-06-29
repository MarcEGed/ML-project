import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════════
# BIG PICTURE PLAN
# ══════════════════════════════════════════════════════════════════════════════
# We want to beat the market by predicting whether tomorrow's market return
# will be above or below average — and adjusting how much capital we deploy
# based on that prediction.
#
# STEP 1 — LOAD & PREPARE  : Clean the historical data and engineer features.
# STEP 2 — TRAIN MODEL     : Teach a Ridge regression model to predict
#                            excess market returns from those features.
# STEP 3 — BACKTEST        : Simulate the strategy on unseen future data:
#                            predict → size position → calculate return.
# STEP 4 — EVALUATE        : Score the strategy vs. simple buy-and-hold.
# STEP 5 — INTERPRET       : Translate numbers into plain-English guidance.
# ══════════════════════════════════════════════════════════════════════════════


# ─── STEP 1: LOAD & PREPARE DATA ─────────────────────────────────────────────
# Sort chronologically so we never accidentally train on "future" data.
# This is critical — peeking at future data would make results look great
# in testing but fail completely in real trading.
print("Loading data...")
df = pd.read_csv("train.csv")
df = df.sort_values(by="date_id").reset_index(drop=True)

# Build a volatility proxy from the V-columns (variance/volume features).
# High volatility = market is erratic. We'll use this later to reduce
# our position size on dangerous days and increase it on calm days.
v_cols = [col for col in df.columns if col.startswith('V')]
df['vol_proxy'] = df[v_cols].abs().mean(axis=1)

# Split 80/20 chronologically: train on the past, backtest on the future.
# We NEVER shuffle here — that would be cheating (data leakage).
split_idx = int(len(df) * 0.80)
train_era    = df.iloc[:split_idx].copy()
backtest_era = df.iloc[split_idx:].copy()

# These columns are targets or leakage risks — exclude them from model inputs.
# 'forward_returns' and 'market_forward_excess_returns' are what we're trying
# to predict, so they must never appear as input features.
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate',
                'market_forward_excess_returns', 'vol_proxy']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X_train = train_era[feature_cols]
y_train = train_era['market_forward_excess_returns']  # What we're predicting

# Fill missing values using the training set median.
# We compute medians ONLY on training data, then apply to both sets —
# otherwise we'd be using future information to fill past gaps.
feature_medians  = X_train.median().fillna(0)
X_train_imputed  = X_train.fillna(feature_medians)

# Standardize: rescale every feature to mean=0, std=1.
# Ridge regression is sensitive to feature scale, so this is required.
# Again, we fit the scaler on training data only, then apply to both.
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train_imputed)


# ─── STEP 2: TRAIN THE MODEL ──────────────────────────────────────────────────
# Ridge regression is linear regression with a regularization penalty (alpha).
# The penalty stops the model from overfitting to noise in the training data.
# alpha=500 is deliberately high — we want a conservative, stable model,
# not one that memorizes quirks of the past.
print("Training model...")
model = Ridge(alpha=500.0, random_state=42)
model.fit(X_train_scaled, y_train)
# After this, model.predict(X) returns an expected excess return for any day X.


# ─── STEP 3: BACKTEST — SIMULATE THE STRATEGY ON UNSEEN DATA ─────────────────
# This section answers: "If we had used this model historically, how would
# we have done?" We treat the backtest period as if it were the future.
print("Running backtest...")

X_backtest        = backtest_era[feature_cols].fillna(feature_medians)
X_backtest_scaled = scaler.transform(X_backtest)  # Use the SAME scaler as training

# Generate daily predictions: positive = model thinks market will beat average,
# negative = model thinks market will underperform.
backtest_era = backtest_era.copy()
backtest_era['predicted_return'] = model.predict(X_backtest_scaled)

# ── Position Sizing via Inverse Volatility Scaling ────────────────────────────
# The core idea: on volatile days, we take less risk. On calm days, more.
# vol_scalar > 1 means today is calmer than average → scale up.
# vol_scalar < 1 means today is more volatile than average → scale down.
train_vol_median         = train_era['vol_proxy'].median()
backtest_era['vol_scalar'] = train_vol_median / (backtest_era['vol_proxy'] + 1e-8)

# Translate the model's prediction into a capital allocation (0% to 200%).
# BASE_MULTIPLIER controls how aggressively we act on the model's signal.
# Formula: allocation = 100% baseline + adjustment based on predicted return.
#   → predicted_return > 0: we go above 100% (overweight the market)
#   → predicted_return < 0: we go below 100% (underweight the market)
# Clipped to [0%, 200%] so we never short or use extreme leverage.
BASE_MULTIPLIER              = 80.0
backtest_era['allocation']   = 1.0 + (backtest_era['predicted_return']
                                       * BASE_MULTIPLIER
                                       * backtest_era['vol_scalar'])
backtest_era['allocation']   = np.clip(backtest_era['allocation'], 0.0, 2.0)

# Daily return = how much of the market return we captured, scaled by allocation.
# allocation=1.0 → exactly match the market. 1.5 → 1.5x the market move.
backtest_era['market_daily_return']   = backtest_era['forward_returns']
backtest_era['strategy_daily_return'] = (backtest_era['allocation']
                                          * backtest_era['market_daily_return'])


# ─── STEP 4: COMPUTE PERFORMANCE METRICS ─────────────────────────────────────

# Information Coefficient (IC): correlation between predictions and actual returns.
# IC > 0 = model has directional edge. IC > 0.05 is considered good in finance.
ic = np.corrcoef(backtest_era['predicted_return'],
                 backtest_era['market_forward_excess_returns'])[0, 1]

# Annualize daily returns (252 trading days per year).
ann_market_return   = backtest_era['market_daily_return'].mean()   * 252 * 100
ann_strategy_return = backtest_era['strategy_daily_return'].mean() * 252 * 100

# Annualized volatility = how much the returns swing year-over-year.
# Higher volatility = more risk = bigger drawdowns possible.
ann_market_vol   = backtest_era['market_daily_return'].std()   * np.sqrt(252) * 100
ann_strategy_vol = backtest_era['strategy_daily_return'].std() * np.sqrt(252) * 100

# Volatility ratio: is our strategy riskier than just holding the market?
# > 1.20 means we're taking on significantly more risk — a red flag.
vol_ratio = ann_strategy_vol / ann_market_vol if ann_market_vol > 0 else 0
vol_pass  = vol_ratio <= 1.20

# Sharpe Ratio: return per unit of risk. Higher = better.
# > 1.0 is good. > 2.0 is exceptional. Negative means risk wasn't rewarded.
market_sharpe   = ann_market_return   / ann_market_vol   if ann_market_vol   > 0 else 0
strategy_sharpe = ann_strategy_return / ann_strategy_vol if ann_strategy_vol > 0 else 0

# Alpha: how much extra return we generated above just holding the market.
net_alpha = ann_strategy_return - ann_market_return

# Average allocation tells us the typical % of capital the model deployed.
avg_allocation = backtest_era['allocation'].mean() * 100


# ─── STEP 4: PRINT SCORECARD ──────────────────────────────────────────────────
print("\n=============================================")
print("           STRATEGY BACKTEST RESULTS        ")
print("=============================================")
print(f"  Prediction Accuracy (IC):       {ic:+.4f}")
print(f"")
print(f"  Market Return (annualized):     {ann_market_return:+.2f}%")
print(f"  Strategy Return (annualized):   {ann_strategy_return:+.2f}%")
print(f"  Alpha Generated:                {net_alpha:+.2f}%")
print(f"")
print(f"  Market Volatility:              {ann_market_vol:.2f}%")
print(f"  Strategy Volatility:            {ann_strategy_vol:.2f}%")
print(f"  Volatility Check:               {vol_ratio:.2f}x  {'✓ PASS' if vol_pass else '✗ FAIL'}")
print(f"")
print(f"  Market Sharpe Ratio:            {market_sharpe:.3f}")
print(f"  Strategy Sharpe Ratio:          {strategy_sharpe:.3f}")
print(f"")
print(f"  Average Daily Allocation:       {avg_allocation:.1f}% of capital")
print("=============================================")


# ─── STEP 5: PLAIN-ENGLISH INTERPRETATION ────────────────────────────────────
# Translate every number above into guidance a non-quant can act on.
print("\n WHAT ALL OF THIS MEANS FOR US")
print("─" * 45)

# Allocation guidance: what % of capital did the model typically deploy?
if avg_allocation > 100:
    alloc_note = (f"On average, the model recommended deploying ~{avg_allocation:.0f}% of your capital "
                  f"— meaning slight leverage (borrowing to invest more than you have).")
else:
    alloc_note = (f"On average, the model recommended having ~{avg_allocation:.0f}% of your capital "
                  f"invested at any given time (the rest held as cash).")

# Performance verdict: did the strategy actually beat the market?
if strategy_sharpe > market_sharpe and net_alpha > 0:
    perf_note = (f"The strategy outperformed the market: {ann_strategy_return:.1f}%/yr vs "
                 f"{ann_market_return:.1f}%/yr, adding {net_alpha:+.1f}% in extra annual return "
                 f"with {'similar' if vol_pass else 'higher'} risk "
                 f"(Sharpe {strategy_sharpe:.2f} vs {market_sharpe:.2f}).")
else:
    perf_note = (f"The strategy did NOT outperform buy-and-hold in this period "
                 f"({ann_strategy_return:.1f}%/yr vs {ann_market_return:.1f}%/yr). "
                 f"The model's predictions did not translate into a reliable edge.")

# Risk label based on annualized volatility thresholds.
# <10% = low (bond-like), 10–20% = moderate (typical equity), >20% = high (aggressive).
risk_level = "LOW" if ann_strategy_vol < 10 else "MODERATE" if ann_strategy_vol < 20 else "HIGH"
risk_note  = (f"Risk level: {risk_level} — the strategy's annualized volatility is "
              f"{ann_strategy_vol:.1f}% (market: {ann_market_vol:.1f}%). "
              f"Volatility check: {'PASSED ✓' if vol_pass else 'FAILED ✗ — strategy is significantly riskier than the market'}.")

print(alloc_note)
print()
print(perf_note)
print()
print(risk_note)
print("─" * 45)