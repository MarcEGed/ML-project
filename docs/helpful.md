# Code Explainer: Feature Engineering & Phase 0 Model

---

## Part 1: `feature_engineering.py`

### What is this file for?

This file takes raw market data (a spreadsheet of daily numbers) and transforms it into hundreds of richer signals before feeding them into a model. The analogy: you are trying to predict tomorrow's weather, and you have a notebook of daily measurements. This file is the step where you enrich that notebook before studying it.

---

### The Pipeline (in order)

#### 1. `impute_features(df)`
Some days have missing values — a measurement wasn't recorded. This function fills in the blanks:
- First it looks **backward** (carry the last known value forward).
- Then it looks **forward** (fill any gaps still remaining at the very start of the series).

Target columns like `forward_returns` are left untouched — you can't "fill in" future returns.

---

#### 2. `get_base_feature_cols(df)`
Returns a plain list of all "input" column names — everything that isn't a target, date, or auxiliary column. This is the raw set of market factor columns you'll be working with.

---

#### 3. `rank_features_by_correlation(df, feature_cols, ...)`
Ranks the input columns by how strongly (in absolute terms) each one correlates with the target. Returns only the top N (default 30).

**Why?** The next step (rolling statistics) is expensive. Running it on every column would create an unmanageably large dataset. By filtering first, you focus the expensive computation on the columns most likely to carry real signal.

---

#### 4. `add_lag_features(df, feature_cols)`
For every input column, adds time-shifted copies at 1, 2, 3, 5, and 10 days back.

If `X` is a feature, you get `X_lag_1`, `X_lag_2`, etc.

**Why?** "Temperature has been rising for 3 days" is more useful than just knowing today's temperature. Lags let the model directly see recent history without having to infer it.

---

#### 5. `add_momentum_features(df, feature_cols)`
For every input column, adds the *change* over 1, 2, 3, and 5 days:

```
momentum_k(X, t) = X(t) - X(t-k)
```

**Why?** Raw levels are often slow-moving and hard to learn from. The *change* — "it went up 5°C since yesterday" — is often more informative than the raw value itself, and tends to correlate better with return targets.

---

#### 6. `add_rolling_features(df, top_features)`
For the top-N correlated features only, computes two statistics over 9 different time windows:

- **Rolling mean** — the smoothed average over the window (captures trend direction)
- **Rolling std** — how much variation there was over the window (captures uncertainty / local volatility)

| Window group | Windows | What it captures |
|---|---|---|
| Short | 3, 5, 10 days | Within-week effects |
| Medium | 20, 30, 60 days | Monthly / quarterly trends |
| Macro | 90, 120, 252 days | Year-scale regime shifts |

---

#### 7. `add_target_lags(df)`
Adds the **target itself**, lagged 1, 2, and 3 days, as new features (`target_lag_1`, `target_lag_2`, `target_lag_3`).

**Why?** This is called an autoregressive (AR) component. "What was the market return yesterday?" is a direct signal the model can use. If the market has been consistently positive for 3 days, that regime information is valuable.

---

#### 8. `add_vol_proxy(df)`
Averages the absolute values of all columns starting with `V` into a single column called `vol_proxy`. This is a rough summary of current market volatility.

Note: this column is **not** used as a training feature. It is only used downstream in the allocation/signal-scaling logic.

---

#### 9. `build_features(df)` — The Master Orchestrator
Calls all of the above in order (steps 1–8), then drops any rows still containing NaN (caused by lag/rolling burn-in at the start of the series). Returns the fully transformed DataFrame.

---

#### 10. `split_and_scale(df, exclude_last_n_days=180)`
Cuts the data **chronologically**:
- Everything except the last 180 rows → **training set**
- The last 180 rows → **backtest set**

Then fits a `StandardScaler` on the training set only, and applies it to both splits.

**Why scale?** Different columns have wildly different units (degrees vs km/h vs percentages). Rescaling everything to the same range prevents large-number columns from accidentally dominating the model.

**Why fit only on training data?** To prevent data leakage — the scaler must never "see" future data when deciding how to scale the past.

---

#### 11. `prepare_data(data_path)` — The Entry Point
The single function you call from your main script. It:
1. Loads the CSV
2. Sorts by date
3. Runs `build_features`
4. Runs `split_and_scale`
5. Returns a dictionary with everything needed: raw splits, scaled arrays, feature names, and the fitted scaler

---

### Pipeline Summary

```
CSV → fill missing values → select top-30 by correlation
    → add lags + momentum (all columns)
    → add rolling mean/std (top-30 only)
    → add lagged target values
    → drop incomplete rows
    → chronological train/backtest split
    → StandardScaler fit on train only
```

---
---

## Part 2: `phase0_final_improved.py`

### What is this file for?

This script trains two models (LightGBM and Ridge Regression) on the engineered features, converts each model's raw predictions into daily market allocation decisions, and measures how profitable and well-controlled the resulting strategy is. The goal is to replicate and slightly beat a **163rd place Kaggle competition solution** with a target Sharpe Ratio of 2.16.

---

### The `Config` class

A single settings panel at the top of the file — all magic numbers live here instead of being scattered through the code.

Key settings:

| Setting | Value | Meaning |
|---|---|---|
| `EXCLUDE_LAST_N_DAYS` | 180 | Days held back for backtesting |
| `TOP_N_FEATURES` | 30 | Features used for rolling stats |
| `RIDGE_ALPHA` | 100 | Regularisation strength for Ridge (lower = less constrained) |
| `MAX_VOL_RATIO` | 1.20 | Strategy risk must stay ≤ 120% of market risk |
| `TANH_BOUND / TANH_SCALE` | 0.006 / 3.0 | Controls how sensitive the allocation signal is |

---

### The Allocation Functions

These three functions work together to convert a raw model prediction (a number like 0.003) into an actual **bet size** — how much capital to put into the market today (0 = nothing, 1.2 = 120% of capital).

#### `convert_ret_to_signal(x)`
Squashes a prediction into a smooth 0–2 range using a **tanh** (S-shaped) curve.
- Small/uncertain predictions → stay near 1.0 (neutral, invest normally)
- Strong positive predictions → push toward 2.0 (invest more)
- Strong negative predictions → push toward 0.0 (stay out)

#### `calculate_volatility_multiplier(predictions)`
Tracks how wildly your recent predictions have been swinging (over the last 20 days).
- **Calm recent predictions** → multiplier > 1 (boost your bet — calm markets are a good opportunity)
- **Volatile recent predictions** → multiplier < 1 (shrink your bet — chaotic markets are risky)

#### `calculate_allocation_vol_constrained(predictions)`
Combines both: scale predictions by the volatility multiplier → run through tanh signal → clip at 1.2x maximum. This is the final bet size.

---

### The Two Models

#### LightGBM (`train_lgbm()`)
A powerful **gradient boosting** model — it builds many small decision trees sequentially, where each tree corrects the mistakes of all previous trees.

- Up to 5,000 trees, but uses **early stopping**: training stops automatically if the validation score hasn't improved in 100 rounds (prevents overfitting).
- 10% of the training data is silently set aside as an internal validation set to guide this stopping.

#### Ridge Regression
A simple **linear model** — it draws a straight hyperplane through the data. The `alpha=100` penalty prevents any single feature from dominating the fit.

**The key improvement over the 163rd place baseline:** they used `alpha=500` (more constrained). Lowering to `alpha=100` gives the model slightly more freedom to fit, which improves the score.

---

### The `evaluate()` function

Measures whether the strategy would actually have made money and stayed within the risk limit. The metrics it computes:

| Metric | Plain English |
|---|---|
| **IC** | How well your predictions correlated with what actually happened. Higher = real signal. |
| **Annualised Return** | Yearly profit as a percentage (×252 trading days) |
| **Volatility** | How wildly daily returns swing. Lower = more stable. |
| **Sharpe Ratio** | Return ÷ Risk. The competition's main score. Higher is better. |
| **Vol Ratio** | Your strategy's risk vs the market's risk. Must be ≤ 1.20 to pass the constraint. |
| **Alpha** | Extra return earned on top of just holding the market. Positive = you beat the market. |
| **Score** | Final competition score = Sharpe ratio, penalised if vol ratio exceeded 1.20. |

---

### How Does It Perform?

| Benchmark | Sharpe Score |
|---|---|
| 225th place (Bronze Medal, similar approach) | 1.958 |
| 163rd place (baseline this script replicates) | ~2.16 |
| This script's target | **2.16** |

A Sharpe Ratio of 2.16 is **very strong** for a market prediction model. For context:
- Most professional hedge funds target Sharpe ≥ 1.0
- Anything above 2.0 is considered excellent
- The 163rd/225th place solutions show that a single well-tuned model with good post-processing can outperform complex ensembles

---

### The Overall Flow

```
Load & engineer features (feature_engineering.py)
    → Train LightGBM (with early stopping)
    → Train Ridge Regression (alpha=100)
    → For each model:
        Apply volatility multiplier to predictions
        Convert to allocation signal via tanh
        Clip allocation to [0, 1.20]
        Measure Sharpe, Alpha, Vol Ratio
    → Report which model hit the 2.16 target
```