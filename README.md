# Hull Tactical Market Prediction - Solution

## Competition Overview

The **Hull Tactical Market Prediction** competition on Kaggle challenges participants to predict excess returns of the S&P 500 while managing volatility constraints. The competition tests the Efficient Market Hypothesis (EMH) by identifying patterns in financial markets that can generate consistent, risk-adjusted outperformance.

- **Objective**: Predict daily allocation to S&P 500 (0-2x leverage) to maximize a modified Sharpe ratio
- **Evaluation Metric**: Variant of Sharpe ratio that penalizes excessive volatility and underperformance
- **Prize Pool**: $100,000
- **Submission**: Kaggle Notebook with evaluation API, runtime limit 8-9 hours

## Requirements

### Submission Output
For each trading day, output an **allocation value** in the range **[0, 2]**:
- `0.0` = 0% in S&P 500 (100% cash)
- `1.0` = 100% in S&P 500
- `2.0` = 200% in S&P 500 (2x leverage)

### Key Constraints
- **No look-ahead bias**: Models must not use future data
- **Chronological validation**: Time-series data must be split chronologically, not randomly
- **Reproducibility**: Code must be deterministic and reproducible
- **Runtime**: Maximum 8 hours for training, 9 hours for forecasting phase
- **Internet**: Disabled during submission
- **External data**: Must be publicly available and freely accessible

### Data
The dataset contains:
- **130+ features**: Categorized as D (daily), E (economic), I (industry), M (macro), P (price), S (sentiment), V (volatility)
- **Target variables**: `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`
- **Time period**: Daily data spanning multiple years

## Solution Approach

### Architecture
The solution uses a **two-stage approach**:

1. **Return Prediction**: Train a Ridge regression model to predict excess returns
2. **Position Sizing**: Convert predicted returns to allocation using volatility-scaled positioning

### Why This Design?

This separation allows for:
- **Interpretable predictions**: Model predicts what the market will do (returns)
- **Independent optimization**: Position sizing logic can be tuned separately
- **Risk management**: Volatility scaling automatically reduces exposure in turbulent markets

## Code Explanation

### 1. Data Preparation
- Load and sort data chronologically by `date_id`
- Create volatility proxy from V-columns (mean absolute value)
- Split 80% train / 20% backtest chronologically
- Define features (exclude date_id, targets, and vol_proxy)
- Impute missing values with training median (no data leakage)
- Standardize features using StandardScaler

### 2. Model Training
- Use **Ridge regression** (L2 regularization) with alpha=500.0
- Target: `market_forward_excess_returns` (excess returns of S&P 500 over risk-free rate)
- Ridge helps prevent overfitting to historical financial noise

### 3. Allocation Calculation
- Predict excess returns on backtest data
- Calculate volatility scalar: `train_vol_median / (current_vol + epsilon)`
- Apply BASE_MULTIPLIER (80.0) to scale predictions to actionable allocations
- Compute allocation: `1.0 + (predicted_return * dynamic_multiplier)`
- Clip final allocation to [0.0, 2.0]

### 4. Performance Metrics
- **Information Coefficient (IC)**: Correlation between predicted and actual excess returns
- **Annualized Returns**: Strategy vs market return (252 trading days)
- **Annualized Volatility**: Strategy vs market volatility
- **Volatility Ratio**: Strategy volatility / market volatility (target: <= 1.20x)
- **Sharpe Ratio**: Return per unit of volatility (modified for competition)
- **Net Alpha**: Strategy return minus market return

## Key Design Decisions

### Chronological Split
Financial time-series data must be split chronologically, not randomly. Random splits create look-ahead bias where the model accidentally learns from future data.

### Volatility Proxy
We create a volatility proxy by averaging absolute values of all V-columns. This captures overall market turbulence and is used for dynamic position sizing.

### Inverse Volatility Scaling
The volatility scalar implements **inverse volatility targeting**:
- High volatility → scale down exposure (< 1.0)
- Low volatility → scale up exposure (> 1.0)
- This automatically reduces risk during turbulent periods

### BASE_MULTIPLIER = 80.0
This converts small percentage return predictions into meaningful allocation adjustments:
- Typical daily excess returns: ~±0.001 to ±0.01 (0.1% to 1%)
- With BASE_MULTIPLIER=80: return of 0.005 → allocation adjustment of 0.4
- Starting from base 1.0 → final allocation of 1.4 (140% long)

### Clipping to [0, 2]
Ensures all allocations meet competition requirements. The clipping is strict and non-negotiable.

## Performance Optimization Tips

1. **Tune BASE_MULTIPLIER**: Try values from 10-50. Higher values increase aggressiveness but may lead to excessive clipping.

2. **Feature Engineering**: Add lagged returns, moving averages, momentum indicators, and technical signals.

3. **Model Selection**: Try XGBoost, LightGBM, Lasso, or ensembles. Ridge is a good baseline but may not be optimal.

4. **Volatility Scaling**: Consider rolling volatility (20-day) instead of median-based scaling for more adaptive risk management.

5. **Walk-Forward Validation**: Implement expanding or rolling window validation instead of single train/test split for more robust testing.

6. **Drawdown Controls**: Add maximum drawdown limits or stop-loss logic to manage risk better.

## File Structure

```
ML-project/
├── README.md                 # This file
├── solution.py              # Main solution code
├── solution.ipynb           # Jupyter notebook version
├── train.csv                # Training data
├── test.csv                 # Test data
└── Comprehensive_Quantitative_Trading_Framework.pdf
```

## Usage

```bash
# Train and evaluate the model
python solution.py

# For Kaggle submission, adapt to use the evaluation API
# See: https://www.kaggle.com/code/sohier/hull-tactical-market-prediction-demo-submission
```

## Competition Links

- [Official Competition Page](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- [Demo Submission Notebook](https://www.kaggle.com/code/sohier/hull-tactical-market-prediction-demo-submission)
- [Hull Tactical Website](https://www.hulltactical.com/)
