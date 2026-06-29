# Phase 1: Bronze Enhancement & Feature Optimization

This document describes the implementation of Phase 1 from the ML project roadmap, covering:

- **1.1 163rd Place Integration** - Data pipeline, tanh function, evaluation protocol
- **1.2 Feature Analysis** - Deconstruction of 261-feature and 1187-feature pipelines
- **1.3 Feature Importance Analysis** - SHAP analysis across model types
- **1.4 Feature Optimization** - Ablations, window sizes, lag depths, interaction terms

## Implementation Files

- `phase1_implementation.py` - Main Phase 1 implementation
- `solution_bronze.py` - Baseline bronze medal implementation
- `phase0_final.py` - Phase 0 volatility constrained implementation

## 1.1 163rd Place Integration

### Implemented Components

**Data Pipeline:**
- No imputation approach (drop missing records)
- Exclude last 180 days for evaluation protocol
- Chronological sorting and splitting

**Tanh Function:**
```python
def convert_ret_to_signal_163rd(x, bound=0.006, scale=3.0):
    x_clipped = np.clip(x, -bound, bound)
    x_norm = x_clipped / bound
    x_scaled = x_norm * scale
    return np.tanh(x_scaled) + 1.0
```

**Volatility Multiplier:**
- Rolling window calculation (default 20 days)
- Inverse relationship: low volatility -> high multiplier
- Clipping to [0, MAX_VOL_RATIO] to ensure volatility constraint

**Evaluation Protocol:**
- Information Coefficient (IC) calculation
- Annualized returns and volatility
- Volatility ratio check (must be <= 1.20)
- Sharpe ratio and net alpha calculation
- Competition-like scoring formula

## 1.2 Feature Analysis from Advanced Solutions

### 261-Feature Pipeline (from ht-final-v1.ipynb)

**Lag Features:**
- V13: [1, 2, 3]
- M4: [1, 2, 3, 5]
- V7: [1, 2]
- E19: [1, 2, 5, 90]
- P12: [20], I4: [20], M5: [180]
- P10: [2], M12: [90, 10]
- V9: [1, 3], M3: [2]
- P5: [1, 3], P4: [5], S8: [1]

**Rolling Statistics:**
- Mean windows: V13, M4, E19, S2, P10, P11, M11, V9, I4, M7, M1
- Std windows: E19, I2, P6, M4, S5, S10, P2, M1

**EMA Features:**
- M4: [2, 3, 5, 10, 20]
- E19: [2, 3, 5]
- V13: [3, 5]
- M3: [2, 10]
- V9: [2, 3, 10]
- P7: [10, 180]
- M2: [5, 10]

**Interaction Features:**
- **DIV_PAIRS** (19 pairs): (E19,P4), (P6,M12), (V13,P4), (P8,P13), (V7,P4), ...
- **MINUS_PAIRS** (20 pairs): (P8,M2), (E19,P6), (V13,E19), (V7,I2), (V13,S2), ...
- **PRODUCT_PAIRS** (5 pairs): (V13,E19), (V13,M4), (V9,E3), (V7,P2), (E19,P10)

**Z-Score Features:**
- Based on rolling std specs with epsilon stabilization

**Relative Features:**
- Division by rolling mean
- Difference from rolling mean

### 1187-Feature Pipeline (from htmp-optuna-v8-final-cpu.ipynb)

**Lag Features:**
- Applied to all base features with lags [1, 2, 5, 10]

**Rolling Statistics:**
- Windows: [5, 10]
- Stats: mean, std, max
- Applied to all base features

**Rank and Z-Score Features:**
- Rank features for all base features using min ranking
- Z-score features: (x - mean) / std with safe division

**Interaction Features:**
- **Targeted Interactions**: For M4, M1, E1, V2
  - c * c_RANK
  - c / (c_RANK + 1e-6)
- **Rank * Rank**: Unique pairs from top 12 rank features
- **Rank * ZScore**: Pairs from top 9 rank and z-score features

## 1.3 Feature Importance Analysis

### SHAP Analysis
- TreeExplainer for tree-based models (LightGBM, RandomForest)
- LinearExplainer for linear models (Ridge, LogisticRegression)
- Automatic sampling for large datasets (>1000 instances)
- Feature importance ranking by mean absolute SHAP values

### Cross-Model Comparison
- Compare feature importance across:
  - LightGBM Regressor
  - Ridge Regression
  - Random Forest Regressor
- Identify overlapping high-importance features
- Create feature importance hierarchy based on average rank

### Feature Importance Hierarchy
- Rank features by average importance across models
- Identify features consistently in top 20 across all models
- Provide prioritization for feature selection

## 1.4 Feature Optimization

### Feature Ablation Study
- Categorize features by prefix (D, E, M, P, S, V, I)
- Remove each category and measure impact on:
  - Model score (R²)
  - Information Coefficient (IC)
- Identify most impactful feature categories

### Lag Depth Optimization
- Test lag depths: [1, 2, 3, 5, 10, 15, 20]
- Applied to key features
- Measure impact on score and IC
- Identify optimal lag depth

### Window Size Optimization
- Test window sizes: [3, 5, 10, 15, 20, 25, 30, 60]
- Test rolling statistics: mean, std
- Measure impact on score and IC
- Identify optimal window size

### Interaction Feature Expansion
- Create comprehensive pairwise interactions:
  - Products (A * B)
  - Ratios (A / B)
  - Differences (A - B)
- Limit to first 15 base features for computational efficiency
- Evaluate impact on model performance

## Usage

### Run Complete Phase 1 Comparison
```bash
python phase1_implementation.py
```

### Import and Use Individual Components
```python
from phase1_implementation import (
    # 1.1 163rd Place Integration
    prepare_data_163rd,
    convert_ret_to_signal_163rd,
    calculate_allocation_163rd,
    evaluate_strategy_163rd,
    
    # 1.2 Feature Analysis
    create_full_261_feature_pipeline,
    create_full_1187_feature_pipeline,
    create_lag_features_261,
    create_rolling_features_261,
    create_ema_features_261,
    create_interaction_features_261,
    create_square_features_261,
    create_zscore_features_261,
    create_relative_features_261,
    
    # 1.3 Feature Importance
    run_shap_analysis,
    compare_feature_importance_across_models,
    
    # 1.4 Feature Optimization
    test_feature_ablations,
    test_lag_depths,
    test_window_sizes,
    create_comprehensive_interaction_terms,
)

# Example usage
data = prepare_data_163rd("train.csv", config)
model = train_lightgbm_model(data['X_train_scaled'], data['train_era']['market_forward_excess_returns'].values, config)
results = evaluate_strategy_163rd(data['backtest_era'], model.predict(data['X_backtest_scaled']), data, config)
```

## Configuration

The `Config` class contains all configurable parameters:

```python
class Config:
    DATA_PATH = "train.csv"
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    EXCLUDE_LAST_N_DAYS = 180  # 163rd place protocol
    TARGET = 'market_forward_excess_returns'
    
    # 163rd Place settings
    TANH_BOUND = 0.006
    TANH_SCALE = 3.0
    VOL_WINDOW = 20
    MAX_VOL_RATIO = 1.20
    BASE_MULTIPLIER = 80.0
    
    # Feature engineering specs
    LAG_SPECS_261 = {...}
    ROLLING_MEAN_SPECS_261 = {...}
    ROLLING_STD_SPECS_261 = {...}
    EMA_SPECS_261 = {...}
    LAGS_1187 = [1, 2, 5, 10]
    WINDOWS_1187 = [5, 10]
    STATS_1187 = ['mean', 'std', 'max']
```

## Results Summary

Initial test results from running the complete pipeline:

### 163rd Place Pipeline
- Features: 94
- IC: -0.0102
- Score: 0.2540
- Vol Ratio: 1.03x (PASS)
- Sharpe: 0.254

### 261-Feature Pipeline
- Features: 256
- IC: -0.0964
- Score: 0.3321
- Vol Ratio: 1.24x
- Sharpe: 0.339

### Simplified 1187-Feature Pipeline
- Features: 335
- IC: -0.1150
- Score: 0.2512
- Vol Ratio: 1.29x
- Sharpe: 0.264

## Dependencies

Required:
- numpy
- pandas
- lightgbm
- scikit-learn

Optional:
- shap (for SHAP analysis)
- optuna (for advanced optimization)

## Next Steps

1. **Improve Data Handling**: The current 163rd place pipeline drops too many rows. Consider:
   - Imputation with training mean/median
   - Forward fill for time-series data
   - More selective feature dropping

2. **Enhance Feature Engineering**:
   - Combine best elements from both 261 and 1187 pipelines
   - Optimize feature selection based on SHAP analysis
   - Test different feature combinations

3. **Model Optimization**:
   - Hyperparameter tuning
   - Ensemble methods
   - Classification vs Regression comparison

4. **Robustness Testing**:
   - Cross-validation across different time periods
   - Walk-forward validation
   - Stress testing during volatile market conditions