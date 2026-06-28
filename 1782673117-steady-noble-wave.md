# Hull Tactical Market Prediction - Comprehensive Improvement Plan

## Executive Summary

**Current State**: 
- Baseline Ridge regression: Sharpe 0.896, 18.22% return, 2.65% alpha, IC +0.0491
- Advanced notebooks show scores ~1.05 using ensemble approaches
- **🥉 Bronze Medal Solution (225th place)**: Single LightGBM + tanh post-processing achieved **1.958 score**
- **🏆 New Benchmark: 163rd Place Solution**: Single model achieved **Sharpe 2.16** with critical data handling fixes

**🎯 New Priority Insight**: The **simplest approach** (single LightGBM with careful feature engineering and post-processing) **dramatically outperforms** complex ensembles (1.958 vs ~1.05). This is the paradigm shift.

**CRITICAL UPDATE FROM 163RD PLACE (SHARPE 2.16)**:
1. **Data Leakage Kills Score**: Using `bfill` for missing values causes **-0.6 score penalty** - must use `dropna()`
2. **Validation Leakage**: Test set is part of training data - **exclude last 180 days** for fair evaluation
3. **tanh > sigmoid**: Their exact tanh implementation with bound clipping outperforms sigmoid
4. **Robustness Proof**: Score held at 2.19 (train) → 2.16 (6 months later) = **no overfitting**

**Key Findings from Bronze Medal Solution (Score: 1.958)**:
1. **Simplicity Wins**: Single LightGBM regressor outperformed complex ensembles
2. **Target**: market_forward_excess_returns (regression, not classification)
3. **Feature Focus**: Quality over quantity - "carefully engineered features" vs hundreds of noisy features
4. **Post-Processing**: Scaled tanh function + adaptive volatility multiplier based on prediction volatility

**Updated Objective**: Replicate and improve upon the **163rd place approach** (target: >2.1 score) while maintaining the simplicity and robustness. **Fix data leakage first (P0.0) - this alone could add +0.6 to score.**

---

## Phase 0: Bronze Medal Replication (Days 1-3) - TOP PRIORITY

### 0.0 Data Foundation (P0.0 - CRITICAL, Do This First)
**163rd place proved these fixes are worth +0.6 score**
- [ ] **Data Cleaning Audit**: 
  - Scan all code for `bfill`, `ffill`, `interpolate` - **REPLACE WITH `.dropna()`**
  - Document all missing value patterns by feature
  - Verify no forward-looking data in features
- [ ] **Validation Split Fix**:
  - Exclude **last 180 days of training data** for evaluation (test set leakage)
  - Implement strict chronological split: `train = data.iloc[:TRAIN_END-180]`
- [ ] **Leakage Verification**:
  - Check feature calculation windows (no future data)
  - Verify target alignment (no lookahead)
  - Audit all rolling calculations

### 0.1 Extract Bronze Medal Approach (Score: 1.958)
- [ ] **Core Strategy**: Single LightGBM Regressor (not ensemble!)
- [ ] **Target**: market_forward_excess_returns
- [ ] **Feature Engineering**:
  - Multi-timeframe rolling statistics (short, medium, macro/quarterly)
  - Momentum & lag features (short-term lagged differences)
  - Autoregressive targets (lagged values of target)
  - Focus on **quality over quantity**
- [ ] **Post-Processing**:
  - Scaled hyperbolic tangent (tanh) function for base allocation
  - Adaptive volatility multiplier based on **trailing volatility of predictions**
  - Dynamic boost in calm markets, penalty in chaotic markets

### 0.2 Implement 163rd Place Solution (Score: 2.16)
- [ ] **Single LightGBM Model**: Conservative boosting with extensive hyperparameter tuning
- [ ] **Multi-Timeframe Features**:
  - Rolling means and std deviations across multiple windows
  - Short-term (days) to quarterly (macro) perspective
- [ ] **163rd Place Exact tanh Allocation** (production-ready):
  ```python
  def convert_ret_to_signal(x: np.ndarray, bound: float = 0.006, scale: float = 3.0) -> np.ndarray:
      """163rd place proven implementation - DO NOT MODIFY"""
      x_clipped = np.clip(x, -bound, bound)      # Step 1: Truncate to (-bound, bound)
      x_norm = x_clipped / bound                   # Step 2: Normalize to [-1, 1]
      x_scaled = x_norm * scale                  # Step 3: Amplify to [-scale, scale]
      return np.tanh(x_scaled) + 1.0             # Step 4: tanh to (-1,1), shift to (0,2)
  ```
- [ ] **Adaptive Volatility Boost**:
  - Track trailing volatility of model predictions (20-day window)
  - Multiplier = `base_multiplier * (target_volatility / prediction_volatility)`
  - Increase multiplier when prediction volatility is low (calm market)
  - Decrease multiplier when prediction volatility is high (chaotic market)

### 0.3 Validate & Benchmark
- [ ] **Replicate 1.958 score** locally (Bronze)
- [ ] **Target 2.1+ score** with 163rd place data fixes
- [ ] **Compare against** ensemble approaches (~1.05)
- [ ] **Verify volatility ratio** <= 1.20x
- [ ] **Test robustness** across different market periods
- [ ] **Critical Check**: Measure score improvement from data leakage fix (+0.6 expected)

---

## Phase 1: Bronze Enhancement & Feature Optimization (Week 1-2)

### 1.1 163rd Place Integration
- [ ] **Adopt their data pipeline**: No imputation, drop missing records
- [ ] **Implement their exact tanh function** as baseline
- [ ] **Replicate their evaluation protocol**: Exclude last 180 days
- [ ] **Verify robustness**: Check score stability across time periods

### 1.2 Feature Analysis from Advanced Solutions
- [ ] **Deconstruct 261-feature pipeline** from ht-final-v1.ipynb:
  - Lag specifications (V13: [1,2,3], M4: [1,2,3,5], etc.)
  - Rolling window features (mean, std, max) with windows [5,10]
  - Z-score normalization
  - Ratio features (DIV_PAIRS)
  - Difference features (MINUS_PAIRS)
  - Product features (PRODUCT_PAIRS)
  
- [ ] **Deconstruct 1187-feature pipeline** from htmp-optuna-v8-final-cpu.ipynb:
  - Lags: [1, 2, 5, 10] for all base features
  - Rolling statistics: mean, std, max with windows [5, 10]
  - Rank and Z-score features
  - Interaction features: Rank*Rank, Rank*Zscore
  - Targeted interactions for specific features (M4, M1, E1, V2)

### 1.3 Feature Importance Analysis
- [ ] **Run SHAP analysis** on both regression and classification models
- [ ] **Compare feature importance** across different model types
- [ ] **Identify overlapping high-importance features**
- [ ] **Create feature importance hierarchy** for prioritization

### 1.4 Feature Optimization
- [ ] **Test feature ablations**: Remove categories to measure impact
- [ ] **Optimize window sizes**: Test different rolling window parameters
- [ ] **Test lag depths**: Test 1-20 day lags for key features
- [ ] **Interaction feature expansion**: Create comprehensive interaction terms

---

## Phase 2: Model Diversity & Ensemble Testing (Week 3)

### 2.1 Classification vs Regression Analysis
- [ ] **Implement both approaches** side-by-side:
  - **Regression**: Predict excess_returns magnitude (current baseline)
  - **Classification**: Predict direction (excess_returns > 0) with confidence
  
- [ ] **Compare performance metrics**:
  - IC for regression vs classification accuracy
  - Sharpe ratio achievement
  - Volatility ratio maintenance
  - Robustness across market regimes

### 2.2 Model Diversity Expansion
- [ ] **Replicate Optuna Trial 3258 Models**:
  - XGBoost: max_depth=14, learning_rate=0.0326, reg_alpha=0.1723, reg_lambda=10.528
  - LightGBM: max_depth=8, learning_rate=0.0140, num_leaves=141, l1=0.9245
  - CatBoost: depth=11, learning_rate=0.0368, l2_reg=4.6013
  
- [ ] **Expand Model Zoo**:
  - ExtraTrees: n_estimators=1200, max_depth=30 (from ht-final-v1)
  - XGB2 variant: Different hyperparameters from first XGB
  - Random Forest with optimized parameters
  - Neural networks (simple MLP for comparison)

### 2.3 Ensemble Strategy Optimization
- [ ] **Replicate Weighting Schemes**:
  - ht-final-v1 weights: (0.08, 0.10, 0.52, 0.30) for (XGB, LGB, Extra, XGB2)
  - htmp-optuna weights: (4.18, 0.30, 1.93) for (XGB, LGB, CatBoost)
  
- [ ] **Test Alternative Weighting Strategies**:
  - **Sharpe-based weighting**: Weight by individual model Sharpe ratio
  - **IC-based weighting**: Weight by Information Coefficient
  - **Risk-adjusted weighting**: Weight by Sharpe/volatility
  - **Dynamic weighting**: Adapt weights based on recent performance
  
- [ ] **Ensemble Architecture Tests**:
  - Simple weighted average (current)
  - Stacking with meta-model
  - Voting ensemble
  - Dynamic model selection based on regime

---

## Phase 3: Advanced Position Sizing & Risk Management (Week 4)

### 3.1 Kelly Criterion Implementation (from ht-final-v1)
- [ ] **Replicate Kelly positioning**:
  - `sigma2 = KELLY_RET_STD ** 2 + 1e-12`
  - `f_raw = pred / sigma2`
  - `f_clipped = np.clip(f_raw, -1.0, 1.0)`
  - `allocation = 1.0 + f_clipped`
  - Final clip to [0.0, 2.0]

- [ ] **Confidence-Based Sizing (from htmp-optuna)**:
  - `confidence = 2 * abs(avg_prob - 0.5)`
  - `position = np.clip(confidence * ML_CONF_FACTOR, 0.0, 2.0)`
  - Optimize ML_CONF_FACTOR (current: 6.6209)

### 3.2 Advanced Position Sizing Strategies
- [ ] **Hybrid Kelly-Confidence**: Combine both approaches
- [ ] **Volatility-Scaled Confidence**: Multiply confidence by inverse volatility
- [ ] **Regime-Adaptive Sizing**: Different multipliers for different market regimes
- [ ] **Dynamic Multiplier**: BASE_MULTIPLIER that adapts to market conditions

### 3.3 Risk Management Enhancement
- [ ] **Implement Volatility Targeting**:
  - Rolling volatility calculation (20-day, 60-day windows)
  - Exponential weighted moving average volatility
  - Dynamic position scaling based on current vs historical volatility
  
- [ ] **Drawdown Controls**:
  - Maximum drawdown limits
  - Stop-loss mechanisms
  - Position reduction during drawdowns
  - Circuit breakers for extreme market conditions

### 3.4 Market Decorrelation (NEW - Address 163rd Place Weakness)
**163rd place explicitly warned: Model correlated with S&P 500 trend (underperformed in declines)**
- [ ] **Correlation Audit**: 
  - Measure Pearson/Spearman correlation between predictions and SPX returns
  - Identify features most correlated with SPX
  - Track correlation over time (regime dependence)
- [ ] **Decorrelation Techniques**:
  - **Residual Approach**: Regress predictions on SPX, use residuals as new features
  - **Orthogonal Features**: Create features orthogonal to SPX using Gram-Schmidt
  - **Market-Neutral Target**: Predict excess returns relative to SPX
- [ ] **Regime Filters**:
  - Reduce positions when SPX trend is negative
  - Add condition: `if spx_ma_20 < spx_ma_200: reduce_position_multiplier`
  - Test different regime detection methods (volatility, momentum, VIX)
- [ ] **Decorrelation Validation**:
  - Target correlation to SPX < 0.3
  - Verify performance improvement in declining markets

---

## Phase 4: Advanced Techniques & Optimization (Week 5)

### 4.1 Sample Weighting & Recency Boosting
- [ ] **Replicate Sample Weighting from ht-final-v1**:
  - Bad window: (7500, 8300) with bad_scale=0.1
  - Recency boost: Linear weight increase based on sample order
  - Combined: `w_quality * w_age`

- [ ] **Test Alternative Weighting Schemes**:
  - Volatility-based weighting
  - Regime-based weighting
  - Feature-specific weighting

### 4.2 Hyperparameter Optimization Framework
- [ ] **Replicate Optuna Setup**:
  - Define search space for each model type
  - Implement validation strategy (time-series split)
  - Set optimization objective (Sharpe ratio)
  
- [ ] **Expand Optimization**:
  - Joint optimization of model + position sizing parameters
  - Multi-objective optimization (Sharpe + volatility ratio)
  - Bayesian optimization for efficiency

### 4.3 Streaming History & Online Learning
- [ ] **Implement Robust Streaming History** (from ht-final-v1):
  - History buffer with cap (HIST_MAX_ROWS=1200)
  - Seed from training data for first prediction
  - Append and maintain sorted history
  - Handle local gateway overlap vs real future data
  
- [ ] **Online Learning Capability**:
  - Models that update with new streaming data
  - Concept drift detection
  - Adaptive window sizing

---

## Phase 5: Implementation & Validation (Week 6)

### 5.1 Production-Ready Implementation
- [ ] **Code Optimization**: Ensure runtime < 8 hours
  - Feature caching/precomputation
  - Parallel model training
  - Efficient data loading
- [ ] **Reproducibility**:
  - Set all random seeds
  - Document all dependencies
  - Containerized environment
- [ ] **Memory Management**: Handle large feature sets efficiently

### 5.2 Comprehensive Validation
- [ ] **Walk-Forward Validation**: Multiple train/test periods
  - Expanding window: Train on 1-2 years, test on next 3-6 months
  - Rolling window: Fixed training window, rolling forward
- [ ] **Out-of-Sample Testing**: Reserve final 20% for true OOS test
- [ ] **Cross-Market Validation**: If data permits, test on different markets

### 5.3 Final Model Selection
- [ ] **Multi-Criteria Optimization**: Balance Sharpe, volatility ratio, robustness
- [ ] **Ensemble of Best Models**: Combine top-performing approaches
- [ ] **Fallback Strategy**: Simple moving average crossover as baseline

---

## Immediate Action Plan (Next 7 Days)

### Day 0: Data Leakage Fix (P0.0 - DO THIS FIRST)
1. **Audit all code** for `bfill`, `ffill`, `interpolate` - replace with `.dropna()`
2. **Implement 180-day exclusion** for validation split
3. **Verify no forward-looking features** in calculation
4. **Document all missing value handling** decisions

### Days 1-3: 163rd Place Replication (P0 - CRITICAL)
1. **Implement exact 163rd place data pipeline** (no imputation, drop missing)
2. **Implement single LightGBM regressor** with market_forward_excess_returns target
3. **Create multi-timeframe features**: short, medium, quarterly rolling stats
4. **Add momentum & lag features**: short-term lagged differences, autoregressive targets
5. **Implement 163rd place exact tanh function** (bound=0.006, scale=3.0)
6. **Add adaptive volatility multiplier** based on trailing prediction volatility
7. **Validate and expect**: Score should jump from ~1.05 to **~1.65-1.95** (leakage fix + tanh)

### Days 4-5: Feature Optimization
1. **Feature importance analysis** (SHAP, permutation importance)
2. **Ablation studies**: Test impact of removing feature categories
3. **Window optimization**: Test different rolling window sizes
4. **Lag depth testing**: Find optimal lag periods

### Days 6-7: Hyperparameter Tuning & Decorrelation
1. **LightGBM hyperparameter sweep**: learning_rate, num_leaves, max_depth, etc.
2. **Post-processing parameter tuning**: tanh bound, scale, volatility multiplier
3. **Implement correlation audit**: Measure SPX correlation
4. **Test decorrelation techniques**: Residual approach, orthogonal features

---

## Success Metrics & Targets

### Primary Metrics
- **Kaggle Score**: Target **> 2.1** (163rd place: 2.16, Bronze medal: 1.958, previous ensemble: ~1.05)
- **Validation Score**: Target > 2.1 (must match or exceed Kaggle performance)
- **Generalization Delta**: Target < -0.1 (small drop from validation to Kaggle)
- **Information Coefficient**: Target > 0.15 (current: 0.0491)
- **Sharpe Ratio**: Target > 1.80 (current: 0.896)

### Secondary Metrics
- **Volatility Ratio**: Must remain <= **1.20x**
- **Maximum Drawdown**: Target < **20%**
- **SPX Correlation**: Target < **0.30** (163rd place weakness to address)
- **Win Rate**: Percentage of profitable days
- **Profit Factor**: Gross wins / gross losses
- **Data Cleanliness Score**: 100% no imputation, 100% leakage-free

### Validation Metrics
- **Walk-forward consistency**: Similar performance across all windows
- **Robustness**: Small performance variance across parameter sweeps
- **Runtime**: Must complete within 8-hour limit

---

## Resource Requirements

### Data
- Training data: train.csv (available)
- Test data: test.csv (available)
- Feature metadata: Understand feature definitions

### Compute
- CPU: Multi-core for parallel model training
- GPU: Optional for LightGBM/CatBoost (used in htmp-optuna notebook)
- Memory: 32GB+ for large feature sets (1187 features)
- Storage: Minimal (data fits in memory)

### Libraries
- Core: numpy, pandas, polars, scikit-learn
- Advanced: xgboost, lightgbm, catboost, optuna
- Optional: tensorflow/pytorch (for neural nets)
- Time-series: statsmodels
- Utilities: joblib, dill

---

## Risk Management

### Technical Risks
1. **Data Leakage**: **CRITICAL** - bfill/imputation causes -0.6 score (163rd place proven)
   - Mitigation: Use `.dropna()` only, audit all features
2. **Look-ahead Bias**: Strict chronological splits, no future data
3. **Overfitting**: Mitigation through proper time-series validation
4. **Runtime Exceeded**: Pre-compute features, optimize code
5. **Memory Issues**: Feature selection, efficient data types

### Model Risks
1. **Regime Change**: Model may not adapt to new market conditions
2. **Feature Decay**: Predictive power of features may diminish
3. **Correlation Breakdown**: Relationships may change over time
4. **SPX Dependence**: Model performance tied to market trend (163rd place warning)
   - Mitigation: Decorrelation techniques, regime filters

---

## Deliverables

1. **Data Audit Report**: Document all leakage fixes, missing value handling
2. **Feature Analysis Report**: Comprehensive analysis of all features
3. **Model Comparison Dashboard**: Performance across all tested models
4. **Optimal Strategy Code**: Final production-ready solution
5. **Validation Results**: Walk-forward and OOS performance
6. **Parameter Sensitivity Analysis**: Impact of key parameters
7. **Correlation Analysis**: SPX correlation measurements and decorrelation results

---

## Priority Order

1. **P0.0 (CRITICAL)**: Fix data leakage (no bfill, dropna, exclude last 180 days) - **+0.6 score potential**
2. **P0 (CRITICAL)**: Replicate 163rd place approach (single LightGBM + exact tanh + adaptive volatility)
3. **P1 (High)**: Optimize Bronze solution features and hyperparameters
4. **P1.5 (High)**: Market decorrelation (address 163rd place weakness)
5. **P2 (High)**: Test alternative post-processing strategies
6. **P3 (Medium)**: Explore model diversity (while keeping single-model focus)
7. **P4 (Medium)**: Advanced position sizing and risk management
8. **P5 (Low)**: Ensemble approaches (only if they don't add complexity)

---

## Next Steps

**Immediate (Today - Day 0)**:
1. **FIX DATA LEAKAGE**: Search and replace all `bfill`/`ffill` with `.dropna()`
2. **Implement 180-day validation split**
3. **Set up data audit script** to verify no leakage

**Short-term (This Week)**:
1. **Replicate 163rd place solution** (single LightGBM + exact tanh + adaptive volatility)
2. **Achieve validation score > 1.95** (Bronze) → **Target > 2.1** (163rd place)
3. **Implement walk-forward validation**
4. **Optimize hyperparameters and post-processing**
5. **Measure SPX correlation** and begin decorrelation work

**Medium-term (Next 2 Weeks)**:
1. **Improve upon 163rd place**: Test feature variations, alternative post-processing
2. **Push to Silver/Gold**: Target >2.1 score with robust performance
3. **Risk management**: Ensure volatility ratio <= 1.20x in all regimes
4. **Decorrelation**: Reduce SPX correlation to < 0.30
5. **Production readiness**: Optimize runtime, ensure reproducibility

---

## Key Insights from Advanced Solutions

### From 163rd Place Solution (Score: 2.16 Sharpe) ⭐ NEW BENCHMARK
- **Data Handling is Everything**: 
  - `bfill` causes **data leakage** and **-0.6 score penalty**
  - **Solution**: Drop missing records instead of imputation
- **Validation Strategy**: Exclude last 180 days of training data (test set is in training)
- **tanh Implementation**: Exact function with bound=0.006, scale=3.0 outperforms sigmoid
- **Robustness**: Score held at 2.19 (train) → 2.16 (6 months later) = **no overfitting**
- **Weakness**: Model correlated with S&P 500 trend (underperformed in declines)
- **Missed Opportunity**: Did not use TimeSeriesSplit for cross-validation

### From 225th Place Bronze Medal (Score: 1.958)
- **Simplicity First**: Single LightGBM regressor (not ensemble!)
- **Target**: market_forward_excess_returns (regression approach)
- **Feature Philosophy**: Quality over quantity, carefully engineered temporal patterns
- **Multi-Timeframe**: Short-term, medium-term, and macro/quarterly rolling statistics
- **Momentum**: Short-term lagged differences for key features
- **Autoregressive**: Lagged target values to capture market autocorrelation
- **Post-Processing**: Scaled tanh function + adaptive volatility multiplier based on prediction volatility
- **Risk Management**: Dynamic boost in calm markets, strict penalty in chaotic markets

### From ht-final-v1.ipynb:
- **RAW Ensemble**: Blend raw model predictions, then apply position sizing (Kelly)
- **Model-Specific Features**: Each model uses different feature subsets (18, 58, 19, 18)
- **Streaming History**: Robust handling of online prediction with history buffer
- **Sample Weighting**: Bad period downweighting + recency boosting
- **Kelly Positioning**: Optimal position sizing based on edge/odds

### From htmp-optuna-v8-final-cpu.ipynb:
- **Classification Approach**: Alternative approach with binary target
- **Optuna Optimization**: Systematic hyperparameter tuning framework
- **Confidence Scoring**: 2 * abs(prob - 0.5) * ML_CONF_FACTOR
- **Ensemble Weights**: XGB-dominant (4.18) with supporting LGB (0.30) and CatBoost (1.93)
- **Feature Scale**: 1187 features (but Bronze solution proves fewer, better features work better)

---

*Plan updated: 2026-06-28*
*Status: **P0.0 DATA LEAKAGE FIX IS NEW CRITICAL PATH** - Fix this first, then replicate 163rd place*
*Key Insight: **163rd place (2.16) > Bronze medal (1.958) - their data handling fixes are the difference maker**
*Expected Score Progression: Current (~1.05) → After leakage fix (~1.65) → After 163rd place replication (~2.1+) → After optimization (>2.1)*
