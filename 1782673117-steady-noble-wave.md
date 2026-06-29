# Hull Tactical Market Prediction - Simplified Improvement Plan

## Executive Summary

**Current State**: 
- Baseline Ridge regression: Sharpe 0.896, 18.22% return, 2.65% alpha, IC +0.0491
- Previous ensemble approaches: ~1.05 score
- **🏆 Benchmark: 163rd Place Solution**: **Single LightGBM** achieved **Sharpe 2.16** with data handling fixes
- **🥉 Bronze Medal (225th place)**: Single LightGBM + tanh achieved **1.958 score**

**🎯 Core Insight**: **Simplicity wins.** A single well-tuned LightGBM model with proper data handling and post-processing **doubles the score** of complex ensembles (2.16 vs 1.05). This is the proven path forward.

**CRITICAL LESSON FROM 163RD PLACE**:
1. **Data Leakage = -0.6 score**: `bfill`/`ffill` for missing values **destroys performance** → use `.dropna()`
2. **Validation Leakage**: Test set embedded in training data → exclude last **180 days**
3. **tanh > sigmoid**: Their exact implementation (bound=0.006, scale=3.0) outperforms alternatives
4. **Robustness**: 2.19 (train) → 2.16 (6 months later) = **no overfitting**

**Key Findings from Bronze Medal (1.958)**:
- Single LightGBM regressor (not ensemble)
- Target: `market_forward_excess_returns`
- Feature philosophy: **Quality over quantity** - carefully engineered temporal patterns
- Post-processing: Scaled tanh + adaptive volatility multiplier
- Multi-timeframe: Short, medium, quarterly rolling statistics

**Updated Objective**: **Replicate 163rd place (2.16) first, then optimize.** Fix data leakage (P0.0) → Replicate simple model (P0) → Optimize (P1) → Address weaknesses (P1.5). **Complexity is the enemy.**

---

## Core Philosophy

### What We Know Works
| Approach | Score | Complexity | Status |
|----------|-------|------------|--------|
| Ridge baseline | 0.896 | Low | Current |
| Previous ensembles | ~1.05 | High | **Abandon** |
| Bronze (single LGBM) | 1.958 | Low | Target |
| **163rd place (single LGBM)** | **2.16** | **Low** | **Benchmark** |

### Principles
1. **Single model first**: Prove we can hit >2.1 with one LightGBM before considering ensembles
2. **Data quality > feature quantity**: Fix leakage, then optimize features
3. **Post-processing matters**: tanh function and volatility scaling are critical
4. **Only add complexity when proven necessary**: No ensembles, no model zoos, no 1187-feature pipelines
5. **Address known weaknesses**: SPX correlation is the only stated gap in 163rd place

---

## Phase 0: Data Foundation & 163rd Place Replication (Days 0-3) - P0.0 & P0

### 0.0 Data Leakage Fix (P0.0 - CRITICAL - Day 0)
**163rd place proved this alone is worth +0.6 score.**
- [ ] **Audit all code** for `bfill`, `ffill`, `interpolate` → **replace with `.dropna()`**
- [ ] **Implement 180-day exclusion**: `train = data.iloc[:TRAIN_END-180]`
- [ ] **Verify no forward-looking data**: Check all rolling windows, lags, target alignment
- [ ] **Document**: All missing value patterns, handling decisions

### 0.1 Replicate 163rd Place Core (P0 - Days 1-3)
**Target: Achieve 2.1+ validation score with single model.**

- [ ] **Data Pipeline**: 
  - No imputation, `dropna()` only
  - Strict chronological split (exclude last 180 days)
  
- [ ] **Model**: Single LightGBM Regressor
  - Target: `market_forward_excess_returns`
  - Conservative hyperparameters (start with Bronze medal params, tune later)
  
- [ ] **Features**: Multi-timeframe (start with Bronze medal approach)
  - Short-term rolling stats (5-20 day windows): mean, std, momentum
  - Medium-term rolling stats (20-60 day windows)
  - Quarterly/macro perspective (60-252 day windows)
  - Lag features: 1-5 day lags for key predictors
  - Autoregressive: Lagged target values (1-3 days)
  - **Philosophy**: Quality over quantity, ~50-100 features max
  
- [ ] **Post-Processing** (163rd place exact implementation):
  ```python
  def convert_ret_to_signal(x: np.ndarray, bound: float = 0.006, scale: float = 3.0) -> np.ndarray:
      """163rd place proven implementation - DO NOT MODIFY"""
      x_clipped = np.clip(x, -bound, bound)      # Truncate to (-bound, bound)
      x_norm = x_clipped / bound                   # Normalize to [-1, 1]
      x_scaled = x_norm * scale                  # Amplify to [-scale, scale]
      return np.tanh(x_scaled) + 1.0             # tanh to (-1,1), shift to (0,2)
  ```
  
- [ ] **Adaptive Volatility Multiplier**:
  - Track trailing volatility of predictions (20-day window)
  - Multiplier = `base_multiplier * (target_volatility / prediction_volatility)`
  - Boost when prediction volatility is low (calm markets)
  - Penalty when prediction volatility is high (chaotic markets)

- [ ] **Validation**:
  - [ ] Replicate 1.958 score (Bronze baseline)
  - [ ] Achieve 2.1+ score (163rd place target)
  - [ ] Verify volatility ratio <= 1.20x
  - [ ] Measure score improvement from data fix (+0.6 expected)

---

## Phase 1: Single Model Optimization (Week 1) - P1

**Goal: Push single LightGBM beyond 2.16.**

### 1.1 Feature Refinement
**Cherry-pick from advanced notebooks, don't replicate entire pipelines.**

- [ ] **Feature Importance Analysis**:
  - Run SHAP on replicated 163rd place model
  - Identify top 20-30 most predictive features
  - Remove noise features (low importance, high correlation)
  
- [ ] **Test Select Feature Ideas from Advanced Solutions**:
  - Lag specs: [1,2,3,5,10] for top features only (not all 261/1187)
  - Rolling windows: mean, std with [5,10,20,60] (test which add value)
  - Z-score normalization for high-impact features
  - Targeted interaction terms (e.g., M4*M1 if both are important)
  
- [ ] **Ablation Studies**:
  - Remove feature categories one by one, measure impact
  - Keep only features that improve score

### 1.2 Hyperparameter Optimization
- [ ] **LightGBM Tuning** (Optuna or grid search):
  - learning_rate: [0.005, 0.01, 0.02, 0.03, 0.05]
  - num_leaves: [31, 63, 127, 255]
  - max_depth: [6, 8, 10, 12]
  - min_child_samples: [20, 50, 100]
  - reg_alpha, reg_lambda: [0, 0.1, 1, 10]
  - n_estimators: [500, 1000, 2000]
  
- [ ] **Post-Processing Tuning**:
  - tanh bound: [0.004, 0.006, 0.008, 0.01]
  - tanh scale: [2.0, 3.0, 4.0, 5.0]
  - Adaptive volatility multiplier base: [0.8, 1.0, 1.2]
  - Volatility window: [10, 20, 30, 60] days

### 1.3 Validation & Robustness
- [ ] **Walk-forward validation**: 3-5 chronological splits
- [ ] **Out-of-sample test**: Reserve final 20% for true OOS
- [ ] **Parameter sensitivity**: Check score stability across hyperparameter ranges
- [ ] **Target**: Consistent >2.1 score across all validation windows

---

## Phase 2: Address Known Weaknesses (Week 2) - P1.5

**163rd place's only stated weakness: SPX correlation (underperformed in declines).**

### 2.1 Market Decorrelation
- [ ] **Correlation Audit**:
  - Measure Pearson/Spearman between predictions and SPX returns
  - Identify top features correlated with SPX
  - Track correlation over time (regime dependence)
  
- [ ] **Decorrelation Techniques** (test in order of simplicity):
  1. **Residual Approach**: Regress model predictions on SPX, use residuals
  2. **Feature Residuals**: Create SPX-residual features (feature - beta*SPX)
  3. **Regime Filter**: Reduce positions when SPX trend is negative
     - Condition: `if spx_ma_20 < spx_ma_200: position_multiplier *= 0.5`
  4. **Target Adjustment**: Predict market-neutral returns (excess vs SPX)
  
- [ ] **Validation**:
  - Target SPX correlation < 0.30
  - Verify performance improvement in declining markets
  - Ensure no score degradation in bull markets

### 2.2 Risk Management
- [ ] **Volatility Ratio Control**:
  - Monitor rolling volatility ratio (20-day, 60-day windows)
  - If ratio > 1.20x: reduce position multiplier
  - Target: volatility ratio <= 1.15x in all regimes
  
- [ ] **Drawdown Controls**:
  - Maximum drawdown limit: 20%
  - Position reduction during drawdowns (>10%: reduce by 50%)
  
- [ ] **Stress Testing**:
  - Backtest on crisis periods (2008, 2020, 2022)
  - Check performance in high-volatility regimes

---

## Phase 3: Production & Final Validation (Week 3) - P2

### 3.1 Code Optimization
- [ ] **Runtime**: Ensure < 8 hours (feature caching, efficient data loading)
- [ ] **Memory**: Handle feature sets efficiently (use float32, sparse where possible)
- [ ] **Reproducibility**: Set all random seeds, document dependencies

### 3.2 Comprehensive Validation
- [ ] **Walk-forward (expanding window)**: Train on 1-2 years, test on next 6 months
- [ ] **Walk-forward (rolling window)**: Fixed 2-year window, roll forward monthly
- [ ] **Multiple start dates**: Test robustness to training period choice

### 3.3 Final Checks
- [ ] **Data Leakage Audit**: Final verification
- [ ] **Feature Importance**: Document top features
- [ ] **Parameter Sensitivity**: Which parameters matter most?
- [ ] **Correlation Analysis**: SPX correlation, feature correlations

---

## Phase 4: Optional - Model Diversity (Only If Needed) - P3

**ONLY if single model hits a ceiling (<2.2 score).**

### 4.1 Minimal Ensemble
- [ ] **Test adding ONE secondary model**: XGBoost (from ht-final-v1 params)
- [ ] **Simple weighted average**: 70% LightGBM, 30% XGBoost
- [ ] **Validation**: Only proceed if ensemble > single model AND adds robustness

### 4.2 Alternative Targets
- [ ] **Classification approach**: Predict direction with confidence (from htmp-optuna)
- [ ] **Hybrid target**: Combine regression + classification signals

**Decision Gate**: If ensemble doesn't beat single model by >0.05 score, **abandon**. Complexity must earn its place.

---

## Priority Order (Strict)

| Priority | Phase | Task | Expected Impact |
|----------|-------|------|-----------------|
| **P0.0** | 0 | Fix data leakage (dropna, 180-day split) | **+0.6 score** |
| **P0** | 0 | Replicate 163rd place (single LGBM + tanh) | **~2.1 score** |
| **P1** | 1 | Optimize single model (features, hyperparams) | **+0.05-0.1 score** |
| **P1.5** | 2 | Decorrelation + risk management | **Robustness** |
| **P2** | 3 | Production readiness | **Reliability** |
| **P3** | 4 | Model diversity (ONLY if needed) | **Optional** |

**Rule**: Do not start P1 until P0 achieves 2.1+ score. Do not start P1.5 until P1 is complete. Do not start P3 unless single model hits a hard ceiling.

---

## Immediate Action Plan

### Day 0: Data Leakage Fix (P0.0)
1. **Audit**: Search all code for `bfill`, `ffill`, `interpolate`
2. **Replace**: All imputation with `.dropna()`
3. **Implement**: 180-day validation split
4. **Verify**: No forward-looking data in any feature

### Days 1-3: 163rd Place Replication (P0)
1. Implement exact data pipeline (no imputation, drop missing)
2. Train single LightGBM regressor on `market_forward_excess_returns`
3. Create multi-timeframe features (short/medium/quarterly)
4. Add lag features (1-5 days) for key predictors
5. Add autoregressive target lags (1-3 days)
6. Implement exact tanh function (bound=0.006, scale=3.0)
7. Add adaptive volatility multiplier (20-day trailing)
8. **Target**: Achieve 2.1+ validation score

### Days 4-7: Single Model Optimization (P1)
1. Run SHAP analysis, identify top 20-30 features
2. Test feature ablations (remove categories, measure impact)
3. Tune LightGBM hyperparameters (Optuna)
4. Tune post-processing parameters (tanh bound/scale, volatility multiplier)
5. Implement walk-forward validation
6. **Target**: Push score to 2.15-2.20+, verify consistency

### Week 2: Address Weaknesses (P1.5)
1. Measure SPX correlation of predictions
2. Test decorrelation techniques (residual approach first)
3. Add regime filter (SPX MA crossover)
4. Implement volatility ratio controls
5. Stress test on crisis periods
6. **Target**: SPX correlation < 0.30, volatility ratio <= 1.15x

### Week 3: Production (P2)
1. Optimize runtime (< 8 hours)
2. Final walk-forward validation
3. Document everything
4. **Target**: Production-ready code

---

## Success Metrics & Targets

### Primary Metrics
| Metric | Current | Bronze | 163rd | Target |
|--------|---------|--------|-------|--------|
| Kaggle Score | ~1.05 | 1.958 | 2.16 | **> 2.1** |
| Validation Score | ~1.05 | - | - | **> 2.1** |
| Sharpe Ratio | 0.896 | - | 2.16 | **> 1.80** |
| Information Coefficient | +0.0491 | - | - | **> 0.15** |

### Secondary Metrics
| Metric | Target | Status |
|--------|--------|--------|
| Volatility Ratio | **<= 1.15x** | Must pass |
| Max Drawdown | **< 20%** | Must pass |
| SPX Correlation | **< 0.30** | Must pass |
| Generalization Delta | **< -0.1** | Must pass |
| Runtime | **< 8 hours** | Must pass |

### Validation Metrics
- Walk-forward consistency: Similar performance across all windows
- Robustness: Small variance across parameter sweeps
- Crisis performance: No catastrophic failure in 2008/2020/2022 backtests

---

## Resource Requirements

### Data
- `train.csv`, `test.csv`
- Feature definitions documentation

### Compute
- CPU: Multi-core for parallel hyperparameter tuning
- Memory: 16-32GB (for ~100-200 features max)
- GPU: Optional for LightGBM (not required)

### Libraries
- Core: numpy, pandas, scikit-learn
- Models: lightgbm, xgboost (optional)
- Optimization: optuna
- Analysis: shap, matplotlib
- Utilities: joblib

---

## Risk Management

### Technical Risks
1. **Data Leakage (CRITICAL)**: bfill/imputation causes -0.6 score
   - **Mitigation**: `.dropna()` only, audit all features, exclude last 180 days
2. **Look-ahead Bias**: Future data in features
   - **Mitigation**: Strict chronological splits, verify all rolling windows
3. **Overfitting**: Model memorizes training data
   - **Mitigation**: Walk-forward validation, time-series splits, robustness checks
4. **Runtime Exceeded**: Code too slow
   - **Mitigation**: Feature caching, efficient data types, parallel processing

### Model Risks
1. **SPX Correlation**: Model tied to market trend
   - **Mitigation**: Decorrelation techniques, regime filters (P1.5)
2. **Regime Change**: Model fails in new market conditions
   - **Mitigation**: Walk-forward validation, stress testing on crisis periods
3. **Feature Decay**: Predictive power diminishes
   - **Mitigation**: Feature importance analysis, regular retraining

---

## Deliverables

1. **Data Audit Report**: All leakage fixes, missing value handling
2. **Replication Notebook**: 163rd place exact implementation
3. **Optimization Notebook**: Single model tuning and results
4. **Decorrelation Report**: SPX correlation analysis and fixes
5. **Final Model Code**: Production-ready single LightGBM solution
6. **Validation Results**: Walk-forward, OOS, stress test performance

---

## Key Insights (Reference)

### From 163rd Place (Score: 2.16) ⭐
- **Data Handling**: `dropna()` > `bfill` (+0.6 score)
- **Validation**: Exclude last 180 days (test set leakage)
- **tanh**: bound=0.006, scale=3.0 (exact implementation)
- **Robustness**: 2.19 → 2.16 over 6 months (no overfitting)
- **Weakness**: Correlated with SPX (underperformed in declines)

### From Bronze Medal (Score: 1.958)
- **Simplicity**: Single LightGBM > complex ensembles
- **Target**: `market_forward_excess_returns`
- **Features**: Multi-timeframe rolling stats, lags, autoregressive
- **Post-Processing**: Scaled tanh + adaptive volatility multiplier
- **Philosophy**: Quality over quantity

---

*Plan updated: 2026-06-28*
*Status: **SIMPLIFIED - Focus on single model replication and optimization**
*Key Insight: **163rd place proved simple > complex. Replicate first, optimize second, add complexity only if necessary.**
*Expected Progression: Current (~1.05) → After leakage fix (~1.65) → After 163rd replication (~2.1+) → After optimization (>2.1)*
