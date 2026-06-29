
# Why we chose Ridge
```
======================================================================
 MODEL COMPARISON SUMMARY
======================================================================

Model                       Sharpe   Return %    Vol %       IC  Alpha %
----------------------------------------------------------------------
Ridge                        0.896      18.22    20.35  +0.0491    +2.65
BayesianRidge                0.893      17.36    19.44  +0.0488    +1.78
Lasso                        0.884      15.59    17.63  -0.0000    +0.02
Stacking Ensemble            0.884      15.59    17.64  +0.0085    +0.02
ElasticNet                   0.879      15.60    17.75  +0.0229    +0.02
Weighted Average Ensemble    0.853      15.96    18.70  +0.0196    +0.38
RandomForest                 0.807      14.72    18.23  -0.0093    -0.86
GradientBoosting             0.735      14.24    19.36  -0.0051    -1.34
----------------------------------------------------------------------
```

- include the technics we read from other people.
- include some info about which features we chose and why (feature_engineering.py)
- Our solution and score (phase0_final_improved.py)