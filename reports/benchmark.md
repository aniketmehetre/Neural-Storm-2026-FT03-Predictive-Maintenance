# NASA CMAPSS Benchmark Evaluation Report

**Generated:** 2026-11-05 13:00:09 UTC  
**Evaluation Protocol:** Official test split evaluated at the final observed cycle of each engine against ground-truth remaining life.

## 1. Summary of Benchmark Results

| Dataset | Operating Conditions | Fault Modes | Engines | RMSE (cycles) | MAE (cycles) | R² Score | NASA Asymmetric Score | Status |
|---|---|---|---|---|---|---|---|---|
| **FD001** | Sea Level (1) | HPC Degradation (1) | 100 | **16.74** | **12.33** | **0.8255** | **655.02** | Verified Official |
| **FD002** | 6 Regimes | HPC Degradation (1) | 259 | — | — | — | — | Raw data not packaged |
| **FD003** | Sea Level (1) | HPC + Fan Degradation (2) | 100 | — | — | — | — | Raw data not packaged |
| **FD004** | 6 Regimes | HPC + Fan Degradation (2) | 248 | — | — | — | — | Raw data not packaged |

## 2. FD001 Detailed Diagnostics

- **Root Mean Squared Error (RMSE):** 16.74 cycles
- **Mean Absolute Error (MAE):** 12.33 cycles
- **Median Absolute Error:** 9.67 cycles
- **NASA Asymmetric Penalty Score:** 655.02
- **Prediction Bias:** 40 early predictions vs 60 late predictions (Mean bias: 3.58 cycles).
- **Precision (Within ±2 cycles):** 16 / 100 engines.

*Note: NASA asymmetric scoring heavily penalizes late estimates ($d > 0$) where maintenance would be scheduled after actual component failure.*
