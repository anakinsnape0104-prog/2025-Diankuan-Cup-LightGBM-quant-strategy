# HS300 LightGBM Multi-Factor Strategy

A machine learning-based quantitative stock selection strategy for CSI 300 constituents, combining price-volume factors, LightGBM return forecasting, dynamic risk control, and robust fallback mechanisms.

---

## Project Overview

This project aims to construct a systematic equity strategy that simultaneously:

* Generates stable excess returns
* Controls drawdowns actively
* Reduces market beta exposure

The strategy uses a rolling LightGBM model to predict future 20-day stock returns and selects the top-ranked CSI 300 constituents every 5 trading days.

---

## Strategy Framework

```text
HS300 Daily Data
        │
        ▼
Price-Volume Factors
(Momentum, Reversal, Volatility, Liquidity)
        │
        ▼
LightGBM Regression
Predict Future 20-Day Return
        │
        ▼
Score Fusion
85% LightGBM + 15% Momentum
        │
        ▼
Top 25 Stock Selection
(Equal Weight)
        │
        ▼
Market Trend Filter
(MA60 / MA120 + Momentum20)
        │
        ▼
Dynamic Position Sizing
(45% ~ 100%)
        │
        ▼
Portfolio Rebalancing
(Every 5 Trading Days)
```

---

## Factor Design

Seven price-volume factors are constructed:

| Category   | Factor |
| ---------- | ------ |
| Momentum   | mom20  |
| Momentum   | mom60  |
| Momentum   | mom120 |
| Reversal   | rev5   |
| Volatility | vol20  |
| Volatility | vol60  |
| Liquidity  | liq20  |

### Economic Intuition

* Momentum captures trend persistence.
* Reversal captures short-term overreaction.
* Volatility measures risk conditions.
* Liquidity reflects trading activity changes.

---

## Machine Learning Model

### LightGBM Regression

Target Variable:

```python
Future 20-Day Return
```

Training Method:

* Rolling panel training
* Approximately 380 trading days
* 220 boosting trees
* Daily retraining during rebalancing

Key Advantages:

* Captures nonlinear relationships
* Handles feature interactions automatically
* Fast training speed
* Suitable for large cross-sectional datasets

---

## Double-Safeguard Mechanism

### Layer 1: Hybrid Scoring

Under normal conditions:

```python
Score =
0.85 × LightGBM Prediction
+
0.15 × 60-Day Momentum
```

This combines:

* AI-driven return forecasting
* Traditional momentum robustness

### Layer 2: Fallback Mode

If any of the following occurs:

* Insufficient training samples
* Training failure
* Prediction collapse
* Data abnormalities

The strategy automatically switches to:

```python
Score = Momentum60
```

This ensures continuous operation even when the machine learning model becomes unreliable.

---

## Risk Management

A synthetic market index is constructed using the average price of CSI 300 constituents.

Market regime is determined by:

* MA60
* MA120
* 20-Day Momentum

Position multiplier:

| Market State                    | Exposure |
| ------------------------------- | -------- |
| Above MA60                      | 100%     |
| Between MA60 and MA120          | 85%      |
| Below MA120                     | 65%      |
| Below MA120 + Negative Momentum | 45%      |

This mechanism significantly reduces drawdowns during market stress.

---

## Portfolio Construction

### Rebalancing

* Frequency: Every 5 trading days
* Universe: CSI 300 constituents
* Holdings: Top 25 stocks
* Weighting: Equal-weighted

### Trading Logic

1. Rank all stocks by score.
2. Select top 25.
3. Allocate capital equally.
4. Adjust total exposure according to market regime.
5. Rebalance portfolio.

---

## Backtest Summary

Sample Period:

```text
2022 – 2025
```

Key Results:

| Metric              | Strategy  |
| ------------------- | --------- |
| Annualized Return   | ~23%      |
| Maximum Drawdown    | ~20%      |
| Information Ratio   | ~1.1      |
| Calmar Ratio        | ~1.08     |
| Rebalance Frequency | 5 Days    |
| Holdings            | 25 Stocks |

The strategy consistently outperformed the CSI 300 benchmark while maintaining controlled drawdowns.

---

## Project Structure

```text
.
├── strategy.py
├── README.md
├── factor_engine.py
├── risk_control.py
├── backtest_results/
├── figures/
└── report/
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* LightGBM
* ATRader
* Quantitative Backtesting Framework

---

## Future Improvements

* Fundamental factors integration
* SHAP factor interpretation
* XGBoost / CatBoost ensemble
* Regime-switching models
* Transformer-based return forecasting
* Full A-share universe expansion

---

## Author

Haoming Li

Financial Engineering, Guangdong University of Foreign Studies

Research Interests:
Quantitative Investment · Machine Learning · Asset Pricing · ESG · Time Series Analysis
