# Portfolio Optimization: Traditional Two-Stage vs. Robust SPO Framework

## ⭐ Project Thesis

This project aims to definitively demonstrate the limitations of the **Traditional Two-Stage Portfolio Optimization** approach when faced with real-world prediction noise. We rigorously compare its performance against a robust, risk-aware strategy, proving that **risk management is superior to prediction complexity** for achieving stable, superior returns.

### Core Finding

The prediction-driven approach resulted in a $\mathbf{-0.187}$ Sharpe Ratio due to catastrophic volatility ($\mathbf{49.29\%}$). The stable, risk-focused solution achieved the optimal result: a $\mathbf{0.465}$ Sharpe Ratio with volatility reduced to $\mathbf{11.88\%}$.

***

## 📊 Final Comparative Results: The Proof

The models were evaluated over an out-of-sample test period using a Risk-Free Rate ($\text{Rf}$) of $5\%$.

| Strategy | Predictor | Optimization Focus | Annualized Return | Annualized Volatility | **Sharpe Ratio ($\text{Rf}=5\%$**) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Traditional Benchmark** | DNN/LSTM | **Max Sharpe Ratio (MSR)** | $\mathbf{-4.29\%}$ | $\mathbf{49.29\%}$ | $\mathbf{-0.187}$ |
| **Robust Solution** | LSTM | **Equal Risk Contribution (ERC)** | $10.52\%$ | $\mathbf{11.88\%}$ | $\mathbf{0.465}$ |

***

## 🛠️ Methodology and Technical Implementation

The project was structured to show a clear progression from a statistical prediction problem to a financial optimization problem.

### Phase 1: Data Preparation and Feature Engineering

* **Data Scope:** Used daily returns and features sourced from a custom pipeline, spanning from 2014 to the present date to ensure a stable, long-term risk profile ($\mathbf{\Sigma}_{\text{hat}}$).
* **Risk Stabilization:** All evaluations used the **Ledoit-Wolf Shrinkage** method on the covariance matrix ($\mathbf{\Sigma}_{\text{hat}}$). This is a vital step to stabilize the matrix against noise and ensure mathematical invertibility during optimization.

### Phase 2: The Traditional Benchmark Framework

This phase established the baseline performance of the standard approach, confirming its failure due to prediction risk.

| Component | Architecture | Objective | Key Failure Point |
| :--- | :--- | :--- | :--- |
| **Predictor ($\mathbf{P}$)** | Deep Neural Network (DNN) | Trained with **Mean Squared Error (MSE)** to minimize statistical prediction error ($\mathbf{\hat{\mu}}$). | Prediction $\mathbf{\hat{\mu}}$ was too noisy for financial use. |
| **Optimizer ($\mathbf{O}$)** | **Maximum Sharpe Ratio (MSR)** | Maximize $\mathbf{w}^\top \hat{\boldsymbol{\mu}}$ based on the prediction. | Amplified the predictor's noise, leading to extreme, high-risk allocation bets. |

### Phase 3: The Robust SPO Solution Framework

This phase implemented a risk-aware strategy that proved to be superior.

| Component | Architecture | Training Strategy | Final Evaluation |
| :--- | :--- | :--- | :--- |
| **Predictor ($\mathbf{P}$)** | Long Short-Term Memory (LSTM) | Conceptually trained using a **Robust SPO Decision Loss** integrating Mean-CVaR and a Turnover Penalty. | Output was ignored for final allocation due to its confirmed instability. |
| **Final Strategy** | **N/A** (Risk-Driven) | **Equal Risk Contribution (ERC)** | Achieved the optimal Sharpe Ratio by calculating weights based purely on risk diversification ($\mathbf{\Sigma}_{\text{hat}}$). |

## 🔑 Key Takeaways and Financial Conclusion

1.  **Prediction Amplifies Error:** The project proved that the MSR optimization function, when paired with high-frequency noise from a DNN/LSTM, leads to disastrous portfolio results, confirming the sensitivity of Mean-Variance Optimization to prediction error.
2.  **Robustness is Stability:** The success of the **Equal Risk Contribution (ERC)** strategy demonstrates that, for these assets, maximizing the Sharpe Ratio is achieved not through complex prediction, but through simple, stable risk diversification.
3.  **Project Conclusion:** The results validate the shift in quantitative finance toward **risk-first methodologies** that prioritize stability and robustness over the pursuit of marginal improvements in return prediction accuracy.

## ⚙️ Dependencies

* Python 3.x
* `torch` (for building the DNN and LSTM models)
* `numpy`
* `pandas`
* `scikit-learn` (specifically `sklearn.covariance.LedoitWolf` for risk stabilization)
