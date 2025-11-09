![Project Logo](docs/source/_static/logo.png)


# DisTreebution


**DisTreebution** fits **distributional regression trees and forests** to generate **calibrated probabilistic forecasts**.

It optimizes trees using:  
- A sum of **Pinball losses** for multiple-quantile regression  
- **CRPS (Continuous Ranked Probability Score)** for full distributional regression  


The package also includes implementations of split conformal prediction methods built upon these forests of distributional trees. Two main conformal approaches are provided:
- [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222), where nested sets are of the form $$\left( [q_{\beta}-t , q_{1-\beta} +t] \right)_t$$ for some nominal level $\beta$.
  
- [Distributional Conformal Prediction](https://arxiv.org/abs/1909.07889), where nested sets are of the form $$\left( [q_{t} , q_{1-t}] \right)_t$$.

  
**Paper:** [https://arxiv.org/abs/2502.05157](https://arxiv.org/abs/2502.05157)

