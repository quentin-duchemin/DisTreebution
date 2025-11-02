Presentation of the project
---------------------------



Challenge
~~~~~~~~~


In this work, we address the problem of distributional (probabilistic) regression for scalar outcomes. Rather than predicting only the mean of a response variable, our goal is to learn its entire conditional distribution, enabling forecasts with calibrated uncertainty. We focus on tree-based models and design training algorithms that optimize proper probabilistic scoring rules—specifically, the Weighted Interval Score (WIS) and the Continuous Ranked Probability Score (CRPS). Our motivation is to combine the nonparametric flexibility and interpretability of regression trees with principled probabilistic objectives that promote accurate calibration and coverage.

Our contributions
~~~~~~~~~~~~~~~~~
Two novel tree algorithms for probabilistic forecasting:

PMQRT (Pinball Multi-Quantile Regression Tree) — a tree that simultaneously predicts multiple quantiles by minimizing a generalized entropy derived from the WIS (a sum of pinball losses across quantile levels).

CRPS-RT (CRPS Regression Tree) — a tree that directly minimizes the CRPS, a strictly proper scoring rule for full-distribution predictions.

Ensemble extensions: We extend both approaches to forests—PMQRF and CRPS-RF—which aggregate multiple probabilistic trees to enhance smoothness and predictive performance.

Efficient implementation
~~~~~~~~~~~~~~~~~~~~~~~~

Training trees under WIS or CRPS losses can be computationally demanding. To address this, we developed efficient procedures for split selection based on specialized data structures such as min–max heaps, weight-balanced binary trees, and Fenwick (binary indexed) trees. These allow us to maintain and update order statistics incrementally during tree construction.

For both the CRPS and WIS losses, we further propose a leave-one-out unbiased estimator of the associated information gains, which provides an efficient and statistically sound criterion for splitting without increasing computational complexity.

Theoretical framework
~~~~~~~~~~~~~~~~~~~~~

We formalize our splitting criteria as the minimization of a generalized entropy (Bayes risk) associated with the underlying scoring rule (WIS or CRPS). This perspective unifies impurity-based decision tree learning with proper probabilistic scoring rules and yields trees whose splits are shared across all quantiles—improving interpretability and ensuring distributional coherence.

Empirical evaluation
~~~~~~~~~~~~~~~~~~~~

Through a series of experiments, we show that our methods are competitive with state-of-the-art probabilistic forecasting models. The proposed trees and forests provide well-calibrated predictive distributions while retaining the interpretability of tree-based models. We also demonstrate that our framework integrates naturally with conformal prediction techniques, enabling models that achieve calibrated uncertainty and group-conditional coverage guarantees.

When to use our methods

Our approach is particularly suited for applications that require:

- Nonparametric, interpretable probabilistic forecasts;

- Direct optimization of proper scoring rules (WIS or CRPS);

- Scalable training on tabular data where tree-based models are advantageous.


Limitations and outlook
~~~~~~~~~~~~~~~~~~~~~~~

Our current methods focus on univariate outputs, and while the proposed optimizations make distributional trees practical, they remain less expressive than large neural generative models for complex conditional densities. Nevertheless, our framework provides a strong balance of interpretability, calibration, and computational efficiency.

Further reading

For algorithmic details, refer to Sections 2–2.4 of the paper (PMQRT, CRPS-RT, and the leave-one-out estimator). Implementation specifics and pseudocode are provided in Appendix, and benchmark results are discussed in Section 3. The full paper and accompanying code are available on arXiv under Efficient distributional regression trees learning algorithms for calibrated non-parametric probabilistic forecasts (Duchemin & Obozinski, 2025).
