Algorithms Unravelled
=====================

This section describes the concrete implementation of node splitting for the WIS and CRPS loss functions.

Regression trees are constructed using a greedy procedure: at each current leaf, if the stopping criteria are not met, the algorithm must identify the best split of the data to generate two child nodes. To do this, one typically evaluates the information gain for all candidate features $j$ and, for each feature, all potential split points $s_{j}$.

This process can become computationally prohibitive when the loss function is expensive to evaluate, which is often the case for naïve implementations of certain loss functions such as the sum of pinball losses or the CRPS. Fortunately, as we demonstrate in our paper, it is possible to efficiently find the optimal split for these two loss functions by leveraging carefully chosen data structures: min-max heaps for the sum of pinball losses and a Fenwick tree for the CRPS.

Below, we illustrate the algorithm on concrete examples. For a fixed feature $j$, we detail the full update of the corresponding data structure used to compute the information gains across all possible splits, demonstrating how the algorithm achieves efficiency in practice.

To perform these iterations efficiently, 



1. Sum of Pinball Losses (Quantile Regression)
----------------------------------------------

When building PMQRTs, the split criterion minimizes the sum of pinball losses in order to estimate specific quantiles.

Mechanism
~~~~~~~~~

To maintain :math:`O(N \log N)` efficiency, the algorithm avoids recomputing the loss from scratch at every candidate split. Instead, it relies on min-max heaps to update the required statistics incrementally.

Procedural steps
~~~~~~~~~~~~~~~~

Refer to the diagram below for the step-by-step update rules of the weighted sums of residuals on a concrete example.

We consider the sum of pinball losses with quantile levels $0.3$ and $0.7$.  We assume that we need to identify the best split for a given feature $j$ for which $x_{1,j} \\leq \\dots \\leq x_{7,j}$ with corresponding observations $\\textbf y= [0,1,2,3,-1,-2,-3]$.


We comment step 6. At step 6, we need to insert $y_6 = -2$ into the min-max heaps.  

Heap $\\mathcal{H}_{1}$ must contain $\\lceil \\tau_1 \\times 6 \\rceil = 2$ elements and it already contains two elements. Since $-2$ is smaller than the current maximum element of $\\mathcal{H}_1$ (namely $0$), we pop out $0$ and insert $-2$ into $\\mathcal{H}_1$.  

Next, we need to insert $0$ into one of the remaining heaps $\\mathcal{H}_2, \\mathcal{H}_3$. Heap $\\mathcal{H}_2$ must contain  
$$
\\lceil \\tau_{2} \\times 6 \\rceil - \\lceil \\tau_1 \\times 6 \\rceil = 5 - 2 = 3
$$  
elements, and it currently contains 2 elements. Since $0$ is smaller than the current maximum value in $\\mathcal{H}_2$ (namely $2$), we insert $0$ into $\\mathcal{H}_2$ and the update is completed.

.. raw:: html

   <iframe src="_static/heaps.pdf" width="100%" height="600px" style="border: none;"></iframe>

2. Continuous Ranked Probability Score (CRPS)
---------------------------------------------

The Continuous Ranked Probability Score (CRPS) is a standard scoring rule for probabilistic forecasting.

Using CRPS as a split criterion encourages child nodes that are both sharp and well-calibrated.

Mechanism
~~~~~~~~~

Unlike the mean squared error (MSE), which only depends on the conditional mean, the CRPS evaluates the entire predictive distribution of the target variable $y$.



Procedural steps
~~~~~~~~~~~~~~~~

Refer to the diagram below for the step-by-step update rules for the CRPS loss on a concrete example.

We assume that we need to identify the best split for a given feature $j$ for which $x_{1,j} \\leq \\dots \\leq x_{6,j}$ with corresponding observations $\\textbf y= [2,1,3,-1,-3,-2]$.   

.. raw:: html

   <iframe src="_static/fenwick.pdf" width="100%" height="600px" style="border: none;"></iframe>
