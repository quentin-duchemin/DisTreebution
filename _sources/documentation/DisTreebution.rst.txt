DisTreebution
=============

Overview
--------

DisTreebution is a Python toolkit for decision-tree based uncertainty quantification and conformal prediction. It bundles several tree learners (quantile, quadratic and CRPS-optimized trees) together with conformalization backends (CQR variants and a
distributional approach) to produce prediction sets with finite-sample coverage guarantees.

This documentation page summarizes the purpose of the main modules and gives quick links to installation and example usage.

Package layout (top-level modules)
----------------------------------

- Regression trees: tree learners in `QRT/` implement quantile or quadratic objectives.
- CRPS trees: `CRPRT/` implements tree splitting based on CRPS-related entropies.
- Conformalization backends: `UQ/Conformalisation_CQR.py` (CQR variants) and
  `UQ/Conformalisation_distributional.py` (distributional nested-sets) provide the
  calibration and prediction-set machinery used by the top-level `UQ` orchestrator.


Installation
------------

See the installation page (with clone and virtualenv instructions):

.. toctree::
	:hidden:

	../installation

Quick start (conceptual)
------------------------

The common workflow is:

1. Train an ensemble of trees with `UQ.train_trees`.
2. Conformalize calibration data with `UQ.conformalize` (pass `nominal_quantiles` for CQR).
3. Predict conformal sets on test inputs with `UQ.predict_conformal_set`.
4. Evaluate widths and coverage with `UQ.compute_width_coverage` or `UQ.compute_group_width_coverage`.

See the simple runnable example in `package_docs/examples/quick_start.py` for a short smoke-test.

API and developer notes
-----------------------

- `UQ.UQ` — main entry-point. Use `type_tree` to choose tree type (`PQRT`, `PMQRT`, `RT`, `CRPS`) and
  `nested_set` to choose conformal backend (`CQR`, `distributional`). Pass a `params` dict with keys such as
  `nTrees`, `max_depth`, `min_samples_split`, and for distributional methods `list_distri_low_quantiles`.
- `Conformalisation_CQR` — offers `conformalize_split`, `predict_conformal_set_split`, and group-aware variants.
- `Conformalisation_distributional` — distributional procedure supporting a grid of low quantiles and group coverage.
- `QRT.RegressionTreeQuantile` and `QRT.RegressionTreeQuadratic` — tree learners exposing `fit`, `predict`,
  and `get_values_leaf[_and_groups]` helpers used by conformalization preprocessors.


Contact
-------

If you need help or want to contribute, open an issue describing the change you propose.

