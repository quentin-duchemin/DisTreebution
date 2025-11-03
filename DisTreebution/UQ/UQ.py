
import numpy as np
import random
from ..QRT.RegressionTreeQuantile import RegressionTreeQuantile
from ..RT.RegressionTreeQuadratic import RegressionTreeQuadratic
from tqdm import tqdm
import copy
from .utils import filter_dict

from ..CRPSRT.RegressionTree import RegressionTree

from .Conformalisation_CQR import Conformalisation_CQR

from .Conformalisation_distributional import Conformalisation_distributional
from .Conformalisation import Conformalisation


available_settings = {}

# Possible choices of regression trees 
available_settings['type_tree'] = ['RT','PMQRT','CRPS']

# Possible choices of nested sets
available_settings['nested_set'] = ['CQR', 'CQR-m', 'CQR-r', 'distributional']

# Possible choices of conformal framework
available_settings['type_aggregation_trees'] = ['vr', 'vr-avg']
available_settings['type_conformal'] = [None, 'split']

# Currently, only symmetric uncertainty intervals are implemented
# available_settings['symmetry'] = ['symmetric']

# Possibility to use group coverage
available_settings['group_coverage'] = [False, True]

# Hyperparameters to train the regression trees
all_params = ['nTrees', 'treeID2quantiles_train', 'max_depth', 'use_LOO', 'min_samples_split', 'max_depth_group', 'list_distri_low_quantiles']


class UQ():
    """
    Uncertainty quantification and conformal prediction helper.

    This class wraps training of forest-like quantile/regression trees,
    conformalization, and construction/analysis of predictive sets.

    It centralizes configuration (:attr:`settings`), builds a non-conformal
    quantile estimator (:attr:`noconf`) for querying, selects a conformal
    backend (:attr:`conf`) based on settings, and exposes utilities to train
    trees, conformalize them, predict conformal sets, and extract quantile
    estimates.

    **Constructor Parameters**
    
    :param str type_tree: Type of tree learner used.
        Examples:
        
        - ``'PMQRT'`` – per-tree quantile regression
        - ``'CRPS'`` – CRPS-optimized regression
        - ``'RT'`` – quadratic regression tree
        
        Other strings may be supported by external tree classes.

    :param str nested_set: Strategy for nested prediction sets (e.g. ``'CQR'``, ``'distri'``).
        Used to choose the conformal backend.
        - If it contains ``'CQR'`` → :class:`Conformalisation_CQR` backend.
        - If it contains ``'distri'`` → :class:`Conformalisation_distributional` backend.

    :param str or None type_conformal: Controls conformalization mode.
        If ``None``, a generic :class:`Conformalisation` backend is instantiated;
        otherwise ``nested_set`` determines which backend is used.

    :param bool group_coverage: Whether to use group-aware conformal methods and
        prediction routines (calls to ``conf.*_group_coverage``).

    :param str type_aggregation_trees: How per-tree quantiles are aggregated across trees.
        Examples:
        - ``'vr'`` – quantile bagging: average quantiles across trees
        - ``'vr-avg'`` – distributional bagging: aggregates conditional CDFs before querying

    :param dict params: Miscellaneous hyperparameters, may include:
        - ``'nTrees'`` – number of trees to train
        - ``'treeID2quantiles_train'`` – mapping tree IDs → quantiles (for PMQRT)
        - ``'max_depth'`` – maximum depth for trees
        - ``'min_samples_split'`` – minimum samples to split
        - ``'use_LOO'`` – whether to use leave-one-out info gains
        - ``'max_depth_group'`` – max depth for group coverage trees
        - ``'list_distri_low_quantiles'`` – lower quantiles for distributional conformal prediction

    **Behavior / Side Effects**
    
    - ``self.settings`` is a dict of configuration options.
    - ``self.params`` stores the params dictionary.
    - ``self.noconf`` is always constructed as a :class:`Conformalisation_CQR` instance
      with the same settings except ``nested_set='CQR'`` (used for querying quantiles
      without conformalization).
    - ``self.conf`` is chosen based on ``type_conformal`` and ``nested_set``.

    **Public Attributes**

    :ivar dict settings: Configuration options described above.
    :ivar dict params: Hyperparameters.
    :ivar noconf: A :class:`Conformalisation_CQR` instance for non-conformal queries.
    :ivar conf: Selected conformalization backend instance
        (:class:`Conformalisation_CQR`, :class:`Conformalisation`,
        or :class:`Conformalisation_distributional`).

    **Example:**
    
    .. code-block:: python

        uq = UQ(type_tree='CRPS', nested_set='CQR', params={'nTrees': 100, 'max_depth': 6})
        trees, sample2calib = uq.train_trees(X_train, y_train)
        uq.conformalize(trees, X_calib, y_calib, alpha=0.1)
        sample2predset = uq.predict_conformal_set(trees, X_test)
        widths, coverages = uq.compute_width_coverage(sample2predset, y_test)
    """
    def __init__(self, type_tree='PQRT', nested_set='CQR', type_conformal='split', group_coverage=False, type_aggregation_trees='vr', params=None):
        self.settings = {}
        self.settings['type_tree'] = type_tree
        self.settings['nested_set'] = nested_set
        self.settings['type_conformal'] = type_conformal
        self.settings['group_coverage'] = group_coverage
        self.settings['type_aggregation_trees'] = type_aggregation_trees
        self.params = params
        
        # used to query quantiles without conformalization
        settings_noconf = copy.deepcopy(self.settings)
        settings_noconf['nested_set'] = 'CQR'
        self.noconf = Conformalisation_CQR(settings=settings_noconf, params=params)
        if type_conformal is None:
            self.conf = Conformalisation(settings=self.settings, params=params) 
        elif 'CQR' in nested_set:
            self.conf = Conformalisation_CQR(settings=self.settings, params=params)
        elif 'distri' in nested_set:
            self.conf = Conformalisation_distributional(settings=self.settings, params=params)
        else:
            assert False, "No conformalisation method found for the given nested set"

            
    def compute_width_coverage(self, sample2predset, y_test):
        """
        Computes the total width of prediction sets and coverage indicators for each test sample.

        For each test sample, this method calculates:

        - The sum of the widths of all predicted intervals (sub-intervals) associated with the sample.
        - Whether the true value falls within any of the predicted intervals (coverage).

        :param list[tuple] sample2predset: 
            A list where each element corresponds to a test sample and contains a tuple
            of the form ``(lower_bound, upper_bound)``.

        :param array-like y_test: 
            The true target values for the test samples.

        :returns: 
            - **widths** (*numpy.ndarray*) – Total widths of the predicted intervals for each test sample.
            - **coverages** (*numpy.ndarray*) – Binary indicators (1 if the true value is covered by any interval,
            0 otherwise) for each test sample.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        widths, coverages = np.zeros(len(y_test)), np.zeros(len(y_test))
        for j,yj in enumerate(y_test):
            MC = False
            width = 0
            low, up = sample2predset[j]
            width += up-low
            MC = MC or ((yj >= low) and (yj <= up))
            widths[j] = width
            coverages[j] = 1.* MC
        return widths, coverages

    def compute_group_width_coverage(self, sample2predset, y_test, testID2group):
        """
        Computes the average prediction interval width and empirical coverage per group.

        Groups are defined by the ``testID2group`` mapping.

        :param list[list[tuple]] sample2predset:
            A list where each element corresponds to a test sample and contains one or more tuples
            of the form ``(lower_bound, upper_bound)`` representing predicted intervals.

        :param array-like y_test:
            The true target values for the test samples.

        :param dict testID2group:
            A mapping from test sample indices to group identifiers.

        :returns:
            - **group2width** (*dict*) – Mapping ``group_id → average interval width`` across all samples in the group.  
            - **group2cov** (*dict*) – Mapping ``group_id → empirical coverage``, i.e., the fraction of samples in the group
            whose true value lies within any of the predicted intervals.
        :rtype: tuple[dict, dict]
        """
        all_groups = np.unique(list(testID2group.values()))
        group2size = {group:0 for group in all_groups}
        group2cov = {group:0 for group in all_groups}
        group2width = {group:0 for group in all_groups}
        for i in range(len(y_test)):
            group2size[testID2group[i]] += 1
            group2cov[testID2group[i]] += 1.*( (y_test[i]>=sample2predset[i][0]) and (y_test[i]<=sample2predset[i][1]) )
            group2width[testID2group[i]] += abs(sample2predset[i][1]-sample2predset[i][0])
        for group, cov in group2cov.items():
            group2cov[group] = cov/group2size[group]
            group2width[group] = group2width[group]/group2size[group]
        return group2width, group2cov

    def train_trees(self, x_train, y_train, max_depth_ref_tree=-1):
        """
        Trains an ensemble of regression trees and tracks calibration indices for Out-of-Bag (OOB)
        conformal methods.

        :param numpy.ndarray x_train:
            Training feature matrix of shape ``(n_train, n_features)``.

        :param numpy.ndarray y_train:
            Training target values of shape ``(n_train,)``.

        :param int max_depth_ref_tree:
            If not equal to ``-1``, a reference tree is constructed and fitted
            (used by certain tree-training routines). If ``-1``, no reference tree is used.

        :returns:
            - **trees** (*list*) – List of fitted tree objects, ordered by training loop.  
            - **sample2calib_trees** (*dict*) – Mapping ``sample_index → list of tree indices`` for which
            the sample was placed in the calibration set (i.e., excluded from that tree’s training set).
            Useful for split-conformal calibration.
        :rtype: tuple[list, dict]
        """
        nTrees = self.params['nTrees']
        trees = []
        ntrain = x_train.shape[0]
        sample2calib_trees = {i:[] for i in range(len(y_train))}
        indexes = np.arange(0,ntrain,1)

        if max_depth_ref_tree == -1:
            ref_tree = None
        else:
            if self.settings['type_tree']=='PMQRT':
                vals = list(self.conf.treeID2quantiles_train.values())
                assert len(set(map(str, vals))) == 1, "All trees should be trained using the same set of quantiles if one want to use group conformal prediction"
                ref_tree = RegressionTreeQuantile(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], quantiles=self.conf.treeID2quantiles_train[0], use_LOO=self.params.get('use_LOO', True))
            elif self.settings['type_tree']=='CRPS':
                ref_tree = RegressionTree(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], limit_use_CRPS=self.params.get('limit_use_CRPS', None), quantiles=self.conf.treeID2quantiles_train.get(0, None), use_LOO=self.params.get('use_LOO', True))
            elif self.settings['type_tree']=='RT':
                ref_tree = RegressionTreeQuadratic(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], use_LOO=self.params.get('use_LOO', True))
            ref_tree.fit(x_train, y_train)
        for l in tqdm(range(nTrees)):
            random.shuffle(indexes)
            ntrain_tree = int(0.6*ntrain)
            if nTrees == 1:
                ntrain_tree = ntrain
            index_train, index_calib = indexes[:ntrain_tree], indexes[ntrain_tree:]

            if self.settings['type_tree']=='PMQRT':
                tree = RegressionTreeQuantile(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], quantiles=self.conf.treeID2quantiles_train[l], use_LOO=self.params.get('use_LOO', True))
            elif self.settings['type_tree']=='CRPS':
                tree = RegressionTree(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], limit_use_CRPS=self.params.get('limit_use_CRPS', None), quantiles=self.conf.treeID2quantiles_train.get(l, None), use_LOO=self.params.get('use_LOO', True))
            elif self.settings['type_tree']=='RT':
                tree = RegressionTreeQuadratic(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], use_LOO=self.params.get('use_LOO', True))
            tree.fit(x_train[index_train,:], y_train[index_train], ref_tree=ref_tree, max_depth_ref_tree=max_depth_ref_tree)
            trees.append(tree)
            for icalib in index_calib:
                sample2calib_trees[icalib].append(l)
        return trees, sample2calib_trees
    
    def conformalize(self, trees, x_calib, y_calib, alpha, **kwargs):
        """
        Runs conformalization (split-conformal) on the provided trees and calibration data.

        :param list trees:
            List of fitted tree objects (the same trees returned by :meth:`train_trees`).

        :param numpy.ndarray x_calib:
            Calibration feature matrix.

        :param numpy.ndarray y_calib:
            Calibration target values.

        :param float alpha:
            Miscoverage level (e.g., ``0.1`` for 90% coverage).

        :param kwargs:
            Additional keyword arguments forwarded to the conformal backend.
            For CQR nested sets, the ``nominal_quantiles`` parameter should be provided.

        **Behavior:**

        - If ``self.settings['group_coverage']`` is ``True``, calls
        ``self.conf.conformalize_split_group_coverage(...)``.
        - Otherwise, calls ``self.conf.conformalize_split(...)``.
        - The conformal backend is expected to update its internal state used by
        :meth:`predict_conformal_set`.
        """

        if self.settings['group_coverage']:
            self.conf.conformalize_split_group_coverage(trees, x_calib, y_calib, alpha, **kwargs)
        else:
            self.conf.conformalize_split(trees, x_calib, y_calib, alpha, **kwargs)
                
    def predict_conformal_set(self, trees, x_test, return_testID2group=False, **kwargs):
        """
        Predict conformalized sets for test points using the conformal backend.

        :param list trees:
            Trained trees (same as passed to :meth:`conformalize`).

        :param numpy.ndarray x_test:
            Test feature matrix of shape (n_test, n_features).

        :param bool return_testID2group:
            When ``group_coverage`` is ``True``, optionally return the mapping from
            test indices to groups alongside the prediction sets.

        :param kwargs:
            Additional keyword arguments forwarded to the conformal backend's prediction routine.

        :returns:
            - If ``group_coverage`` is ``False``:
            
            **sample2predset** – Sequence-like (length n_test) of (low, up) prediction interval pairs.
            
            - If ``group_coverage`` is ``True``:
            
            - If ``return_testID2group`` is ``False``:
                
                **sample2predset** as above.
            
            - If ``return_testID2group`` is ``True``:
                
                Tuple ``(testID2group, sample2predset)`` where ``testID2group`` is the mapping used.

        **Notes:**

        - Delegates to either :meth:`predict_conformal_set_split` or
        :meth:`predict_conformal_set_split_group_coverage` on ``self.conf``.
        """
        if self.settings['group_coverage']:
            testID2group, sample2predset = self.conf.predict_conformal_set_split_group_coverage(trees, x_test, **kwargs)
            if return_testID2group:
                return testID2group, sample2predset
        else:
            sample2predset = self.conf.predict_conformal_set_split(trees, x_test, **kwargs)
        return sample2predset
    
    def get_quantile_estimate(self, trees, x_test, quantiles):
        """
        Query aggregate quantile estimates for each test sample from the (non- or pre-) conformal tree predictions.

        :param list trees:
            Trained tree objects.

        :param numpy.ndarray x_test:
            Test features matrix of shape (n_test, n_features).

        :param iterable quantiles:
            List or array of quantile levels to estimate (values in [0, 1]).

        :returns:
            **sample2quantiles** (*dict*) – Mapping from sample index to list of estimated quantile values.
            The inner list has the same ordering as the input `quantiles` argument.

        **Behavior / Implementation details:**

        - Calls ``self.conf.preprocess_trees(trees, x_test)`` to obtain a structure
        ``treeID2testID2values`` mapping tree IDs to test sample leaf/value indexes.

        - For each requested quantile ``q``, finds the nearest available query quantile
        in ``self.conf.quantiles_query`` (via argmin on absolute distance) and obtains
        the set of tree IDs associated with that query from
        ``self.conf.quantile_query2treeIDs``.
        """
        treeID2testID2values = self.conf.preprocess_trees(trees, x_test)
        sample2quantiles = {i:[] for i in range(x_test.shape[0])}
        for i_q, q in enumerate(quantiles):
            i_q_query = np.argmin(np.abs(q-self.conf.quantiles_query))
            IDs_trees = self.conf.quantile_query2treeIDs[self.conf.quantiles_query[i_q_query]]
            for i in range((x_test.shape[0])):
                if 'vr-avg' in self.settings['type_aggregation_trees']:
                    def f(quant, treeID2sampleID2values):
                        qhat_test = 0
                        weights_test = []
                        ls_leaves_test = []
                        for k in treeID2sampleID2values.keys():
                            ls_new = list(treeID2sampleID2values[k][i])
                            ls_leaves_test = ls_leaves_test + ls_new
                            weights_test = weights_test + list(np.ones(len(ls_new))/len(ls_new))

                        ls_leaves_test_unique, indices_test = np.unique(ls_leaves_test, return_inverse=True)
                        summed_weights_test = np.bincount(indices_test, weights_test)
                        order_unique_test = np.argsort(ls_leaves_test_unique)
                        ls_leaves_test_unique = ls_leaves_test_unique[order_unique_test]
                        summed_weights_test = summed_weights_test[order_unique_test]
                        qhat_test = self.conf.operation_leaf(ls_leaves_test_unique, quant, weights=summed_weights_test)
                        return qhat_test
                    
                elif 'vr' in self.settings['type_aggregation_trees']:
                    def f(quant, treeID2sampleID2values):
                        qhat_test = 0
                        count_test = 0
                        for k in treeID2sampleID2values.keys():
                            ls_leaves_test = list(treeID2sampleID2values[k][i])
                            qhat_test += self.conf.operation_leaf(ls_leaves_test, quant)
                            count_test += 1 
                        avg_test = qhat_test / count_test if count_test != 0 else 0
                        return avg_test
                    
                qhat_test = f(q, filter_dict(treeID2testID2values, IDs_trees))
                sample2quantiles[i].append(qhat_test)
        return sample2quantiles
