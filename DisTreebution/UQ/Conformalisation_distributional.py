import numpy as np
import random
from tqdm import tqdm
from .Conformalisation import Conformalisation
from .utils import filter_dict



class Conformalisation_distributional(Conformalisation):
    """Conformalisation for distributional tree-based models.

    This class implements conformalisation utilities for tree ensembles that
    produce distributional (quantile/interval) predictions. It extends
    :class:`Conformalisation` and provides methods to compute conformity
    thresholds on a calibration set and to produce conformal prediction sets
    for test inputs.

    :param settings: Configuration settings for the conformalisation process, such as the type of tree.
    :type settings: dict, optional
    :param params: Hyperparameters for the conformalisation process.
    :type params: dict, optional
    """
    def __init__(self, settings=None, params=None):
        """Initialize the conformalisation instance.

        :param settings: See class description.
        :type settings: dict, optional
        :param params: See class description.
        :type params: dict, optional
        """
        super().__init__(settings=settings, params=params)

    def conformalize_split(self, trees, x_calib, y_calib, alpha):
        """Compute conformity threshold from a calibration set.

        The method computes conformity scores on the provided calibration
        dataset and sets :attr:`self.conf_thresh` to the empirical quantile
        corresponding to ``alpha``.

        :param trees: List of trained tree models.
        :type trees: list
        :param x_calib: Calibration input data.
        :type x_calib: numpy.ndarray
        :param y_calib: Calibration output data.
        :type y_calib: numpy.ndarray
        :param alpha: Significance level for conformalization (e.g. 0.1 for 90% sets).
        :type alpha: float
        :returns: None. Sets ``self.conf_thresh`` on success.
        :rtype: None
        """
        sample2calib_trees = {j: [i for i in range(len(trees))] for j in range(len(y_calib))}

        treeID2calibID2values = self.preprocess_trees(trees, x_calib)
        _, _, conf_scores = self.get_low_up_score(0, y_calib, sample2calib_trees, treeID2calibID2values, {})
        conf_scores = np.sort(conf_scores)
        self.conf_thresh = conf_scores[int(alpha*len(conf_scores))]


    def predict_conformal_set_split(self, trees, x_test):
        """Predict conformal prediction sets for test inputs using a fixed threshold.

        :param trees: List of trained tree models.
        :type trees: list
        :param x_test: Test input data.
        :type x_test: numpy.ndarray
        :returns: Mapping from test sample index to conformal prediction set [low, up].
        :rtype: dict
        :raises AssertionError: If ``self.conf_thresh`` has not been set by a previous calibration.
        """
        assert self.conf_thresh is not None
        sample2predset = {}
        treeID2testID2values = self.preprocess_trees(trees, x_test)
        sample2calib_trees = {0: [i for i in range(self.params['nTrees'])]}
        for i in range(x_test.shape[0]):
            low, up, _ = self.get_low_up_score(i, [0], sample2calib_trees, {}, treeID2testID2values, t_fixed=self.conf_thresh)
            sample2predset[i] = [low[0],up[0]]
            #coverage += 1.*( (y_test[i]>=low[0]) and (y_test[i]<=up[0]) ) / len(y_test)
        return sample2predset
    
    def conformalize_split_group_coverage(self, trees, x_calib, y_calib, alpha, max_depth_group=None, get_res_on_calib=False):
        """Compute group-specific conformity thresholds from a calibration set.

        For each group (as defined by the tree partitioning up to
        ``max_depth_group``) this method computes an empirical conformity
        threshold and stores the results in ``self.group2conf_thresh``.

        :param trees: List of trained tree models.
        :type trees: list
        :param x_calib: Calibration input data.
        :type x_calib: numpy.ndarray
        :param y_calib: Calibration output data.
        :type y_calib: numpy.ndarray
        :param alpha: Significance level for conformalization.
        :type alpha: float
        :param max_depth_group: Maximum depth for defining groups (tree levels).
        :type max_depth_group: int, optional
        :param get_res_on_calib: If True, also return calibration set statistics.
        :type get_res_on_calib: bool, optional
        :returns: None. On success sets ``self.group2conf_thresh``. If
            ``get_res_on_calib`` is True the method additionally computes and
            would return calibration group sizes and coverages (the caller
            should call the corresponding predict method to retrieve them).
        :rtype: None
        """
        sample2calib_trees = {j: [i for i in range(len(trees))] for j in range(len(y_calib))}
        self.max_depth_group = max_depth_group
        self.alpha = alpha
        treeID2calibID2values, treeID2calibID2group = self.preprocess_trees_with_groups(trees, x_calib, max_depth_group)
        all_groups = list(set([treeID2calibID2group[k][i] for k in range(self.params['nTrees']) for i in range(len(y_calib))]))
        self.group2conf_thresh = {group:[] for group in all_groups}
        group2calibID2 = {group:[] for group in all_groups}
        for j in range(len(y_calib)):
            try:
                group2calibID2[treeID2calibID2group[0][j]].append(j)
            except:
                pass
        for group in all_groups:
            group2calibID2[group] = np.array(group2calibID2[group]).astype(int)
        for group in all_groups:
            temp_treeID2calibID2values = {k: {count: treeID2calibID2values[k][i] for count, i in enumerate(group2calibID2[group])} for k in range(self.params['nTrees'])}
            _, _, conf_scores = self.get_low_up_score(0, y_calib[group2calibID2[group]], sample2calib_trees, temp_treeID2calibID2values, {})
            conf_scores = np.sort(conf_scores)
            self.group2conf_thresh[group] = conf_scores[int(alpha*len(conf_scores))]

    def predict_conformal_set_split_group_coverage(self, trees, x_test, get_res_on_calib=False, x_calib=None, y_calib=None):
        """Predict conformal sets for test inputs with group-wise thresholds.

        Uses ``self.group2conf_thresh`` (computed by
        :meth:`conformalize_split_group_coverage`) to produce conformal sets for
        each test sample, adapting the threshold to the group that a sample
        belongs to.

        :param trees: List of trained tree models.
        :type trees: list
        :param x_test: Test input data.
        :type x_test: numpy.ndarray
        :param get_res_on_calib: If True, also compute and return calibration-set statistics.
        :type get_res_on_calib: bool, optional
        :param x_calib: Calibration input data (required when ``get_res_on_calib`` is True).
        :type x_calib: numpy.ndarray, optional
        :param y_calib: Calibration output data (required when ``get_res_on_calib`` is True).
        :type y_calib: numpy.ndarray, optional
        :returns: When ``get_res_on_calib`` is True returns a tuple
            ``(group2sizecalib, group2covcalib, treeID2testID2group, sample2predset)``.
            Otherwise returns ``(treeID2testID2group[0], sample2predset)``.
        :rtype: tuple
        :raises AssertionError: If ``self.group2conf_thresh`` has not been computed.
        """
        assert self.group2conf_thresh is not None
        sample2calib_trees = {0: [i for i in range(len(trees))]}
        sample2predset = {}
        treeID2testID2values, treeID2testID2group = self.preprocess_trees_with_groups(trees, x_test, self.max_depth_group)
        all_groups_test = list(set([treeID2testID2group[k][i] for k in range(self.params['nTrees']) for i in range(x_test.shape[0])]))
        for i in tqdm(range(x_test.shape[0])):
            # we can keep any tree since they all share the same group structure (for the first levels)
            t = self.group2conf_thresh[treeID2testID2group[0][i]]
            low, up, conf_scores = self.get_low_up_score(i, [None], sample2calib_trees, {}, treeID2testID2values, t_fixed=t)
            sample2predset[i] = [low[0],up[0]]
            
        if get_res_on_calib:
            treeID2calibID2values, treeID2calibID2group = self.preprocess_trees_with_groups(trees, x_calib, self.max_depth_group)

            group2sizecalib = {group:0 for group in all_groups_test}
            group2covcalib = {group:0 for group in all_groups_test}
            sample2predsetcalib = {}
            for i in tqdm(range(len(y_calib))):
                t = self.group2conf_thresh[treeID2calibID2group[0][i]]
                low, up, _ = self.get_low_up_score(i, y_calib, sample2calib_trees, treeID2calibID2values, treeID2calibID2values, t_fixed=t)
                sample2predsetcalib[i] = [[low[0],up[0]]]
                group2sizecalib[treeID2calibID2group[0][i]] += 1
                group2covcalib[treeID2calibID2group[0][i]] += 1.*( (y_calib[i]>=low[0]) and (y_calib[i]<=up[0]) )
            for group, cov in group2covcalib.items():
                group2covcalib[group] = cov/group2sizecalib[group]
            return group2sizecalib, group2covcalib, treeID2testID2group, sample2predset
        else:
            return treeID2testID2group[0], sample2predset
        
          
    def get_low_up_test_i(self, yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=None):
        """Select lower/upper bounds for a test sample using conformity thresholds.

        The method inspects arrays of calibration quantile estimates and their
        corresponding test quantile estimates and either uses the provided
        threshold index ``t`` or searches for the largest index where the
        calibration value covers ``yj``.

        :param yj: Calibration sample true value.
        :type yj: float
        :param qhat_low: Lower quantile estimates from calibration.
        :type qhat_low: numpy.ndarray
        :param qhat_up: Upper quantile estimates from calibration.
        :type qhat_up: numpy.ndarray
        :param qhat_low_test: Lower quantile estimates for the test sample.
        :type qhat_low_test: numpy.ndarray
        :param qhat_up_test: Upper quantile estimates for the test sample.
        :type qhat_up_test: numpy.ndarray
        :param t: Optional fixed threshold index to use. If ``None`` the method searches.
        :type t: int or None, optional
        :returns: ``(low_j, up_j, t_idx)`` where ``low_j`` and ``up_j`` are the
            selected bounds for the test sample and ``t_idx`` is the chosen threshold index.
        :rtype: tuple(float, float, int)
        """
        t_idx = t
        low_quantiles = self.params['list_distri_low_quantiles']
        if t_idx is None:
            t_idx = len(low_quantiles)-1
            condition = (qhat_low[t_idx]<=yj) and (yj<=qhat_up[t_idx])
            while not(condition) and (t_idx>0):
                t_idx = t_idx - 1
                condition = (qhat_low[t_idx]<=yj) and (yj<=qhat_up[t_idx])
        return qhat_low_test[t_idx], qhat_up_test[t_idx], t_idx

    def get_low_up_score(self, i, y_train, sample2calib_trees, treeID2trainID2values, treeID2testID2values, t_fixed=None):
        """Compute lower/upper bounds and conformity scores for one test index.

        This method aggregates quantile estimates across trees (according to the
        aggregation configured in ``self.settings['type_aggregation_trees']``)
        and returns, for the test sample at index ``i``, three lists:
        the lower bounds, the upper bounds and the conformity score indices.

        :param i: Index of the test sample.
        :type i: int
        :param y_train: Array of calibration/training output values used to compute scores.
        :type y_train: numpy.ndarray or list
        :param sample2calib_trees: Mapping from sample indices to lists of tree IDs used for calibration.
        :type sample2calib_trees: dict
        :param treeID2trainID2values: Mapping from tree ID to a mapping of training sample IDs to leaf values.
        :type treeID2trainID2values: dict
        :param treeID2testID2values: Mapping from tree ID to a mapping of test sample IDs to leaf values.
        :type treeID2testID2values: dict
        :param t_fixed: Optional fixed threshold/index to use when selecting bounds. If ``None`` the method will search.
        :type t_fixed: int or float, optional
        :returns: A tuple ``(lower, upper, conf_scores)`` where each entry is a list
            with one element per calibration sample in ``y_train``.
        :rtype: tuple(list, list, list)
        """
        # i: index of the test sample
        low_quantiles = self.params['list_distri_low_quantiles']
        lower = []
        upper = []
        conf_scores = []
        for j, yj in enumerate(y_train):
            if 'vr-avg' in self.settings['type_aggregation_trees']:
                def f(quant, j, treeID2trainID2values, treeID2testID2values):
                    ls_leaves_y = []
                    weights = []
                    weights_test = []
                    ls_leaves_test = []
                    qhat, qhat_test = 0, 0
                    if treeID2trainID2values != {}:
                        for k in treeID2trainID2values.keys():
                            ls_new = list(treeID2trainID2values[k][j])
                            ls_leaves_y = ls_leaves_y + ls_new
                            weights = weights + list(np.ones(len(ls_new))/len(ls_new))
                    if treeID2testID2values != {}:
                        for k in treeID2testID2values.keys():
                            ls_new = list(treeID2testID2values[k][i])
                            ls_leaves_test = ls_leaves_test + ls_new
                            weights_test = weights_test + list(np.ones(len(ls_new))/len(ls_new))

                    if treeID2trainID2values != {}:
                        ls_leaves_y_unique, indices = np.unique(ls_leaves_y, return_inverse=True)
                        summed_weights = np.bincount(indices, weights)
                        order_unique = np.argsort(ls_leaves_y_unique)
                        ls_leaves_y_unique = ls_leaves_y_unique[order_unique]
                        summed_weights = summed_weights[order_unique]
                        qhat = self.operation_leaf(ls_leaves_y_unique, quant, weights=summed_weights)

                    if treeID2testID2values != {}:
                        ls_leaves_test_unique, indices = np.unique(ls_leaves_test, return_inverse=True)
                        summed_weights_test = np.bincount(indices, weights_test)
                        order_unique_test = np.argsort(ls_leaves_test_unique)
                        ls_leaves_test_unique = ls_leaves_test_unique[order_unique_test]
                        summed_weights_test = summed_weights_test[order_unique_test]
                        qhat_test = self.operation_leaf(ls_leaves_test_unique, quant, weights=summed_weights_test)
                    
                    return qhat, qhat_test
                
                qhat_low = np.zeros(len(low_quantiles))
                qhat_up = np.zeros(len(low_quantiles))
                qhat_low_test = np.zeros(len(low_quantiles))
                qhat_up_test = np.zeros(len(low_quantiles))
                for i_q, q in enumerate(low_quantiles):
                    i_q_low = np.argmin(np.abs(q-self.quantiles_query))
                    i_q_up = np.argmin(np.abs(1-q-self.quantiles_query))
                    IDs_low = self.quantile_query2treeIDs[self.quantiles_query[i_q_low]]
                    IDs_up = self.quantile_query2treeIDs[self.quantiles_query[i_q_up]]
                    qhat_low[i_q], qhat_low_test[i_q] = f(q, j, filter_dict(treeID2trainID2values, IDs_low), filter_dict(treeID2testID2values, IDs_low))
                    qhat_up[i_q], qhat_up_test[i_q] = f(1-q, j, filter_dict(treeID2trainID2values, IDs_up), filter_dict(treeID2testID2values, IDs_up))    

                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)


            elif 'vr' in self.settings['type_aggregation_trees']:
                def f(quant, j, treeID2trainID2values, treeID2testID2values):
                    qhat, qhat_test = 0, 0
                    count_intern = 0
                    count_intern_test = 0
                    if treeID2trainID2values != {}:
                        for k in treeID2trainID2values.keys():
                            ls_leaves_y = list(treeID2trainID2values[k][j])
                            ls_leaves_y = ls_leaves_y
                            qhat += self.operation_leaf(ls_leaves_y, quant)
                            count_intern += 1 
                    if treeID2testID2values != {}:
                        for k in treeID2testID2values.keys():
                            ls_leaves_test = list(treeID2testID2values[k][i])
                            ls_leaves_test = ls_leaves_test
                            qhat_test += self.operation_leaf(ls_leaves_test, quant)
                            count_intern_test += 1 
                    return qhat/max(1,count_intern), qhat_test/max(1,count_intern_test)
                
                qhat_low = np.zeros(len(low_quantiles))
                qhat_up = np.zeros(len(low_quantiles))
                qhat_low_test = np.zeros(len(low_quantiles))
                qhat_up_test = np.zeros(len(low_quantiles))
                for i_q, q in enumerate(low_quantiles):
                    i_q_low = np.argmin(np.abs(q-self.quantiles_query))
                    i_q_up = np.argmin(np.abs(1-q-self.quantiles_query))
                    IDs_low = self.quantile_query2treeIDs[self.quantiles_query[i_q_low]]
                    IDs_up = self.quantile_query2treeIDs[self.quantiles_query[i_q_up]]
                    qhat_low[i_q], qhat_low_test[i_q] = f(q, j, filter_dict(treeID2trainID2values, IDs_low), filter_dict(treeID2testID2values, IDs_low))
                    qhat_up[i_q], qhat_up_test[i_q] = f(1-q, j, filter_dict(treeID2trainID2values, IDs_up), filter_dict(treeID2testID2values, IDs_up))
                
                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)
        
        return lower, upper, conf_scores