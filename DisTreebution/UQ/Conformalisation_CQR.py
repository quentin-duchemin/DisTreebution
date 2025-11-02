import numpy as np
import random
from tqdm import tqdm
from .Conformalisation import Conformalisation
from .utils import filter_dict

class Conformalisation_CQR(Conformalisation):
    """
    A class for handling conformalised quantile regression (CQR) processes on tree-based models.
    """
    def __init__(self, settings=None, params=None):
        super().__init__(settings=settings, params=params)


    def conformalize_split(self, trees, x_calib, y_calib, alpha, nominal_quantiles=None):
        """
        Conformalizes the quantile regression predictions using the CQR method with specified nominal quantiles.
        Parameters
        ----------
        trees : list
            List of trained tree models.
        x_calib : np.ndarray
            Calibration input data.
        y_calib : np.ndarray
            Calibration output data.
        alpha : float
            Significance level for conformalization.
        nominal_quantiles : list or float, optional
            List of nominal quantiles to conformalize. If a single float is provided, it is converted to a list.
        """
        assert nominal_quantiles is not None, "Please provide nominal_quantiles for CQR conformalization."
        if type(nominal_quantiles)!=list:
            nominal_quantiles = [nominal_quantiles]
        self.nominal_quantiles = nominal_quantiles
        self.alpha = alpha
        sample2calib_trees = {j: [i for i in range(self.params['nTrees'])] for j in range(len(y_calib))}
        treeID2calibID2values = self.preprocess_trees(trees, x_calib)
        self.conf_thresh = []
        for i_q, q in enumerate(nominal_quantiles):
            i_q_low = np.argmin(np.abs(q-self.quantiles_query))
            i_q_up = np.argmin(np.abs(1-q-self.quantiles_query))
            i_q_median = np.argmin(np.abs(0.5-self.quantiles_query))
            IDs_low = self.quantile_query2treeIDs[self.quantiles_query[i_q_low]]
            IDs_up = self.quantile_query2treeIDs[self.quantiles_query[i_q_up]]
            IDs_median = self.quantile_query2treeIDs[self.quantiles_query[i_q_median]]
            _, _, conf_scores = self.get_low_up_score(0, q, y_calib, sample2calib_trees, filter_dict(treeID2calibID2values, IDs_low), {}, filter_dict(treeID2calibID2values, IDs_up), {}, filter_dict(treeID2calibID2values, IDs_median), {})
            conf_scores = np.sort(conf_scores)
            ncal = len(conf_scores)
            self.conf_thresh.append(conf_scores[int(((1-alpha)+1/ncal)*ncal)])
    
    def predict_conformal_set_split(self, trees, x_test):
        """
        Predicts conformalized quantile regression sets for test data using the CQR method.
        Parameters
        ----------
        trees : list
            List of trained tree models.
        x_test : np.ndarray
            Test input data.
        Returns
        -------
        sample2predset : dict
            Dictionary mapping nominal quantile indices to predicted conformal sets for each test sample.
        """
        assert self.conf_thresh is not None
        treeID2testID2values = self.preprocess_trees(trees, x_test)
        sample2predset = {i_q:{} for i_q in range(len(self.nominal_quantiles))}
        sample2calib_trees = {0: [i for i in range(self.params['nTrees'])]}
        for i_q, q in enumerate(self.nominal_quantiles):
            i_q_low = np.argmin(np.abs(q-self.quantiles_query))
            i_q_up = np.argmin(np.abs(1-q-self.quantiles_query))
            i_q_median = np.argmin(np.abs(0.5-self.quantiles_query))
            IDs_low = self.quantile_query2treeIDs[self.quantiles_query[i_q_low]]
            IDs_up = self.quantile_query2treeIDs[self.quantiles_query[i_q_up]]
            IDs_median = self.quantile_query2treeIDs[self.quantiles_query[i_q_median]]
            for i in range((x_test.shape[0])):
                low, up, _ = self.get_low_up_score(i, q, [0], sample2calib_trees, {}, filter_dict(treeID2testID2values, IDs_low), {}, filter_dict(treeID2testID2values, IDs_up), {}, filter_dict(treeID2testID2values, IDs_median), t_fixed=self.conf_thresh[i_q])
                sample2predset[i_q][i] = [low[0],up[0]]
        return sample2predset
    
    
    def conformalize_split_group_coverage(self, trees, x_calib, y_calib, alpha, nominal_quantiles=None, max_depth_group=None):
        """
        Conformalizes the quantile regression predictions using the CQR method with specified nominal quantiles,
        ensuring group coverage up to a specified maximum group depth.
        Parameters
        ----------
        trees : list
            List of trained tree models.   
        x_calib : np.ndarray
            Calibration input data.
        y_calib : np.ndarray
            Calibration output data.
        alpha : float
            Significance level for conformalization.
        nominal_quantiles : list or float, optional
            List of nominal quantiles to conformalize. If a single float is provided, it is converted to a list.
        max_depth_group : int
            Maximum depth for group coverage trees.
        """
        assert nominal_quantiles is not None, "Please provide nominal_quantiles for CQR conformalization."
        assert max_depth_group is not None, "Please provide max_depth_group for group coverage."
        self.nominal_quantiles = nominal_quantiles
        sample2calib_trees = {j: [i for i in range(len(trees))] for j in range(len(y_calib))}
        self.max_depth_group = max_depth_group
        self.alpha = alpha
        treeID2calibID2values, treeID2calibID2group = self.preprocess_trees_with_groups(trees, x_calib, max_depth_group)
        #all_groups_test = list(set([treeID2testID2group[k][i] for k in range(self.params['nTrees']) for i in range(len(y_test))]))
        all_groups = list(set([treeID2calibID2group[k][i] for k in range(self.params['nTrees']) for i in range(x_calib.shape[0])]))
        self.group2conf_thresh = {i_q: {group:[] for group in all_groups} for i_q in range(len(self.nominal_quantiles))}
        group2calibID2 = {group:[] for group in all_groups}
        for j in range(len(y_calib)):
            try:
                group2calibID2[treeID2calibID2group[0][j]].append(j)
            except:
                pass
        for group in all_groups:
            group2calibID2[group] = np.array(group2calibID2[group]).astype(int)

        for i_q, q in enumerate(self.nominal_quantiles):
            i_q_low = np.argmin(np.abs(q-self.quantiles_query))
            i_q_up = np.argmin(np.abs(1-q-self.quantiles_query))
            i_q_median = np.argmin(np.abs(0.5-self.quantiles_query))
            IDs_low = self.quantile_query2treeIDs[self.quantiles_query[i_q_low]]
            IDs_up = self.quantile_query2treeIDs[self.quantiles_query[i_q_up]]
            IDs_median = self.quantile_query2treeIDs[self.quantiles_query[i_q_median]]
            for group in all_groups:
                temp_treeID2calibID2values = {k: {count: treeID2calibID2values[k][i] for count, i in enumerate(group2calibID2[group])} for k in range(self.params['nTrees'])}
                _, _, conf_scores = self.get_low_up_score(0, q, y_calib[group2calibID2[group]], sample2calib_trees, filter_dict(temp_treeID2calibID2values, IDs_low), {}, filter_dict(temp_treeID2calibID2values, IDs_up), {}, filter_dict(temp_treeID2calibID2values, IDs_median), {})
                conf_scores = np.sort(conf_scores)
                ncal = len(conf_scores)
                self.group2conf_thresh[i_q][group] = conf_scores[int((1-alpha+1/ncal)*ncal)]

    def predict_conformal_set_split_group_coverage(self, trees, x_test):
        """
        Predicts conformalized quantile regression sets for test data using the CQR method with group coverage.
        Parameters
        ----------
        trees : list
            List of trained tree models.
        x_test : np.ndarray
            Test input data.
        Returns
        -------
        sample2predset : dict
            Dictionary mapping nominal quantile indices to predicted conformal sets for each test sample.
        """
        sample2predset = {i_q:{} for i_q in range(len(self.nominal_quantiles))}
        treeID2testID2values, treeID2testID2group = self.preprocess_trees_with_groups(trees, x_test, self.max_depth_group)
        sample2calib_trees = {0: [i for i in range(self.params['nTrees'])]}
        for i_q, q in enumerate(self.nominal_quantiles):
            i_q_low = np.argmin(np.abs(q-self.quantiles_query))
            i_q_up = np.argmin(np.abs(1-q-self.quantiles_query))
            i_q_median = np.argmin(np.abs(0.5-self.quantiles_query))
            IDs_low = self.quantile_query2treeIDs[self.quantiles_query[i_q_low]]
            IDs_up = self.quantile_query2treeIDs[self.quantiles_query[i_q_up]]
            IDs_median = self.quantile_query2treeIDs[self.quantiles_query[i_q_median]]
            for i in tqdm(range(x_test.shape[0])):
                # we can keep any tree since they all share the same group structure (for the first levels)
                group_i = treeID2testID2group[0][i]
                low, up, _ = self.get_low_up_score(i, q, [0], sample2calib_trees, {}, filter_dict(treeID2testID2values, IDs_low), {}, filter_dict(treeID2testID2values, IDs_up), {}, filter_dict(treeID2testID2values, IDs_median), t_fixed=self.group2conf_thresh[i_q][group_i])
                sample2predset[i_q][i] = [low[0],up[0]]
        return treeID2testID2group[0], sample2predset

    def get_low_up_test_i(self, yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qmedian, qmedian_test, t=None):
        """
        Computes the conformalized lower and upper quantile predictions for a single test sample.
        Parameters
        ----------
        yj : float
            Calibration output value.
        qhat_low : float
            Predicted lower quantile from calibration data.
        qhat_up : float
            Predicted upper quantile from calibration data.
        qhat_low_test : float
            Predicted lower quantile for the test sample.
        qhat_up_test : float
            Predicted upper quantile for the test sample.
        qmedian : float
            Predicted median quantile from calibration data.
        qmedian_test : float
            Predicted median quantile for the test sample.
        t : float, optional
            Conformalization threshold. If None, it will be computed.
        Returns
        -------
        low_test : float
            Conformalized lower quantile prediction for the test sample.
        up_test : float
            Conformalized upper quantile prediction for the test sample.
        t : float
            Conformalization threshold used.
        """
        if self.settings['nested_set']=='CQR':
            if t is None:
                t = -(qhat_up-qhat_low)/2
                condition = (qhat_low-t<=yj) and (yj<=qhat_up+t)
                while not(condition):
                    t += 0.02
                    condition = (qhat_low-t<=yj) and (yj<=qhat_up+t)
            return qhat_low_test-t, qhat_up_test+t, t
        
        elif self.settings['nested_set']=='CQR-m':
            if t is None:
                t = -(qhat_up-qhat_low)/(2*qmedian+qhat_up-qhat_low)
                condition = (qhat_low*(1+t)-t*qmedian<=yj) and (yj<=qhat_up*(1+t)+t*qmedian)
                while not(condition):
                    t += 0.02
                    condition = (qhat_low*(1+t)-t*qmedian<=yj) and (yj<=qhat_up*(1+t)+t*qmedian)
            return qhat_low_test*(1+t)-t*qmedian_test, qhat_up_test*(1+t)+t*qmedian_test, t

        elif self.settings['nested_set']=='CQR-r':
            gap = (qhat_up-qhat_low)
            if t is None:
                t = -1/2
                condition = (qhat_low-t*gap<=yj) and (yj<=qhat_up+t*gap)
                while not(condition):
                    t += 0.02
                    condition = (qhat_low-t*gap<=yj) and (yj<=qhat_up+t*gap)
            gap = (qhat_up_test-qhat_low_test)
            return qhat_low_test-t*gap, qhat_up_test+t*gap, t

    def get_low_up_score(self, i, q, y_train, sample2calib_trees, treeID2trainID2values_low, treeID2testID2values_low, treeID2trainID2values_up, treeID2testID2values_up, treeID2trainID2values_median, treeID2testID2values_median, t_fixed=None):
        """
        Computes the conformalized lower and upper quantile predictions and conformity scores for a test sample.
        Parameters
        ----------
        i : int
            Index of the test sample.
        q : float
            Nominal quantile level.
        y_train : list
            List of calibration output values.
        sample2calib_trees : dict
            Dictionary mapping calibration sample indices to lists of tree IDs used for calibration.
        treeID2trainID2values_low : dict
            Dictionary mapping tree IDs to calibration sample IDs and their corresponding lower quantile leaf values.
        treeID2testID2values_low : dict
            Dictionary mapping tree IDs to test sample IDs and their corresponding lower quantile leaf values.
        treeID2trainID2values_up : dict
            Dictionary mapping tree IDs to calibration sample IDs and their corresponding upper quantile leaf values.
        treeID2testID2values_up : dict
            Dictionary mapping tree IDs to test sample IDs and their corresponding upper quantile leaf values.
        treeID2trainID2values_median : dict
            Dictionary mapping tree IDs to calibration sample IDs and their corresponding median quantile leaf values.
        treeID2testID2values_median : dict
            Dictionary mapping tree IDs to test sample IDs and their corresponding median quantile leaf values.
        t_fixed : float, optional
            Fixed conformalization threshold. If None, it will be computed.
        Returns
        -------
        lower : list
            List of conformalized lower quantile predictions for the test sample.
        upper : list
            List of conformalized upper quantile predictions for the test sample.
        conf_scores : list
            List of conformity scores for the calibration samples.
        """
        # i: index of the test sample
        lower = []
        upper = []
        conf_scores = []
        for j, yj in enumerate(y_train):                       
            if 'vr-avg' in self.settings['type_aggregation_trees']:
                def f(quant, j, treeID2trainID2values, treeID2testID2values):
                    qhat, qhat_test = 0, 0
                    ls_leaves_y = []
                    weights = []
                    weights_test = []
                    ls_leaves_test = []
                    for k in sample2calib_trees[j]:
                        if treeID2trainID2values!={} and k in treeID2trainID2values.keys():
                            ls_new = list(treeID2trainID2values[k][j])
                            ls_leaves_y = ls_leaves_y + ls_new
                            weights = weights + list(np.ones(len(ls_new))/len(ls_new))
                        if treeID2testID2values != {} and k in treeID2testID2values.keys():
                            ls_new = list(treeID2testID2values[k][i])
                            ls_leaves_test = ls_leaves_test + ls_new
                            weights_test = weights_test + list(np.ones(len(ls_new))/len(ls_new))
                    if treeID2trainID2values!={}:
                        ls_leaves_unique, indices = np.unique(ls_leaves_y, return_inverse=True)
                        summed_weights = np.bincount(indices, weights)
                        order_unique = np.argsort(ls_leaves_unique)
                        ls_leaves_unique = ls_leaves_unique[order_unique]
                        summed_weights = summed_weights[order_unique]
                        qhat = self.operation_leaf(ls_leaves_unique, quant, weights=summed_weights)

                    if treeID2testID2values != {}:
                        ls_leaves_test_unique, indices_test = np.unique(ls_leaves_test, return_inverse=True)
                        summed_weights_test = np.bincount(indices_test, weights_test)
                        order_unique_test = np.argsort(ls_leaves_test_unique)
                        ls_leaves_test_unique = ls_leaves_test_unique[order_unique_test]
                        summed_weights_test = summed_weights_test[order_unique_test]
                        qhat_test = self.operation_leaf(ls_leaves_test_unique, quant, weights=summed_weights_test)
                    return qhat, qhat_test
                
                qhat_low, qhat_low_test = f(q, j, treeID2trainID2values_low, treeID2testID2values_low)
                qhat_up, qhat_up_test = f(1-q, j, treeID2trainID2values_up, treeID2testID2values_up)
                qhat_median, qhat_median_test = f(0.5, j, treeID2trainID2values_median, treeID2testID2values_median)

                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qhat_median, qhat_median_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)


            elif 'vr' in self.settings['type_aggregation_trees']:
                def f(quant, j, treeID2trainID2values, treeID2testID2values):
                    qhat, qhat_test = 0, 0
                    count_calib = 0
                    count_test = 0
                    for k in sample2calib_trees[j]:
                        if k in treeID2trainID2values.keys():
                            ls_leaves_y = list(treeID2trainID2values[k][j])
                            qhat += self.operation_leaf(ls_leaves_y, quant)
                            count_calib += 1
                        if k in treeID2testID2values.keys():
                            ls_leaves_test = list(treeID2testID2values[k][i])
                            qhat_test += self.operation_leaf(ls_leaves_test, quant)
                            count_test += 1 
                    avg_calib = qhat / count_calib if count_calib != 0 else 0
                    avg_test = qhat_test / count_test if count_test != 0 else 0
                    return avg_calib, avg_test
                
                qhat_low, qhat_low_test = f(q, j, treeID2trainID2values_low, treeID2testID2values_low)
                qhat_up, qhat_up_test = f(1-q, j, treeID2trainID2values_up, treeID2testID2values_up)
                qhat_median, qhat_median_test = f(0.5, j, treeID2trainID2values_median, treeID2testID2values_median)                
                
                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qhat_median, qhat_median_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)

        return lower, upper, conf_scores