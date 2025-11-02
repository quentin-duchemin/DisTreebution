import numpy as np
import random
from tqdm import tqdm
from .Conformalisation import Conformalisation
from .utils import filter_dict



class Conformalisation_distributional(Conformalisation):
    """
    A class for handling conformalisation processes on tree-based models for distributional nested sets.
    Parameters
    ----------
    settings : dict, optional
        Configuration settings for the conformalisation process, such as the type of tree.
    params : dict, optional
        Hyperparameters for the conformalisation process.
    Methods
    -------
    conformalize_split(trees, x_calib, y_calib, alpha)
        Performs conformalization on a calibration dataset to determine the conformity threshold.
    predict_conformal_set_split(trees, x_test)
        Predicts conformal prediction sets for test data based on the conformity threshold.
    conformalize_split_group_coverage(trees, x_calib, y_calib, alpha, max_depth_group=None, get_res_on_calib=False)
        Performs group-wise conformalization on a calibration dataset to determine group-specific conformity thresholds.
    predict_conformal_set_split_group_coverage(trees, x_test, get_res_on_calib=False, x_calib=None, y_calib=None)
        Predicts conformal prediction sets for test data with group coverage based on group-specific conformity thresholds.
    """
    def __init__(self, settings=None, params=None):
        super().__init__(settings=settings, params=params)

    def conformalize_split(self, trees, x_calib, y_calib, alpha):
        """
        Performs conformalization on a calibration dataset by cmoputing conformity scores and taking the appropriate quantile.   
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
        """
        sample2calib_trees = {j: [i for i in range(len(trees))] for j in range(len(y_calib))}

        treeID2calibID2values = self.preprocess_trees(trees, x_calib)
        _, _, conf_scores = self.get_low_up_score(0, y_calib, sample2calib_trees, treeID2calibID2values, {})
        conf_scores = np.sort(conf_scores)
        self.conf_thresh = conf_scores[int(alpha*len(conf_scores))]


    def predict_conformal_set_split(self, trees, x_test):
        """
        Predicts conformal prediction sets for test data based on the conformity threshold determined during calibration.
        Parameters
        ----------
        trees : list
            List of trained tree models.
        x_test : np.ndarray
            Test input data.
        Returns
        -------
        sample2predset : dict
            Dictionary mapping test sample indices to their conformal prediction sets [lower_bound, upper_bound].
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
        """
        Performs group-wise conformalization on a calibration dataset to determine group-specific conformity thresholds.   
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
        max_depth_group : int, optional
            Maximum depth for group coverage trees.
        get_res_on_calib : bool, optional
            Whether to return results on the calibration set.
        Returns
        -------
        If get_res_on_calib is True:
            group2sizecalib : dict
                Dictionary mapping group identifiers to the size of the calibration set in that group.
            group2covcalib : dict
                Dictionary mapping group identifiers to the coverage of the calibration set in that group.
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
        """
        Predicts conformal prediction sets for test data with group coverage based on group-specific conformity thresholds.
        Parameters
        ----------
        trees : list
            List of trained tree models.
        x_test : np.ndarray
            Test input data.
        get_res_on_calib : bool, optional
            Whether to return results on the calibration set.
        x_calib : np.ndarray, optional
            Calibration input data (required if get_res_on_calib is True).
        y_calib : np.ndarray, optional
            Calibration output data (required if get_res_on_calib is True).
        Returns
        -------
        If get_res_on_calib is True:
            group2sizecalib : dict
                Dictionary mapping group identifiers to the size of the calibration set in that group.
            group2covcalib : dict
                Dictionary mapping group identifiers to the coverage of the calibration set in that group.
            treeID2testID2group : dict
                Dictionary mapping tree IDs to another dictionary that maps test sample IDs to their group information.
            sample2predset : dict
                Dictionary mapping test sample indices to their conformal prediction sets [lower_bound, upper_bound].
        Else:
            treeID2testID2group : dict
                Dictionary mapping tree IDs to another dictionary that maps test sample IDs to their group information.
            sample2predset : dict
                Dictionary mapping test sample indices to their conformal prediction sets [lower_bound, upper_bound].
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
        """
        Determines the appropriate lower and upper bounds for a test sample based on conformity scores and a fixed threshold.
        Parameters
        ----------
        yj : float
            The true output value for the calibration sample.
        qhat_low : np.ndarray
            Array of lower quantile estimates from the calibration data.
        qhat_up : np.ndarray
            Array of upper quantile estimates from the calibration data.
        qhat_low_test : np.ndarray
            Array of lower quantile estimates for the test data.
        qhat_up_test : np.ndarray
            Array of upper quantile estimates for the test data.
        t : float, optional
            Fixed conformity threshold. If None, the method searches for the appropriate threshold.
        Returns
        -------
        low_j : float
            The lower bound for the test sample.
        up_j : float
            The upper bound for the test sample.
        t_idx : int
            The index of the selected threshold.
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
        """
        Computes lower and upper bounds along with conformity scores for a given test sample.
        Parameters
        ----------
        i : int
            Index of the test sample.
        y_train : np.ndarray
            Array of training output values.
        sample2calib_trees : dict
            Dictionary mapping sample indices to lists of tree IDs used for calibration.
        treeID2trainID2values : dict
            Dictionary mapping tree IDs to another dictionary that maps training sample IDs to sorted leaf values.
        treeID2testID2values : dict
            Dictionary mapping tree IDs to another dictionary that maps test sample IDs to sorted leaf values.
        t_fixed : float, optional
            Fixed conformity threshold. If None, the method searches for the appropriate threshold.
        Returns
        -------
        lower : list
            List of lower bounds for the test sample.
        upper : list
            List of upper bounds for the test sample.
        conf_scores : list
            List of conformity scores for the test sample.
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