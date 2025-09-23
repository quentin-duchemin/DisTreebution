import numpy as np
import random
from tqdm import tqdm
from UQ.Conformalisation import Conformalisation
            
class Conformalisation_CQR_PQRT(Conformalisation):
    """
    Implements quantile-based nested conformal prediction sets, including CQR, CQR-m, and CQR-r variants.
    This class provides methods to compute conformal prediction intervals using quantile regression forests,
    supporting various nested set constructions. It handles calibration and prediction for conformalized quantile regression
    with different types of set shapes, as specified by the 'nested_set' setting.
    Methods
    -------
    __init__(self, settings=None, params=None)
        Initializes the conformalization object with given settings and parameters.
    get_conformal_set_split(self, trees, x_test, y_test, x_calib, y_calib, alpha)
        Computes conformal prediction sets for test samples using calibration data and a collection of trees.
        Returns a mapping from quantile index to predicted intervals for each test sample.
    get_low_up_test_i(self, yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qmedian, qmedian_test, t=None)
        Computes the lower and upper bounds of the prediction interval for a single sample, according to the selected
        nested set type ('CQR', 'CQR-m', or 'CQR-r'). Optionally uses a fixed threshold t.
    get_low_up_score(self, i, q, y_train, sample2calib_trees, treeID2trainID2values_low, treeID2testID2values_low,
                     treeID2trainID2values_up, treeID2testID2values_up, treeID2trainID2values_median,
                     treeID2testID2values_median, t_fixed=None)
        For a given test sample, computes the lower and upper bounds and conformity scores for all calibration samples.
        Supports both 'vr-avg' and 'vr' types of conformalization, aggregating predictions across trees.
    """
    def __init__(self, settings=None, params=None):
        super().__init__(settings=settings, params=params)      
        
    
    def get_conformal_set_split(self, trees, x_test, y_test, x_calib, y_calib, alpha):
        low_quantiles = self.params['nominal_quantiles']
        sample2calib_trees = {j: [i for i in range(self.params['nTrees'])] for j in range(len(y_calib))}
        sample2predset = {i_q:{} for i_q in range(len(low_quantiles))}

        treeID2calibID2values, treeID2testID2values = self.preprocess_trees(trees, x_calib, x_test)
        for i_q, q in enumerate(low_quantiles):
            i_q_low = np.argmin(np.abs(q-self.params['train_quantiles']))
            i_q_up = np.argmin(np.abs(1-q-self.params['train_quantiles']))
            i_q_median = np.argmin(np.abs(0.5-self.params['train_quantiles']))
            low, up, conf_scores = self.get_low_up_score(0, q, y_calib, sample2calib_trees, treeID2calibID2values[i_q_low], treeID2testID2values[i_q_low], treeID2calibID2values[i_q_up], treeID2testID2values[i_q_up], treeID2calibID2values[i_q_median], treeID2testID2values[i_q_median])
            conf_scores = np.sort(conf_scores)
            t = conf_scores[int((1-alpha)*len(conf_scores))]
            for i in range(len(y_test)):
                low, up, conf_scores = self.get_low_up_score(i, q, y_calib[:1], sample2calib_trees, treeID2calibID2values[i_q_low], treeID2testID2values[i_q_low], treeID2calibID2values[i_q_up], treeID2testID2values[i_q_up], treeID2calibID2values[i_q_median], treeID2testID2values[i_q_median], t_fixed=t)
                sample2predset[i_q][i] = [[low[0],up[0]]]
        return None, sample2predset

    def get_low_up_test_i(self, yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qmedian, qmedian_test, t=None):
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
        # i: index of the test sample
        lower = []
        upper = []
        conf_scores = []
        for j, yj in enumerate(y_train):                       
            if 'vr-avg' in self.settings['type_conformal']:
                def f(quant, j, treeID2trainID2values, treeID2testID2values):
                    ls_leaves_y = []
                    weights = []
                    weights_test = []
                    ls_leaves_test = []
                    for k in sample2calib_trees[j]:
                        ls_new = list(treeID2trainID2values[k][j])
                        ls_leaves_y = ls_leaves_y + ls_new
                        weights = weights + list(np.ones(len(ls_new))/len(ls_new))
                        ls_new = list(treeID2testID2values[k][i])
                        ls_leaves_test = ls_leaves_test + ls_new
                        weights_test = weights_test + list(np.ones(len(ls_new))/len(ls_new))

                    ls_leaves_unique, indices = np.unique(ls_leaves_y, return_inverse=True)
                    summed_weights = np.bincount(indices, weights)

                    ls_leaves_test_unique, indices_test = np.unique(ls_leaves_test, return_inverse=True)
                    summed_weights_test = np.bincount(indices_test, weights_test)

                    order_unique = np.argsort(ls_leaves_unique)
                    ls_leaves_unique = ls_leaves_unique[order_unique]
                    summed_weights = summed_weights[order_unique]
                    
                    order_unique_test = np.argsort(ls_leaves_test_unique)
                    ls_leaves_test_unique = ls_leaves_test_unique[order_unique_test]
                    summed_weights_test = summed_weights_test[order_unique_test]
                    
                    qhat = self.operation_leaf(ls_leaves_unique, quant, weights=summed_weights)
                    qhat_test = self.operation_leaf(ls_leaves_test_unique, quant, weights=summed_weights_test)
                    return qhat, qhat_test
                
                qhat_low, qhat_low_test = f(q, j, treeID2trainID2values_low, treeID2testID2values_low)
                qhat_up, qhat_up_test = f(1-q, j, treeID2trainID2values_up, treeID2testID2values_up)
                qhat_median, qhat_median_test = f(0.5, j, treeID2trainID2values_median, treeID2testID2values_median)

                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qhat_median, qhat_median_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)


            elif 'vr' in self.settings['type_conformal']:
                def f(quant, j, treeID2trainID2values, treeID2testID2values):
                    qhat, qhat_test = 0, 0
                    count_intern = 0
                    for k in sample2calib_trees[j]:
                        ls_leaves_y = list(treeID2trainID2values[k][j])
                        ls_leaves_y = np.sort(ls_leaves_y)
                        qhat += self.operation_leaf(ls_leaves_y, quant)

                        ls_leaves_test = list(treeID2testID2values[k][i])
                        ls_leaves_test = np.sort(ls_leaves_test)
                        n = len(ls_leaves_test)
                        qhat_test += self.operation_leaf(ls_leaves_test, quant)
                        count_intern += 1 
                    return qhat/count_intern, qhat_test/count_intern
                
                qhat_low, qhat_low_test = f(q, j, treeID2trainID2values_low, treeID2testID2values_low)
                qhat_up, qhat_up_test = f(1-q, j, treeID2trainID2values_up, treeID2testID2values_up)
                qhat_median, qhat_median_test = f(0.5, j, treeID2trainID2values_median, treeID2testID2values_median)                
                
                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qhat_median, qhat_median_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)

        return lower, upper, conf_scores