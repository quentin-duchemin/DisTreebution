import numpy as np
import random
from tqdm import tqdm
from UQ.Conformalisation import Conformalisation


class Conformalisation_CQR(Conformalisation):
    """
    Class handling quantile based nested sets (i.e. CQR, CQR-m, CQR-r).
    """
    def __init__(self, settings=None, params=None):
        super().__init__(settings=settings, params=params)      
        
    
    def get_conformal_set_split(self, trees, x_test, y_test, x_calib, y_calib, alpha):
        low_quantiles = self.params['nominal_quantiles']
        sample2calib_trees = {j: [i for i in range(self.params['nTrees'])] for j in range(len(y_calib))}
        sample2predset = {i_q:{} for i_q in range(len(low_quantiles))}

        treeID2calibID2values, treeID2testID2values = self.preprocess_trees(trees, x_calib, x_test)
        for i_q, q in enumerate(low_quantiles):
            low, up, conf_scores = self.get_low_up_score(0, q, y_calib, sample2calib_trees, treeID2calibID2values, treeID2testID2values)
            conf_scores = np.sort(conf_scores)
            t = conf_scores[int((1-alpha)*len(conf_scores))]
            for i in range(len(y_test)):
                low, up, conf_scores = self.get_low_up_score(i, q, y_calib, sample2calib_trees, treeID2calibID2values, treeID2testID2values, t_fixed=t)
                sample2predset[i_q][i] = [[low[0],up[0]]]
        return None, sample2predset
      
    def get_conformal_set_split_group_coverage(self, trees, x_test, y_test, x_calib, y_calib, max_depth_group, alpha):
        low_quantiles = self.params['nominal_quantiles']
        sample2calib_trees = {j: [i for i in range(len(trees))] for j in range(len(y_calib))}
        sample2predset = {i_q:{} for i_q in range(len(low_quantiles))}

        treeID2calibID2values, treeID2testID2values, treeID2calibID2group, treeID2testID2group = self.preprocess_trees_with_groups(trees, x_calib, x_test, max_depth_group)
        all_groups_test = list(set([treeID2testID2group[k][i] for k in range(self.params['nTrees']) for i in range(len(y_test))]))
        group2conf_scores = {group:[] for group in all_groups_test}
        group2calibID2 = {group:[] for group in all_groups_test}
        for j in range(len(y_calib)):
            try:
                group2calibID2[treeID2calibID2group[0][j]].append(j)
            except:
                pass
        for group in all_groups_test:
            group2calibID2[group] = np.array(group2calibID2[group]).astype(int)
        for i_q, q in enumerate(low_quantiles):
            for group in all_groups_test:
                temp_treeID2calibID2values = {k: {count: treeID2calibID2values[k][i] for count, i in enumerate(group2calibID2[group])} for k in range(self.params['nTrees'])}
                low, up, conf_scores = self.get_low_up_score(0, q, y_calib[group2calibID2[group]], sample2calib_trees, temp_treeID2calibID2values, treeID2testID2values)
                conf_scores = np.sort(conf_scores)
                group2conf_scores[group] = conf_scores[int((1-alpha)*len(conf_scores))]
            for i in tqdm(range(len(y_test))):
                t = group2conf_scores[treeID2testID2group[0][i]]
                low, up, conf_scores = self.get_low_up_score(i, q, y_calib, sample2calib_trees, treeID2calibID2values, treeID2testID2values, t_fixed=t)
                sample2predset[i_q][i] = [[low[i],up[i]] for i in range(len(low))]
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
        
    def get_low_up_score(self, i, q, y_train, sample2calib_trees, treeID2trainID2values, treeID2testID2values, t_fixed=None):
        # i: index of the test sample
        lower = []
        upper = []
        conf_scores = []
        for j, yj in enumerate(y_train):                       
            if 'vr-avg' in self.settings['type_conformal']:
                ls_leaves_y = []
                weights = []
                weights_test = []
                ls_leaves_test = []
                qhat_low = 0
                qhat_up = 0
                qmedian = 0
                qmedian_test = 0
                qhat_low_test = 0
                qhat_up_test = 0
                count = 0
                for k in sample2calib_trees[j]:
                    ls_new = list(treeID2trainID2values[k][j])
                    ls_leaves_y = ls_leaves_y + ls_new
                    weights = weights + list(np.ones(len(ls_new))/len(ls_new))
                    ls_new = list(treeID2testID2values[k][i])
                    ls_leaves_test = ls_leaves_test + ls_new
                    weights_test = weights_test + list(np.ones(len(ls_new))/len(ls_new))

                ls_leaves_y_unique, indices = np.unique(ls_leaves_y, return_inverse=True)
                summed_weights = np.bincount(indices, weights)

                ls_leaves_test_unique, indices = np.unique(ls_leaves_test, return_inverse=True)
                summed_weights_test = np.bincount(indices, weights_test)

                order_unique = np.argsort(ls_leaves_y_unique)
                ls_leaves_y_unique = ls_leaves_y_unique[order_unique]
                summed_weights = summed_weights[order_unique]
                qhat_low += self.operation_leaf(ls_leaves_y_unique, q, weights=summed_weights)
                qhat_up += self.operation_leaf(ls_leaves_y_unique, 1-q, weights=summed_weights)
                qmedian += self.operation_leaf(ls_leaves_y_unique, 0.5, weights=summed_weights)
               
                order_unique_test = np.argsort(ls_leaves_test_unique)
                ls_leaves_test_unique = ls_leaves_test_unique[order_unique_test]
                summed_weights_test = summed_weights_test[order_unique_test]
                qhat_low_test += self.operation_leaf(ls_leaves_test_unique, q, weights=summed_weights_test)
                qhat_up_test += self.operation_leaf(ls_leaves_test_unique, 1-q, weights=summed_weights_test)
                qmedian_test += self.operation_leaf(ls_leaves_test_unique, 0.5, weights=summed_weights_test)

                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qmedian, qmedian_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)


            elif 'vr' in self.settings['type_conformal']:
                ls_leaves_y = []
                ls_leaves_test = []
                qhat_low = 0
                qhat_up = 0
                qmedian = 0
                qmedian_test = 0
                qhat_low_test = 0
                qhat_up_test = 0
                count = 0
                for k in sample2calib_trees[j]:
                    ls_leaves_y = list(treeID2trainID2values[k][j])
                    qhat_low += self.operation_leaf(ls_leaves_y, q)
                    qhat_up += self.operation_leaf(ls_leaves_y, 1-q)
                    qmedian += self.operation_leaf(ls_leaves_y, 0.5)

                    ls_leaves_test = list(treeID2testID2values[k][i])
                    n = len(ls_leaves_test)
                    qhat_low_test += self.operation_leaf(ls_leaves_test, q)
                    qhat_up_test += self.operation_leaf(ls_leaves_test, 1-q)
                    qmedian_test += self.operation_leaf(ls_leaves_test, 0.5)

                    count += 1

                qhat_low /= count
                qhat_up /= count  
                qmedian /= count
                qhat_low_test /= count
                qhat_up_test /= count
                qmedian_test /= count
                
                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, qmedian, qmedian_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)

        return lower, upper, conf_scores
