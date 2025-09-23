import numpy as np
import random
from tqdm import tqdm
from UQ.Conformalisation import Conformalisation


class Conformalisation_distri(Conformalisation):
    def __init__(self, settings=None, params=None):
        super().__init__(settings=settings, params=params)
        
    def get_conformal_set_split(self, trees, x_test, y_test, x_calib, y_calib, alpha):
        sample2calib_trees = {j: [i for i in range(len(trees))] for j in range(len(y_calib))}
        sample2predset = {}

        if 'standard' in self.settings['type_conformal']:
            treeID2calibID2values, treeID2testID2values = self.preprocess_trees(trees, x_calib, x_test)
            low, up, conf_scores = self.get_low_up_score(0, y_calib, sample2calib_trees, treeID2calibID2values, treeID2testID2values)
            conf_scores = np.sort(conf_scores)
            t = conf_scores[int(alpha*len(conf_scores))]
            for i in range(len(y_test)):
                low, up, conf_scores = self.get_low_up_score(i, y_calib[:1], sample2calib_trees, treeID2calibID2values, treeID2testID2values, t_fixed=t)
                sample2predset[i] = [[low[0],up[0]]]
            return None, sample2predset
                    
        else:
            return self.get_conformal_set_oob(trees, x_calib, y_calib, x_test, y_test, sample2calib_trees, alpha)
        
    def get_conformal_set_split_group_coverage(self, trees, x_test, y_test, x_calib, y_calib, max_depth_group, alpha, get_res_on_calib=False):
        sample2calib_trees = {j: [i for i in range(len(trees))] for j in range(len(y_calib))}
        sample2predset = {}
        if 'standard' in self.settings['type_conformal']:
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
            for group in all_groups_test:
                temp_treeID2calibID2values = {k: {count: treeID2calibID2values[k][i] for count, i in enumerate(group2calibID2[group])} for k in range(self.params['nTrees'])}
                low, up, conf_scores = self.get_low_up_score(0, y_calib[group2calibID2[group]], sample2calib_trees, temp_treeID2calibID2values, treeID2testID2values)
                conf_scores = np.sort(conf_scores)
                group2conf_scores[group] = conf_scores[int(alpha*len(conf_scores))]
                print(group, 'idx', int(alpha*len(conf_scores)), group2conf_scores[group])
                
            group2size = {group:0 for group in all_groups_test}
            group2cov = {group:0 for group in all_groups_test}
            for i in tqdm(range(len(y_test))):
                t = group2conf_scores[treeID2testID2group[0][i]]
                low, up, conf_scores = self.get_low_up_score(i, y_calib, sample2calib_trees, treeID2calibID2values, treeID2testID2values, t_fixed=t)
                sample2predset[i] = [[low[0],up[0]]]
                group2size[treeID2testID2group[0][i]] += 1
                group2cov[treeID2testID2group[0][i]] += 1.*( (y_test[i]>=low[0]) and (y_test[i]<=up[0]) )
            for group, cov in group2cov.items():
                group2cov[group] = cov/group2size[group]
                
            if get_res_on_calib:
                group2sizecalib = {group:0 for group in all_groups_test}
                group2covcalib = {group:0 for group in all_groups_test}
                sample2predsetcalib = {}
                for i in tqdm(range(len(y_calib))):
                    t = group2conf_scores[treeID2calibID2group[0][i]]
                    low, up, conf_scores = self.get_low_up_score(i, y_calib, sample2calib_trees, treeID2calibID2values, treeID2calibID2values, t_fixed=t)
                    sample2predsetcalib[i] = [[low[0],up[0]]]
                    group2sizecalib[treeID2calibID2group[0][i]] += 1
                    group2covcalib[treeID2calibID2group[0][i]] += 1.*( (y_calib[i]>=low[0]) and (y_calib[i]<=up[0]) )
                for group, cov in group2covcalib.items():
                    group2covcalib[group] = cov/group2sizecalib[group]
                return group2size, group2cov, group2sizecalib, group2covcalib, treeID2testID2group, sample2predset
            else:
                return group2size, group2cov, treeID2testID2group, sample2predset
        else:
            pass
        
        
    def get_low_up_test_i(self, yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=None):
        t_idx = t
        low_quantiles = self.params['list_distri_low_quantiles']
        if t_idx is None:
            t_idx = len(low_quantiles)-1
            condition = (qhat_low[t_idx]<=yj) and (yj<=qhat_up[t_idx])
            while not(condition) and (t_idx>0):
                t_idx = t_idx - 1
                condition = (qhat_low[t_idx]<=yj) and (yj<=qhat_up[t_idx])
        return qhat_low_test[t_idx], qhat_up_test[t_idx], t_idx
        
    def get_conformal_set_oob(self, trees, x_train, y_train, x_test, y_test, sample2calib_trees, alpha):
        """
        OOB conformal method.
        """
        sample2predset = {}
        sample2JK = {}
        treeID2trainID2values, treeID2testID2values = self.preprocess_trees(trees, x_train, x_test)
                    
        for i in tqdm(range(len(y_test))):
            yi = y_test[i]
            lower, upper, conf_scores = self.get_low_up_score(i, q, y_train, sample2calib_trees, treeID2trainID2values, treeID2testID2values)
            sample2JK[i], sample2predset[i] = self.oob_get_set(lower, upper, alpha)

        interval_JK = np.zeros((len(y_test),2))
        for i in range(len(y_test)):
            interval_JK[i,0] = sample2JK[i][0]
            interval_JK[i,1] = sample2JK[i][1]
        return interval_JK, sample2predset

    def get_low_up_score(self, i, y_train, sample2calib_trees, treeID2trainID2values, treeID2testID2values, t_fixed=None):
        # i: index of the test sample
        low_quantiles = self.params['list_distri_low_quantiles']
        lower = []
        upper = []
        conf_scores = []
        for j, yj in enumerate(y_train):                       
            if 'vr-avg' in self.settings['type_conformal']:
                ls_leaves_y = []
                weights = []
                weights_test = []
                ls_leaves_test = []
                qhat_low = np.zeros(len(low_quantiles))
                qhat_up = np.zeros(len(low_quantiles))
                qhat_low_test = np.zeros(len(low_quantiles))
                qhat_up_test = np.zeros(len(low_quantiles))
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

                ls_leaves_test_unique, indices_test = np.unique(ls_leaves_test, return_inverse=True)
                summed_weights_test = np.bincount(indices_test, weights_test)

                order_unique = np.argsort(ls_leaves_y_unique)
                ls_leaves_y_unique = ls_leaves_y_unique[order_unique]
                summed_weights = summed_weights[order_unique]

                order_unique_test = np.argsort(ls_leaves_test_unique)
                ls_leaves_test_unique = ls_leaves_test_unique[order_unique_test]
                summed_weights_test = summed_weights_test[order_unique_test]
                
                for i_q, q in enumerate(low_quantiles):
                    qhat_low[i_q] = self.operation_leaf(ls_leaves_y_unique, q, weights=summed_weights)
                    qhat_up[i_q] = self.operation_leaf(ls_leaves_y_unique, 1-q, weights=summed_weights)

                    qhat_low_test[i_q] = self.operation_leaf(ls_leaves_test_unique, q, weights=summed_weights_test)
                    qhat_up_test[i_q] = self.operation_leaf(ls_leaves_test_unique, 1-q, weights=summed_weights_test)

                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)


            elif 'vr' in self.settings['type_conformal']:
                ls_leaves_y = []
                ls_leaves_test = []
                qhat_low = np.zeros(len(low_quantiles))
                qhat_up = np.zeros(len(low_quantiles))
                qhat_low_test = np.zeros(len(low_quantiles))
                qhat_up_test = np.zeros(len(low_quantiles))
                count = 0
                for k in sample2calib_trees[j]:
                    ls_leaves_y = list(treeID2trainID2values[k][j])
                    ls_leaves_y = ls_leaves_y

                    ls_leaves_test = list(treeID2testID2values[k][i])
                    ls_leaves_test = ls_leaves_test
                    
                    for i_q, q in enumerate(low_quantiles):
                        qhat_low[i_q] += self.operation_leaf(ls_leaves_y, q)
                        qhat_up[i_q] += self.operation_leaf(ls_leaves_y, 1-q)
                        
                        qhat_low_test[i_q] += self.operation_leaf(ls_leaves_test, q)
                        qhat_up_test[i_q] += self.operation_leaf(ls_leaves_test, 1-q)
                    count += 1

                qhat_low /= count
                qhat_up /= count  
                qhat_low_test /= count
                qhat_up_test /= count
                
                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)

            elif 'csa' in self.settings['type_conformal']:
                for k in sample2calib_trees[j]:
                    qhat_low = np.zeros(len(low_quantiles))
                    qhat_up = np.zeros(len(low_quantiles))
                    qhat_low_test = np.zeros(len(low_quantiles))
                    qhat_up_test = np.zeros(len(low_quantiles))
                    ls_leaves_y = list(treeID2trainID2values[k][j])
                    ls_leaves_test = list(treeID2testID2values[k][i])
                    ls_leaves_y = ls_leaves_y
                    ls_leaves_test = ls_leaves_test
                    for i_q, q in enumerate(low_quantiles):
                        qhat_low[i_q] = self.operation_leaf(ls_leaves_y, q)
                        qhat_up[i_q] = self.operation_leaf(ls_leaves_y, 1-q)
                        
                        qhat_low_test[i_q] = self.operation_leaf(ls_leaves_test, q)
                        qhat_up_test[i_q] = self.operation_leaf(ls_leaves_test, 1-q)

                    low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=t_fixed)

                    lower.append(low_j)
                    upper.append(up_j)
                    conf_scores.append(conf_score_j)
        return lower, upper, conf_scores