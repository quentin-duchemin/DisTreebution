import numpy as np
import random
from tqdm import tqdm
from UQ.Conformalisation import Conformalisation


class Conformalisation_distri_PQRT(Conformalisation):
    def __init__(self, settings=None, params=None):
        super().__init__(settings=settings, params=params)
          
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
        sample2predset = {}
        sample2JK = {}
        treeID2trainID2values, treeID2testID2values = self.preprocess_trees(trees, x_train, x_test)
                    
        for i in tqdm(range(len(y_test))):
            yi = y_test[i]
            lower, upper, conf_scores = self.get_low_up_score(i, y_train, sample2calib_trees, treeID2trainID2values, treeID2testID2values)
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

                    ls_leaves_y_unique, indices = np.unique(ls_leaves_y, return_inverse=True)
                    summed_weights = np.bincount(indices, weights)

                    ls_leaves_test_unique, indices = np.unique(ls_leaves_test, return_inverse=True)
                    summed_weights_test = np.bincount(indices, weights_test)

                    ls_leaves_y_unique = np.sort(ls_leaves_y_unique)
                    qhat = self.operation_leaf(ls_leaves_y_unique, quant, weights=summed_weights)
                    qhat_test = self.operation_leaf(ls_leaves_test_unique, quant, weights=summed_weights_test)
                    return qhat, qhat_test
                
                qhat_low = np.zeros(len(low_quantiles))
                qhat_up = np.zeros(len(low_quantiles))
                qhat_low_test = np.zeros(len(low_quantiles))
                qhat_up_test = np.zeros(len(low_quantiles))
                for i_q, q in enumerate(low_quantiles):
                    i_q_low = np.argmin(np.abs(q-self.params['train_quantiles']))
                    i_q_up = np.argmin(np.abs(1-q-self.params['train_quantiles']))
                    qhat_low[i_q], qhat_low_test[i_q] = f(q, j, treeID2trainID2values[i_q_low], treeID2testID2values[i_q_low])
                    qhat_up[i_q], qhat_up_test[i_q] = f(1-q, j, treeID2trainID2values[i_q_up], treeID2testID2values[i_q_up])
                

                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)


            elif 'vr' in self.settings['type_conformal']:
                
                def f(quant, j, treeID2trainID2values, treeID2testID2values):
                    qhat, qhat_test = 0, 0
                    count_intern = 0
                    for k in sample2calib_trees[j]:
                        ls_leaves_y = list(treeID2trainID2values[k][j])
                        ls_leaves_y = ls_leaves_y
                        qhat += self.operation_leaf(ls_leaves_y, quant)

                        ls_leaves_test = list(treeID2testID2values[k][i])
                        ls_leaves_test = ls_leaves_test
                        n = len(ls_leaves_test)
                        qhat_test += self.operation_leaf(ls_leaves_test, quant)
                        count_intern += 1 
                    return qhat/count_intern, qhat_test/count_intern
                
                qhat_low = np.zeros(len(low_quantiles))
                qhat_up = np.zeros(len(low_quantiles))
                qhat_low_test = np.zeros(len(low_quantiles))
                qhat_up_test = np.zeros(len(low_quantiles))
                for i_q, q in enumerate(low_quantiles):
                    i_q_low = np.argmin(np.abs(q-self.params['train_quantiles']))
                    i_q_up = np.argmin(np.abs(1-q-self.params['train_quantiles']))
                    qhat_low[i_q], qhat_low_test[i_q] = f(q, j, treeID2trainID2values[i_q_low], treeID2testID2values[i_q_low])
                    qhat_up[i_q], qhat_up_test[i_q] = f(1-q, j, treeID2trainID2values[i_q_up], treeID2testID2values[i_q_up])
                
                
                low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=t_fixed)
                lower.append(low_j)
                upper.append(up_j)
                conf_scores.append(conf_score_j)

            elif 'csa' in self.settings['type_conformal']:
                for k in sample2calib_trees[j]:
                    def f(quant, k, j, treeID2trainID2values, treeID2testID2values):
                        ls_leaves_y = list(treeID2trainID2values[k][j])
                        ls_leaves_test = list(treeID2testID2values[k][i])

                        ls_leaves_y = ls_leaves_y
                        qhat = self.operation_leaf(ls_leaves_y, quant)

                        ls_leaves_test = ls_leaves_test
                        qhat_test = self.operation_leaf(ls_leaves_test, quant)
                        return qhat, qhat_test
                    
                    qhat_low = np.zeros(len(low_quantiles))
                    qhat_up = np.zeros(len(low_quantiles))
                    qhat_low_test = np.zeros(len(low_quantiles))
                    qhat_up_test = np.zeros(len(low_quantiles))
                    for i_q, q in enumerate(low_quantiles):
                        i_q_low = np.argmin(np.abs(q-self.params['train_quantiles']))
                        i_q_up = np.argmin(np.abs(1-q-self.params['train_quantiles']))
                        qhat_low[i_q], qhat_low_test[i_q] = f(q, k, j, treeID2trainID2values[i_q_low], treeID2testID2values[i_q_low])
                        qhat_up[i_q], qhat_up_test[i_q] = f(1-q, k, j, treeID2trainID2values[i_q_up], treeID2testID2values[i_q_up])

                    low_j, up_j, conf_score_j = self.get_low_up_test_i(yj, qhat_low, qhat_up, qhat_low_test, qhat_up_test, t=t_fixed)
                    lower.append(low_j)
                    upper.append(up_j)
                    conf_scores.append(conf_score_j)
        
        return lower, upper, conf_scores
    
#     def get_conformal_set_split(self, trees, x_test, y_test, x_calib, y_calib, alpha):
#         low_quantiles = self.params['nominal_quantiles']
#         sample2calib_trees = {j: [i for i in range(self.params['nTrees'])] for j in range(len(y_calib))}
#         sample2predset = {i_q:{} for i_q in range(len(low_quantiles))}

#         if 'standard' in self.settings['type_conformal']:
#             treeID2calibID2values, treeID2testID2values = self.preprocess_trees(trees, x_calib, x_test)
#             for i_q, q in enumerate(low_quantiles):
#                 i_q_low = np.argmin(np.abs(q-self.params['train_quantiles']))
#                 i_q_up = np.argmax(np.abs(1-q-self.params['train_quantiles']))
#                 i_q_median = np.argmax(np.abs(0.5-self.params['train_quantiles']))
#                 low, up, conf_scores = self.get_low_up_score(0, q, y_calib, sample2calib_trees, treeID2calibID2values[i_q_low], treeID2testID2values[i_q_low], treeID2calibID2values[i_q_up], treeID2testID2values[i_q_up], treeID2calibID2values[i_q_median], treeID2testID2values[i_q_median])
#                 conf_scores = np.sort(conf_scores)
#                 t = conf_scores[int((1-alpha)*len(conf_scores))]
#                 for i in range(len(y_test)):
#                     low, up, conf_scores = self.get_low_up_score(i, q, y_calib[:1], sample2calib_trees, treeID2calibID2values[i_q_low], treeID2testID2values[i_q_low], treeID2calibID2values[i_q_up], treeID2testID2values[i_q_up], treeID2calibID2values[i_q_median], treeID2testID2values[i_q_median], t_fixed=t)
#                     sample2predset[i_q][i] = [[low[0],up[0]]]
#             return None, sample2predset