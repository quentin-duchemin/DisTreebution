import numpy as np
from .entropies_MultiQuantiles import entropies_MultiQuantiles
import matplotlib.pyplot as plt

class RegressionTreeQuantile:
    def __init__(self, quantiles, max_depth=None, min_samples_split=2, use_LOO=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.quantiles = quantiles
        self.use_LOO = use_LOO

    def fit(self, X, y, depth=0, ref_tree=None, max_depth_ref_tree=-1):
        if depth == self.max_depth or X.shape[0] < self.min_samples_split:
            self.y = y
            return

        if not(ref_tree is None) and (max_depth_ref_tree>depth):
            best_split = ref_tree.feature_index, ref_tree.threshold
            ref_tree_left = ref_tree.left
            ref_tree_right = ref_tree.right
        else:
            best_split = self.find_best_split(X, y)
            ref_tree_left = None
            ref_tree_right = None
        
        if best_split is not None:
            feature_index, threshold = best_split
            self.feature_index = feature_index
            self.threshold = threshold

            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask            
            

            self.left = RegressionTreeQuantile(self.quantiles, max_depth=self.max_depth, min_samples_split=self.min_samples_split, use_LOO=self.use_LOO)
            self.right = RegressionTreeQuantile(self.quantiles, max_depth=self.max_depth, min_samples_split=self.min_samples_split, use_LOO=self.use_LOO)

            self.left.fit(X[left_mask,:], y[left_mask], depth + 1, ref_tree=ref_tree_left, max_depth_ref_tree=max_depth_ref_tree)
            self.right.fit(X[right_mask,:], y[right_mask], depth + 1, ref_tree=ref_tree_right, max_depth_ref_tree=max_depth_ref_tree)
        else:
            self.y = y
            return
            #self.value = [np.sort(y)[min(int(q*len(y)),len(y)-1)] for q in self.quantiles]

    def find_best_split(self, X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None

        best_score = None
        best_split = None

        for feature_index in range(num_features):
            order = np.argsort(X[:, feature_index])
            entropies_up = entropies_MultiQuantiles(order, y, self.quantiles, use_LOO=self.use_LOO)
            entropies_down = entropies_MultiQuantiles(np.flip(order), y, self.quantiles, use_LOO=self.use_LOO)
            
            weights = np.cumsum(np.ones(len(entropies_up)-1))
            weights = np.concatenate((np.array([0]), weights), axis=0)
            
            all_scores = weights * entropies_up + np.flip(weights) * np.flip(entropies_down)
            global_score = all_scores[-1]
            all_idx_split = np.argsort(all_scores[1:-1])
            condition = False
            idx_best = 0
            while not(condition) and idx_best<len(all_idx_split):
                idx_split = all_idx_split[idx_best]
                score = all_scores[idx_split+1]
                if abs(X[order[idx_split],feature_index]-X[order[idx_split+1],feature_index])>1e-5:
                    threshold = (X[order[idx_split], feature_index]+X[order[idx_split+1], feature_index])/2

                    nb_left_leaves = np.sum(X[:, feature_index] <= threshold)
                    if (nb_left_leaves>=self.min_samples_split) and (num_samples-nb_left_leaves>=self.min_samples_split):
                        if self.use_LOO:
                            # Check if the split will lead to decrease the entropy
                            if score<global_score:
                                if (best_score is None) or (best_score > score):
                                    condition = True
                                    best_score = score
                                    best_split = (feature_index, threshold) 
                            
                        else:
                            if (best_score is None) or (best_score > score):
                                condition = True
                                best_score = score
                                best_split = (feature_index, threshold)
                idx_best += 1

        return best_split
                                           

    def predict(self, X):
        if hasattr(self, 'value'):
            return np.tile(np.array(self.value).reshape(1,-1), (X.shape[0], 1))
        else:
            left_mask = X[:, self.feature_index] < self.threshold
            right_mask = ~left_mask

            y_left = self.left.predict(X[left_mask])
            y_right = self.right.predict(X[right_mask])

            result = np.empty((X.shape[0], len(self.quantiles)))
            result[left_mask,:] = y_left
            result[right_mask,:] = y_right

            return result
        
    def get_values_leaf(self, X, indexes):
        if hasattr(self, 'y'):
            return [[list(indexes), self.y]]
        else:
            left_mask = X[:, self.feature_index] < self.threshold
            right_mask = ~left_mask
            idxleft = indexes[np.where(left_mask)[0].astype(int)]
            y_left, y_right = [], []
            if len(idxleft)!=0:
                y_left = self.left.get_values_leaf(X[left_mask], idxleft)
            idxright = indexes[np.where(right_mask)[0].astype(int)]
            if len(idxright)!=0:
                y_right = self.right.get_values_leaf(X[right_mask], idxright)

            return y_left + y_right
        
        
        
    def get_values_leaf_and_groups(self, X, indexes, current_group_depth=str(), max_depth_group=1):
        if hasattr(self, 'y'):
            return [[list(indexes), self.y, current_group_depth]]
        else:
            if len(current_group_depth)>=max_depth_group:
                left_group_depth = current_group_depth
                right_group_depth = current_group_depth
            else:
                left_group_depth = current_group_depth + str('0')
                right_group_depth = current_group_depth + str('1')
            left_mask = X[:, self.feature_index] < self.threshold
            right_mask = ~left_mask
            idxleft = indexes[np.where(left_mask)[0].astype(int)]
            y_left, y_right = [], []
            if len(idxleft)!=0:
                y_left = self.left.get_values_leaf_and_groups(X[left_mask], idxleft, current_group_depth=left_group_depth, max_depth_group=max_depth_group)
            idxright = indexes[np.where(right_mask)[0].astype(int)]
            if len(idxright)!=0:
                y_right = self.right.get_values_leaf_and_groups(X[right_mask], idxright, current_group_depth=right_group_depth, max_depth_group=max_depth_group)

            return y_left + y_right
