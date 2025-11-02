import numpy as np
from .entropies_CRPS import entropies_CRPS
from ..QRT.entropies_MultiQuantiles import entropies_MultiQuantiles
import random

# https://github.com/leimao/Decision_Tree_Python

class RegressionTree:
    """
    A custom regression tree class computing information gains based on the entropy associated to the CRPS loss.
    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    limit_use_CRPS : int or None, optional
        If set, use CRPS loss for nodes with sample size <= limit_use_CRPS, otherwise use multiple quantile loss.
    quantiles : list or None, optional
        List of quantiles to use for the multiple quantile loss. Required if limit_use_CRPS is set.
    use_LOO : bool, default=True
        Whether to use leave-one-out estimation in the entropy/loss calculation.
    Attributes
    ----------
    feature_index : int
        Index of the feature used for splitting at the current node.
    threshold : float
        Threshold value for the split at the current node.
    left : RegressionTree
        Left child node.
    right : RegressionTree
        Right child node.
    y : array-like
        Target values at the leaf node.
    Methods
    -------
    fit(X, y, depth=0, ref_tree=None, max_depth_ref_tree=-1)
        Fit the regression tree to the data.
    find_best_split(X, y)
        Find the best feature and threshold to split the data at the current node.
    predict(X)
        Predict target values for given input samples.
    get_values_leaf(X, indexes)
        Retrieve the samples and their target values for each leaf.
    get_values_leaf_and_groups(X, indexes, current_group_depth=str(), max_depth_group=1)
        Retrieve the samples, their target values, and group identifiers for each leaf up to a specified group depth.
    Notes
    -----
    - This implementation supports custom loss functions CRPS for splitting.
    - The tree can optionally follow the structure of a reference tree up to a certain depth.
    """
    def __init__(self, max_depth=None, min_samples_split=2, limit_use_CRPS=None, quantiles=None, use_LOO=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.limit_use_CRPS = limit_use_CRPS
        self.use_LOO = use_LOO
        if not(limit_use_CRPS is None) and (quantiles is None):
            assert False, ("You need to provide a list of quantiles if you plan to use the Multiple Quantile loss at node with a large number of samples")
        else:
            self.quantiles = quantiles

    def fit(self, X, y, depth=0, ref_tree=None, max_depth_ref_tree=-1):
        """
        Recursively fits the regression tree to the provided data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        depth : int, optional
            Current depth of the tree (default is 0).
        ref_tree : RegressionTree or None, optional
            Reference tree to guide the splitting process (default is None). If provided, the tree will copy splits from the reference tree up to `max_depth_ref_tree`.
        max_depth_ref_tree : int, optional
            Maximum depth up to which the reference tree is used for splitting (default is -1, meaning not used).

        Returns
        -------
        None

        Notes
        -----
        - If the maximum depth is reached or the number of samples is less than `min_samples_split`, the node becomes a leaf and stores the target values.
        - If a reference tree is provided and the current depth is less than `max_depth_ref_tree`, the split is copied from the reference tree.
        - Otherwise, the best split is found using the `find_best_split` method.
        - The method recursively fits the left and right child nodes.
        """
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

            self.left = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, limit_use_CRPS=self.limit_use_CRPS, quantiles=self.quantiles, use_LOO=self.use_LOO)
            self.right = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, limit_use_CRPS=self.limit_use_CRPS, quantiles=self.quantiles, use_LOO=self.use_LOO)

            self.left.fit(X[left_mask], y[left_mask], depth + 1, ref_tree=ref_tree_left, max_depth_ref_tree=max_depth_ref_tree)
            self.right.fit(X[right_mask], y[right_mask], depth + 1, ref_tree=ref_tree_right, max_depth_ref_tree=max_depth_ref_tree)
        else:
            self.y = y
            return

    def find_best_split(self, X, y):
        """
        Finds the best feature and threshold to split the data for a regression tree node.
        This method evaluates all possible splits across all features to determine the optimal split point
        that minimizes a custom score based on the CRPS entropy. It ensures that the resulting child nodes satisfy the minimum sample split constraint.
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        Returns
        -------
        best_split : tuple or None
            A tuple (feature_index, threshold) representing the best split found, or None if no valid split exists.
        """
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None


        best_score = None
        best_split = None

        for feature_index in range(num_features):
            order = np.argsort(X[:, feature_index])
            
            if (self.limit_use_CRPS is None) or (len(y)<=self.limit_use_CRPS):
                entropies_up = entropies_CRPS(order, y, use_LOO=self.use_LOO)
                entropies_down = entropies_CRPS(np.flip(order), y, use_LOO=self.use_LOO)
            else:
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
                        if (score<global_score):
                            condition = True
                            if (best_score is None) or (best_score > score):
                                best_score = score
                                best_split = (feature_index, threshold)
                idx_best += 1

        return best_split

    def predict(self, X):
        """
        Predict target values for the given input samples.
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        Returns
        -------
        np.ndarray
            Predicted target values of shape (n_samples,).
        """
        if hasattr(self, 'y'):
            return random.choices(list(self.y), k=X.shape[0])
            #np.full(X.shape[0], self.value)
        else:
            left_mask = X[:, self.feature_index] < self.threshold
            right_mask = ~left_mask

            y_left = self.left.predict(X[left_mask])
            y_right = self.right.predict(X[right_mask])

            result = np.empty(X.shape[0])
            result[left_mask] = y_left
            result[right_mask] = y_right

            return result
        
    def get_values_leaf(self, X, indexes):
        """
        Retrieve the samples and their target values for each leaf.
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        indexes : np.ndarray
            Array of sample indexes corresponding to the rows in X.
        Returns
        -------
        list
            A list of lists, where each sublist contains the sample indexes and target values for each leaf.
        """
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
        """
        Retrieve the samples, their target values, and group identifiers for each leaf up to a specified group depth.
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        indexes : np.ndarray
            Array of sample indexes corresponding to the rows in X.
        current_group_depth : str, optional
            Current group depth identifier (default is an empty string).
        max_depth_group : int, optional
            Maximum depth for group identifiers (default is 1).
        Returns
        -------
        list
            A list of lists, where each sublist contains the sample indexes, target values, and group identifier for each leaf.
        """
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
                y_left = self.left.get_values_leaf_and_groups(X[left_mask,:], idxleft, current_group_depth=left_group_depth, max_depth_group=max_depth_group)
            idxright = indexes[np.where(right_mask)[0].astype(int)]
            if len(idxright)!=0:
                y_right = self.right.get_values_leaf_and_groups(X[right_mask,:], idxright, current_group_depth=right_group_depth, max_depth_group=max_depth_group)

            return y_left + y_right