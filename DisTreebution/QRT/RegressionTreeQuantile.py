import numpy as np
from .entropies_MultiQuantiles import entropies_MultiQuantiles
import matplotlib.pyplot as plt

class RegressionTreeQuantile:
    """Regression tree that predicts multiple quantiles.

    This tree recursively partitions the feature space to minimize a
    multiquantile entropy measure. It produces predictions for a list of
    requested quantiles at each leaf.

    :param quantiles: Sequence of quantile levels to predict (values in (0,1)).
    :type quantiles: list or numpy.ndarray
    :param max_depth: Maximum recursion depth for the tree. If ``None``, splitting continues until the sample-size criterion is met.
    :type max_depth: int or None
    :param min_samples_split: Minimum number of samples required in a node to consider a split.
    :type min_samples_split: int
    :param use_LOO: Whether to use leave-one-out adjustments in entropy computations.
    :type use_LOO: bool
    """

    def __init__(self, quantiles, max_depth=None, min_samples_split=2, use_LOO=True):
        """Initialize a :class:`RegressionTreeQuantile` instance.

        :param quantiles: See class description.
        :type quantiles: list or numpy.ndarray
        :param max_depth: See class description.
        :type max_depth: int or None
        :param min_samples_split: See class description.
        :type min_samples_split: int
        :param use_LOO: See class description.
        :type use_LOO: bool
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.quantiles = quantiles
        self.use_LOO = use_LOO

    def fit(self, X, y, depth=0, ref_tree=None, max_depth_ref_tree=-1):
        """Fit the regression-quantile tree to data.

        This method recursively grows the tree. If a split is found it sets
        ``self.feature_index`` and ``self.threshold`` and creates ``left`` and
        ``right`` child nodes. Otherwise the node becomes a leaf and stores
        the observed ``y`` values in ``self.y``.

        :param X: Feature matrix of shape (n_samples, n_features).
        :type X: numpy.ndarray
        :param y: Target vector of shape (n_samples,).
        :type y: numpy.ndarray
        :param depth: Current depth in recursion (used internally).
        :type depth: int
        :param ref_tree: Optional reference tree whose splits will be reused up to ``max_depth_ref_tree``.
        :type ref_tree: RegressionTreeQuantile or None
        :param max_depth_ref_tree: Maximum depth on ``ref_tree`` to reuse splits from.
        :type max_depth_ref_tree: int
        :returns: None. Tree structure is constructed in-place.
        :rtype: None
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
            

            self.left = RegressionTreeQuantile(self.quantiles, max_depth=self.max_depth, min_samples_split=self.min_samples_split, use_LOO=self.use_LOO)
            self.right = RegressionTreeQuantile(self.quantiles, max_depth=self.max_depth, min_samples_split=self.min_samples_split, use_LOO=self.use_LOO)

            self.left.fit(X[left_mask,:], y[left_mask], depth + 1, ref_tree=ref_tree_left, max_depth_ref_tree=max_depth_ref_tree)
            self.right.fit(X[right_mask,:], y[right_mask], depth + 1, ref_tree=ref_tree_right, max_depth_ref_tree=max_depth_ref_tree)
        else:
            self.y = y
            return
            #self.value = [np.sort(y)[min(int(q*len(y)),len(y)-1)] for q in self.quantiles]

    def find_best_split(self, X, y):
        """Find the best split for the current node.

        The method evaluates candidate splits on every feature and returns the
        best (feature_index, threshold) pair that satisfies the minimum
        samples per side criterion and improves the multiquantile entropy
        (when ``self.use_LOO`` is True the entropy uses leave-one-out
        adjustments).

        :param X: Feature matrix for current node.
        :type X: numpy.ndarray
        :param y: Target values for current node.
        :type y: numpy.ndarray
        :returns: Tuple ``(feature_index, threshold)`` for the best split, or ``None`` if no valid split found.
        :rtype: tuple(int, float) or None
        """
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
        """Predict quantiles for input samples.

        If the node is a leaf the stored quantile values (``self.value``) are
        repeated for each input sample. Otherwise the method routes samples
        down to child nodes and assembles per-sample quantile predictions.

        :param X: Feature matrix of shape (n_samples, n_features).
        :type X: numpy.ndarray
        :returns: Array of shape (n_samples, n_quantiles) with predicted quantile values.
        :rtype: numpy.ndarray
        """
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
        """Return leaf values and associated sample indexes.

        Traverses the tree and collects, for each leaf, a pair containing the
        list of sample indexes that fall in the leaf and the observed target
        values stored at that leaf.

        :param X: Feature matrix of shape (n_samples, n_features).
        :type X: numpy.ndarray
        :param indexes: Array of original sample indices corresponding to rows in ``X``.
        :type indexes: numpy.ndarray or list
        :returns: List of items ``[indexes_list, y_values]`` for each leaf encountered.
        :rtype: list
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
        """Return leaf values together with group identifiers.

        Similar to :meth:`get_values_leaf` but also returns a group identifier
        (a string encoding the path down the tree) for each leaf. Groups are
        truncated at ``max_depth_group``.

        :param X: Feature matrix of shape (n_samples, n_features).
        :type X: numpy.ndarray
        :param indexes: Array of original sample indices corresponding to rows in ``X``.
        :type indexes: numpy.ndarray or list
        :param current_group_depth: Current group identifier string (used during recursion).
        :type current_group_depth: str
        :param max_depth_group: Maximum length of the group identifier to produce.
        :type max_depth_group: int
        :returns: List of items ``[indexes_list, y_values, group_id]`` for each leaf.
        :rtype: list
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
                y_left = self.left.get_values_leaf_and_groups(X[left_mask], idxleft, current_group_depth=left_group_depth, max_depth_group=max_depth_group)
            idxright = indexes[np.where(right_mask)[0].astype(int)]
            if len(idxright)!=0:
                y_right = self.right.get_values_leaf_and_groups(X[right_mask], idxright, current_group_depth=right_group_depth, max_depth_group=max_depth_group)

            return y_left + y_right
