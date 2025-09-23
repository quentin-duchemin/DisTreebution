
import numpy as np
import random
from tqdm import tqdm

def get_quantile(ls, n, q):
    # ls: must be sorted ! 
    i_q = min(n-1,int(n*q))
    i_q_p = min(n-1, i_q+1)
    return ls[i_q] + (n*q-i_q)*(ls[i_q_p]-ls[i_q])


class Conformalisation():
    """
    A class for handling conformalisation processes on tree-based models, including preprocessing of tree outputs
    for both training and test data, optionally with group information, and performing operations on leaf values.
    Parameters
    ----------
    settings : dict, optional
        Configuration settings for the conformalisation process, such as the type of tree.
    params : dict, optional
        Additional parameters for the conformalisation process.
    Methods
    -------
    preprocess_trees(trees, x_train, x_test)
        Preprocesses the given trees to extract and organize leaf values for each sample in the training and test sets.
        Handles both standard and PQRT tree types.
    preprocess_trees_with_groups(trees, x_train, x_test, max_depth_group)
        Preprocesses the given trees to extract leaf values and group information for each sample in the training and test sets,
        up to a specified maximum group depth.
    operation_leaf(ls, q, weights=None)
        Computes the q-th quantile value from a sorted list of leaf values, optionally using sample weights.
    """
    def __init__(self, settings=None, params=None):
        """
        Initializes the Conformalisation object with optional settings and parameters.
        Parameters
        ----------
        settings : dict, optional
            Configuration settings for the conformalisation process.
        params : dict, optional
            Additional parameters for the conformalisation process.
        """
        self.settings = settings
        self.params = params
    
    def preprocess_trees(self, trees, x_train, x_test):
        """
        Preprocesses the given trees to extract and organize leaf values for each sample in the training and test sets.
        For 'PQRT' tree types, processes each quantile tree separately. For other types, processes each tree individually.
        Parameters
        ----------
        trees : list or list of lists
            The collection of tree objects to preprocess.
        x_train : np.ndarray
            Training data features.
        x_test : np.ndarray
            Test data features.
        Returns
        -------
        tuple
            Two dictionaries mapping tree and sample indices to sorted leaf values for training and test data, respectively.
        """
        if self.settings['type_tree']=='PQRT':
            treeID2trainID2values = {i_q:{} for i_q in range(len(trees))}
            treeID2testID2values = {i_q:{} for i_q in range(len(trees))}
            for i_q in range(len(trees)):
                for k, tree in enumerate(trees[i_q]):
                    treeID2trainID2values[i_q][k] = {}
                    on_leaf = tree.get_values_leaf(x_train, np.arange(0,x_train.shape[0], 1))
                    for leaf in on_leaf:
                        sample_idxs, yvalues = leaf[0], leaf[1]
                        for sample_idx in sample_idxs:
                            treeID2trainID2values[i_q][k][sample_idx] = np.sort(yvalues)

                    treeID2testID2values[i_q][k] = {}
                    on_leaf_test = tree.get_values_leaf(x_test, np.arange(0,x_test.shape[0], 1))
                    for leaf in on_leaf_test:
                        sample_idxs, yvalues = leaf[0], leaf[1]
                        for sample_idx in sample_idxs:
                            treeID2testID2values[i_q][k][sample_idx] = np.sort(yvalues)
            return treeID2trainID2values, treeID2testID2values
    
            
            
        else:
            treeID2trainID2values = {}
            treeID2testID2values = {}
            for k, tree in enumerate(trees):
                treeID2trainID2values[k] = {}
                on_leaf = tree.get_values_leaf(x_train, np.arange(0,x_train.shape[0], 1))
                for leaf in on_leaf:
                    sample_idxs, yvalues = leaf[0], leaf[1]
                    for sample_idx in sample_idxs:
                        treeID2trainID2values[k][sample_idx] = np.sort(yvalues)

                treeID2testID2values[k] = {}
                on_leaf_test = tree.get_values_leaf(x_test, np.arange(0,x_test.shape[0], 1))
                for leaf in on_leaf_test:
                    sample_idxs, yvalues = leaf[0], leaf[1]
                    for sample_idx in sample_idxs:
                        treeID2testID2values[k][sample_idx] = np.sort(yvalues)
            return treeID2trainID2values, treeID2testID2values

    
    def preprocess_trees_with_groups(self, trees, x_train, x_test, max_depth_group):
        """
        Preprocesses the given trees to extract leaf values and group information for each sample in the training and test sets.
        Parameters
        ----------
        trees : list
            The collection of tree objects to preprocess.
        x_train : np.ndarray
            Training data features.
        x_test : np.ndarray
            Test data features.
        max_depth_group : int
            The maximum depth for grouping samples within the trees.
        Returns
        -------
        tuple
            Four dictionaries mapping tree and sample indices to sorted leaf values and group identifiers for training and test data.
        """
        treeID2trainID2values = {}
        treeID2testID2values = {}
        treeID2trainID2group = {}
        treeID2testID2group = {}
        for k, tree in enumerate(trees):
            treeID2trainID2values[k] = {}
            treeID2trainID2group[k] = {}
            on_leaf = tree.get_values_leaf_and_groups(x_train, np.arange(0,x_train.shape[0], 1), current_group_depth='', max_depth_group=max_depth_group)
            for leaf in on_leaf:
                sample_idxs, yvalues, group = leaf[0], leaf[1], leaf[2]
                for sample_idx in sample_idxs:
                    treeID2trainID2values[k][sample_idx] = np.sort(yvalues)
                    treeID2trainID2group[k][sample_idx] = group

            treeID2testID2values[k] = {}
            treeID2testID2group[k] = {}
            on_leaf_test = tree.get_values_leaf_and_groups(x_test, np.arange(0,x_test.shape[0], 1), current_group_depth='', max_depth_group=max_depth_group)
            for leaf in on_leaf_test:
                sample_idxs, yvalues, group = leaf[0], leaf[1], leaf[2]
                for sample_idx in sample_idxs:
                    treeID2testID2values[k][sample_idx] = np.sort(yvalues)
                    treeID2testID2group[k][sample_idx] = group

        return treeID2trainID2values, treeID2testID2values, treeID2trainID2group, treeID2testID2group
    
    def operation_leaf(self, ls, q, weights=None):
        """
        Computes the q-th quantile value from a sorted list of leaf values, optionally using sample weights.
        Parameters
        ----------
        ls : array-like
            Sorted list or array of leaf values.
        q : float
            Quantile to compute, between 0 and 1.
        weights : array-like, optional
            Weights for each value in `ls`. If None, uniform weights are used.
        Returns
        -------
        float
            The q-th quantile value from the list.
        """
        n = len(ls)
        if weights is None:
            weights = np.ones(n)
        cumulative_weights = np.cumsum(weights)
        total_weight = cumulative_weights[-1]
        i = np.argmax(cumulative_weights >= q * total_weight)
        return ls[i]