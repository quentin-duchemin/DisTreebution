
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
    A class for handling conformalisation processes on tree-based models, including preprocessing of trees
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
        self.conf_thresh = None
        self.group2conf_thresh = None

        if params.get('treeID2quantiles_train', None) is not None:
            self.treeID2quantiles_train = params['treeID2quantiles_train']
        else:
            self.treeID2quantiles_train = {}

        if settings.get('type_tree', None) in ['CRPS','RT']:
            self.treeID2quantiles_query = {ID: [0.01 * i for i in range(1, 100)] for ID in range(params['nTrees'])}
            self.quantiles_query = np.array([0])
            self.quantile_query2treeIDs = {0: [i for i in range(params['nTrees'])]}
        elif params.get('treeID2quantiles_query', None) is not None:
            self.treeID2quantiles_query = params['treeID2quantiles_query']
        elif settings.get('type_tree', None) != 'RT':
            self.treeID2quantiles_query = params['treeID2quantiles_train']

        if not(settings.get('type_tree', None) in ['CRPS','RT']):
            self.quantile_train2treeIDs, self.quantile_query2treeIDs = {}, {}
            for k, ls_q in self.treeID2quantiles_train.items():
                for q in ls_q:
                    q_rounded = np.round(q, 3)
                    if q_rounded not in self.quantile_train2treeIDs:
                        self.quantile_train2treeIDs[q_rounded] = []
                    self.quantile_train2treeIDs[q_rounded].append(k)
            self.quantiles_train = np.sort(list(self.quantile_train2treeIDs.keys()))
            for k, ls_q in self.treeID2quantiles_query.items():
                for q in ls_q:
                    q_rounded = np.round(q, 3)
                    if q_rounded not in self.quantile_query2treeIDs:
                        self.quantile_query2treeIDs[q_rounded] = []
                    self.quantile_query2treeIDs[q_rounded].append(k)
            self.quantiles_query = np.sort(list(self.quantile_query2treeIDs.keys()))

        if settings.get('nested_set', None) == 'distributional':
            print('WARNING: using default list of low quantiles for distributional conformal prediction.')
            params['list_distri_low_quantiles'] = params.get('list_distri_low_quantiles', [0.01 * i for i in range(1, 35)])
    
    def preprocess_trees(self, trees, x):
        """
        Preprocesses the given trees to extract and organize leaf values for each sample in the dataset.
        Handles both standard and PQRT tree types.
        Parameters
        ----------
        trees : list
            List of tree objects to preprocess.
        x : array-like
            Input data for which to extract leaf values.
        Returns
        -------
        treeID2sampleID2values : dict
            A dictionary mapping tree IDs to another dictionary that maps sample IDs to sorted leaf values.
        """
        treeID2sampleID2values = {}
        for k, tree in enumerate(trees):
            treeID2sampleID2values[k] = {}
            on_leaf = tree.get_values_leaf(x, np.arange(0,x.shape[0], 1))
            for leaf in on_leaf:
                sample_idxs, yvalues = leaf[0], leaf[1]
                for sample_idx in sample_idxs:
                    treeID2sampleID2values[k][sample_idx] = np.sort(yvalues)
        return treeID2sampleID2values

    
    def preprocess_trees_with_groups(self, trees, x_train, max_depth_group):
        """
        Preprocesses the given trees to extract leaf values and group information for each sample in the training set,
        up to a specified maximum group depth.
        Parameters
        ----------
        trees : list
            List of tree objects to preprocess.
        x_train : array-like
            Training input data for which to extract leaf values and group information.
        max_depth_group : int
            Maximum depth for group coverage in the trees.
        Returns
        -------
        treeID2trainID2values : dict
            A dictionary mapping tree IDs to another dictionary that maps training sample IDs to sorted leaf values.
        treeID2trainID2group : dict
            A dictionary mapping tree IDs to another dictionary that maps training sample IDs to their group information.
        """
        treeID2trainID2values = {}
        treeID2trainID2group = {}
        for k, tree in enumerate(trees):
            treeID2trainID2values[k] = {}
            treeID2trainID2group[k] = {}
            on_leaf = tree.get_values_leaf_and_groups(x_train, np.arange(0,x_train.shape[0], 1), current_group_depth='', max_depth_group=max_depth_group)
            for leaf in on_leaf:
                sample_idxs, yvalues, group = leaf[0], leaf[1], leaf[2]
                for sample_idx in sample_idxs:
                    treeID2trainID2values[k][sample_idx] = np.sort(yvalues)
                    treeID2trainID2group[k][sample_idx] = group

        return treeID2trainID2values, treeID2trainID2group
    
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