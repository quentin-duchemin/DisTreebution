
import numpy as np
import random
from QRT.entropies_MultiQuantiles import entropies_MultiQuantiles
from QRT.RegressionTreeQuantile import RegressionTreeQuantile
from QRT.RegressionTreeQuadratic import RegressionTreeQuadratic
from tqdm import tqdm

from CRPStreeC.RegressionTree import RegressionTree
from CRPStreeC.entropies_CRPS import entropies_CRPS
from CRPStreeC.compute_CRPS import compute_CRPS

from UQ.Conformalisation_CQR import Conformalisation_CQR
from UQ.Conformalisation_CQR_PQRT import Conformalisation_CQR_PQRT
from UQ.Conformalisation_distri_PQRT import Conformalisation_distri_PQRT

from UQ.Conformalisation_distri import Conformalisation_distri
from UQ.Conformalisation import Conformalisation


available_settings = {}

# Possible choices of regression trees 
available_settings['type_tree'] = ['RT','PQRT','PMQRT','CRPS']

# Possible choices of nested sets
available_settings['nested_set'] = ['CQR', 'CQR-m', 'CQR-r', 'distributional']

# Possible choices of conformal framework
available_settings['type_conformal'] = ['no-conformalisation-vr','no-conformalisation-vr-avg','split-vr','split-vr-avg']
available_settings['ope_in_leaves'] = ['standard', 'robust']

# Currently, only symmetric uncertainty intervals are implemented
available_settings['symmetry'] = ['symmetric']

# Possibility to use group coverage
available_settings['group_coverage'] = [False, True]

# Hyperparameters to train the regression trees
all_params = ['nTrees', 'train_quantiles', 'max_depth', 'use_LOO', 'min_samples_split', 'nominal_quantiles', 'max_depth_group', 'list_distri_low_quantiles']


"""
The type of regression tree to use. Options include:
    - 'PQRT': Quantile Regression Tree (one tree per quantile).
    - 'PMQRT': Multi-Quantile Regression Tree (single tree for multiple quantiles).
    - 'CRPS': Regression tree optimized for Continuous Ranked Probability Score.
    - 'RT': Standard regression tree with quadratic loss.
The type of nested set or conformalization method to use. Options include:
    - 'CQR': Conformalized Quantile Regression.
    - 'CQR-m': Modified CQR.
    - 'CQR-r': Robust CQR.
    - 'distributional': Distributional conformalization.
The type of conformalization operation to use in leaves. Options include:
    - 'no-conformalisation-vr': No conformalization, vanilla regression.
    - 'no-conformalisation-vr-avg': No conformalization, average over trees.
    - 'split-vr': Split conformalization, vanilla regression.
    - 'split-vr-avg': Split conformalization, average over trees.
The operation to perform in the leaves of the regression trees. Options:
    - 'standard': Standard operation.
    - 'robust': Robust operation.
Whether to use group coverage in conformalization (i.e., enforce coverage for subgroups).
Additional parameters for tree construction and conformalization. Possible keys include:
    - 'nTrees': int, number of trees in the ensemble.
    - 'train_quantiles': array-like, quantiles to train for (used in PQRT/PMQRT).
    - 'max_depth': int, maximum depth of the trees.
    - 'use_LOO': bool, whether to use leave-one-out estimation in to compute the information gains.
    - 'min_samples_split': int, minimum samples required to split a node.
    - 'nominal_quantiles': array-like, quantiles for prediction sets.
    - 'max_depth_group': int, maximum depth for group coverage trees.
    - 'list_distri_low_quantiles': array-like, lower quantiles for distributional methods.
    - 'limit_use_CRPS': int, limit for CRPS optimization (used in CRPS trees).
"""


class UQ():
    """
    Uncertainty Quantification (UQ) class for conformal prediction using various regression tree-based methods.
    This class provides an interface for training ensembles of regression trees, applying conformalization techniques,
    and evaluating prediction sets for uncertainty quantification in regression tasks. It supports multiple tree types
    (e.g., PQRT, PMQRT, CRPS, RT) and conformalization methods (e.g., CQR, distributional, standard).
    Parameters
    ----------
    type_tree : str, default='PQRT'
        The type of regression tree to use. Options include 'PQRT', 'PMQRT', 'CRPS', 'RT'.
    nested_set : str or None, default=None
        The type of nested set or conformalization method to use (e.g., 'CQR', 'distri').
    type_conformal : str or None, default=None
        The type of conformalization operation to use in leaves (e.g., 'vr', 'vr-avg', 'vr-trimmed').
    ope_in_leaves : str, default='standard'
        The operation to perform in the leaves of the regression trees.
    group_coverage : bool, default=False
        Whether to use group coverage in conformalization.
    params : dict or None, default=None
        Additional parameters for tree construction and conformalization.
    Attributes
    ----------
    settings : dict
        Stores configuration settings for the UQ instance.
    params : dict
        Stores parameters for tree and conformalization methods.
    conf : object
        The conformalization object corresponding to the chosen method.
    Methods
    -------
    compute_width_coverage(sample2predset, y_test)
        Computes the width and coverage of prediction sets for each test sample.
    train_trees(x_train, y_train)
        Trains an ensemble of regression trees according to the specified type and parameters.
    conformal_split(x_train, y_train, x_test, y_test, x_calib, y_calib, alpha)
        Applies conformalization to obtain prediction sets and computes their widths and coverages.
    get_quantile_estimate(x_train, y_train, x_test, trees=None, y_test=None)
        Estimates quantiles for test samples using the trained trees.
    get_quantile_estimate_i(i, q, treeID2testID2values)
        Computes the quantile estimate for a single test sample using the specified conformalization method.
    """
    def __init__(self, type_tree='PQRT', nested_set=None, type_conformal=None, ope_in_leaves='standard', group_coverage=False, params=None):
        self.settings = {}
        self.settings['type_tree'] = type_tree
        self.settings['nested_set'] = nested_set
        self.settings['type_conformal'] = type_conformal
        self.settings['ope_in_leaves'] = ope_in_leaves
        self.settings['group_coverage'] = group_coverage
        self.params = params
        try:
            self.params['train_quantiles'] = np.array(self.params['train_quantiles'])
        except:
            pass
        
        if nested_set is None:
            self.conf = Conformalisation(settings=self.settings, params=params) 
        elif 'CQR' in nested_set:
            if type_tree=='PQRT':
                self.conf = Conformalisation_CQR_PQRT(settings=self.settings, params=params)
            else:
                self.conf = Conformalisation_CQR(settings=self.settings, params=params)
        elif 'distri' in nested_set:
            if type_tree=='PQRT':
                self.conf = Conformalisation_distri_PQRT(settings=self.settings, params=params)
            else:
                self.conf = Conformalisation_distri(settings=self.settings, params=params)
        else:
            assert False, "No conformalisation method found for the given nested set"
            
    def compute_width_coverage(self, sample2predset, y_test):
        """
        Computes the total width of prediction sets and coverage indicators for each test sample.

        For each test sample, this method calculates:
          - The sum of the widths of all predicted intervals (sub-intervals) associated with the sample.
          - Whether the true value falls within any of the predicted intervals (coverage).

        Args:
            sample2predset (list of list of tuple): 
                A list where each element corresponds to a test sample and contains a list of predicted intervals (tuples of (lower_bound, upper_bound)).
            y_test (array-like): 
                The true target values for the test samples.

        Returns:
            widths (np.ndarray): 
                Array of total widths of the predicted intervals for each test sample.
            coverages (np.ndarray): 
                Array of binary indicators (1 if the true value is covered by any interval, 0 otherwise) for each test sample.
        """
        widths, coverages = np.zeros(len(y_test)), np.zeros(len(y_test))
        for j,yj in enumerate(y_test):
            MC = False
            width = 0
            for subint in sample2predset[j]:
                width += subint[1]-subint[0]
                MC = MC or ((yj >= subint[0]) and (yj <= subint[1]))
            widths[j] = width
            coverages[j] = 1.* MC
        return widths, coverages

    
    def train_trees(self, x_train, y_train, ref_tree=None, max_depth_ref_tree=-1):
        """
        Trains an ensemble of regression trees according to the specified tree type and parameters.

        Depending on the 'type_tree' setting, this method supports different tree types:
            - 'PQRT': Trains a separate quantile regression tree for each quantile in 'train_quantiles'.
            - 'PMQRT': Trains a single regression tree for multiple quantiles.
            - 'CRPS': Trains a regression tree optimized for the Continuous Ranked Probability Score.
            - 'RT': Trains a standard regression tree with quadratic loss.

        For each tree, a random 60% subsample of the training data is used for fitting, and the remaining 40% is used for calibration.
        If only one tree is trained, the entire dataset is used for training.

        Args:
            x_train (np.ndarray): Training features of shape (n_samples, n_features).
            y_train (np.ndarray): Training targets of shape (n_samples,).

        Returns:
            trees (list or dict): 
                - If 'type_tree' is 'PQRT', returns a dictionary mapping quantile indices to lists of trained RegressionTreeQuantile objects.
                - Otherwise, returns a list of trained tree objects (RegressionTreeQuantile, RegressionTree, or RegressionTreeQuadratic).
            sample2calib_trees (dict): 
                Dictionary mapping each sample index in the training set to a list of tree indices for which the sample was used as a calibration point.
        """
            

        nTrees = self.params['nTrees']
        if self.settings['type_tree']=='PQRT':
            trees = {i_q: [] for i_q, q in enumerate(self.params['train_quantiles'])}
        else:
            trees = []
        ntrain = x_train.shape[0]
        sample2calib_trees = {i:[] for i in range(len(y_train))}
        indexes = np.arange(0,ntrain,1)
        for l in tqdm(range(nTrees)):
            random.shuffle(indexes)
            ntrain_tree = int(0.6*ntrain)
            if nTrees == 1:
                ntrain_tree = ntrain
            index_train, index_calib = indexes[:ntrain_tree], indexes[ntrain_tree:]
            if self.settings['type_tree']=='PQRT':
                for i_q, q in enumerate(self.params['train_quantiles']):
                    tree = RegressionTreeQuantile(max_depth=self.params['max_depth'], 
                                                  min_samples_split=self.params['min_samples_split'], 
                                                  quantiles=[q], use_LOO=self.params.get('use_LOO', True))
                    tree.fit(x_train[index_train,:], y_train[index_train])
                    trees[i_q].append(tree)
            else:
                if self.settings['type_tree']=='PMQRT':
                    tree = RegressionTreeQuantile(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], quantiles=self.params['train_quantiles'], use_LOO=self.params.get('use_LOO', True))
                elif self.settings['type_tree']=='CRPS':
                    tree = RegressionTree(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], limit_use_CRPS=self.params.get('limit_use_CRPS', None), quantiles=self.params.get('train_quantiles', None), use_LOO=self.params.get('use_LOO', True))
                elif self.settings['type_tree']=='RT':
                    tree = RegressionTreeQuadratic(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], use_LOO=self.params.get('use_LOO', True))
                tree.fit(x_train[index_train,:], y_train[index_train], ref_tree=ref_tree, max_depth_ref_tree=max_depth_ref_tree)
                trees.append(tree)
            for icalib in index_calib:
                sample2calib_trees[icalib].append(l)
        return trees, sample2calib_trees
    
    def conformal_split(self, x_train, y_train, x_test, y_test, x_calib, y_calib, alpha):
        trees, _ = self.train_trees(x_train, y_train)
            
        if self.settings['group_coverage']:
            interval_JK, sample2predset = self.conf.get_conformal_set_split_group_coverage(trees, x_test, y_test, x_calib, y_calib, self.params['max_depth_group'], alpha)
        else:
            interval_JK, sample2predset = self.conf.get_conformal_set_split(trees, x_test, y_test, x_calib, y_calib, alpha)
                
        if 'CQR' in self.settings['nested_set']:
            low_quantiles = self.params['nominal_quantiles']
            widths = {i_q:np.zeros(x_test.shape[0]) for i_q in range(len(low_quantiles))}
            coverages = {i_q:np.zeros(x_test.shape[0]) for i_q in range(len(low_quantiles))}
            for i_q in range(len(low_quantiles)):
                widths[i_q], coverages[i_q] = self.compute_width_coverage(sample2predset[i_q], y_test)
        else:
            widths, coverages = self.compute_width_coverage(sample2predset, y_test)
        return interval_JK, sample2predset, widths, coverages
    
    
    def get_quantile_estimate(self, x_train, y_train, x_test, quantiles=None, trees=None, y_test=None):
        """
        Estimates quantiles for the test set using trained trees and returns predicted quantiles for each sample.

        Parameters
        ----------
        x_train : np.ndarray
            Training feature matrix of shape (n_samples_train, n_features).
        y_train : np.ndarray
            Training target vector of shape (n_samples_train,).
        x_test : np.ndarray
            Test feature matrix of shape (n_samples_test, n_features).
        trees : list, optional
            List of pre-trained tree models. If None, trees will be trained using x_train and y_train.
        y_test : np.ndarray, optional
            True target values for the test set, used to compute marginal coverage levels.

        Returns
        -------
        sample2predset : dict
            Dictionary mapping quantile indices to arrays of predicted quantile values for each test sample.
        marginal_levels : np.ndarray, optional
            Array of marginal coverage levels for each quantile, returned only if y_test is provided.

        Notes
        -----
        - The method supports different types of trees, including 'PQRT'.
        - If `y_test` is provided, the method also computes the empirical coverage of the predicted quantiles.
        """
        if quantiles is None:
            quantiles = self.params['nominal_quantiles']
        sample2predset = {i_q:np.zeros(x_test.shape[0]) for i_q in range(len(quantiles))}
        if trees is None:
            trees, _ = self.train_trees(x_train, y_train)
        treeID2calibID2values, treeID2testID2values = self.conf.preprocess_trees(trees, np.zeros((1,x_test.shape[1])), x_test)
        for i_q, q in enumerate(quantiles):
            if self.settings['type_tree']=='PQRT':
                i_q = np.argmin(np.abs(q-self.params['train_quantiles']))
                for i in range(x_test.shape[0]):
                    qhat = self.get_quantile_estimate_i(i, q, treeID2testID2values[i_q])
                    sample2predset[i_q][i] = qhat
            else:
                for i in range(x_test.shape[0]):
                    qhat = self.get_quantile_estimate_i(i, q, treeID2testID2values)
                    sample2predset[i_q][i] = qhat
        if y_test is None:
            return sample2predset
        else:
            marginal_levels = np.zeros(len(quantiles))
            for i_q in range(len(quantiles)):
                marginal_levels[i_q] = np.mean(y_test <= sample2predset[i_q])
            return sample2predset, marginal_levels

    def get_quantile_estimate_i(self, i, q, treeID2testID2values):
        """
        Estimate the q-th quantile for a specific test sample using different conformal prediction strategies.
        Parameters
        ----------
        i : int
            Index of the test sample for which the quantile estimate is computed.
        q : float
            Quantile level to estimate (e.g., 0.5 for median).
        treeID2testID2values : list of dict or list of list
            A nested structure where each element corresponds to a tree, and for each tree, 
            the values for each test sample are provided (e.g., treeID2testID2values[tree][test_sample_index]).
        Returns
        -------
        qhat_test : float
            Estimated q-th quantile for the i-th test sample, computed according to the conformal prediction 
            strategy specified in `self.settings['type_conformal']`.
        Notes
        -----
        The method supports several conformal prediction strategies:
            - 'vr-avg': Averages quantile estimates across trees using weighted leaves.
            - 'vr-trimmed': Computes quantile estimates per tree, sorts them, and averages the central 80%.
            - 'vr': Averages quantile estimates across trees without weighting or trimming.
        The actual quantile computation per leaf is delegated to `self.conf.operation_leaf`.
        """
        # i: index of the test sample
        if 'vr-avg' in self.settings['type_conformal']:
            weights_test = []
            ls_leaves_test = []
            for k in range(len(treeID2testID2values)):
                ls_new = list(treeID2testID2values[k][i])
                ls_leaves_test = ls_leaves_test + ls_new
                weights_test = weights_test + list(np.ones(len(ls_new))/len(ls_new))

            ls_leaves_test_unique, indices = np.unique(ls_leaves_test, return_inverse=True)
            summed_weights_test = np.bincount(indices, weights_test)


            order_unique_test = np.argsort(ls_leaves_test_unique)
            ls_leaves_test_unique = ls_leaves_test_unique[order_unique_test]
            summed_weights_test = summed_weights_test[order_unique_test]
            qhat_test = self.conf.operation_leaf(ls_leaves_test_unique, q, weights=summed_weights_test)

        elif 'vr-trimmed' in self.settings['type_conformal']:
            ls_leaves_test = []
            ls_qhat_test = []
            count = 0
            for k in range(len(treeID2testID2values)):
                ls_leaves_test = list(treeID2testID2values[k][i])
                ls_leaves_test = np.sort(ls_leaves_test)
                ls_qhat_test.append(self.conf.operation_leaf(ls_leaves_test, q))

            ls_qhat_test = np.sort(ls_qhat_test)
            
            qhat_test = np.mean(ls_qhat_test[int(0.1*len(ls_qhat_test)):int(0.9*len(ls_qhat_test))])

        elif 'vr' in self.settings['type_conformal']:
            ls_leaves_test = []
            qhat_test = 0
            count = 0
            for k in range(len(treeID2testID2values)):
                ls_leaves_test = list(treeID2testID2values[k][i])
                ls_leaves_test = np.sort(ls_leaves_test)
                qhat_test += self.conf.operation_leaf(ls_leaves_test, q)
                count += 1

            qhat_test /= count

        return qhat_test
