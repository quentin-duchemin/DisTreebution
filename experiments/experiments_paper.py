import numpy as np
import os
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from load_dataset import load_dataset

sys.path.append('../')
from UQ.UQ import UQ

# Set the root directory and folder names
root = '/mydata/watres/quentin/PinballRT/EXPE_FINAL/'

datasets = ['abalone', 'gas_turbine', 'combined_cycle_power_plant', 'red_wine', 'white_wine']

nbites = 300
ntrain = 1000

ls_ites= [i for i in range(1,300)]
def process_dataset(name_dataset):
    for ite in ls_ites:
        np.random.seed(ite)
        random.seed(ite)
        X, y = load_dataset(name_dataset)
        print(f'Processing {name_dataset} - ITE {ite}')
        ls_q = [0.1 * i for i in range(1, 10)]
        ls_q_get = [0.05 * i for i in range(1, 20)]
        N = X.shape[0]

        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        x_train = X[idxs[:N][:ntrain], :]
        y_train = y[idxs[:N][:ntrain]]
        x_test = X[idxs[:N][ntrain:], :]
        y_test = y[idxs[:N][ntrain:]]

        

                
        from sklearn.linear_model import QuantileRegressor
        marginal_level = np.zeros(len(ls_q_get))
        sample2predset = {i_q: np.zeros(len(y_test)) for i_q in range(len(ls_q_get))}
        for i_q, q in enumerate(ls_q_get):
            model = QuantileRegressor(quantile=q, alpha=0)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            marginal_level[i_q] = np.mean(y_test <= predictions)
            sample2predset[i_q] = predictions
        with open(root + f'sklearn_linear/{ntrain}_{name_dataset}_{ite}_sample2predset.pkl', 'wb') as handle:
            pickle.dump(sample2predset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(root + f'sklearn_linear_new/{ntrain}_{name_dataset}_{ite}_marginal_levels.npy', marginal_level)
        np.save(root + f'sklearn_linear_new/{ntrain}_{name_dataset}_{ite}_y_test.npy', y_test)

        if True:
            model_configs = [
                    (50, 'QRF', 'qrf_new', 'RT', 'no-conformalisation-vr', None, False),
                    (50, 'CRPS-RT', 'crpsrt_new', 'CRPS', 'no-conformalisation-vr', None, False),
                    (50, 'pmqrt_01', 'pmqrt_01_new', 'PMQRT', 'no-conformalisation-vr', None, False),
                    (50, 'pmqrt_05', 'pmqrt_05_new', 'PMQRT', 'no-conformalisation-vr', None, False)

            ]
    
            for nTrees, model_name, folder_name, type_tree, type_conformal, nested_set, use_alpha in model_configs:
                print(f'{model_name}')
                alpha = 0.1
                train_quantiles = ls_q_get if 'pmqrt_05' not in model_name else ls_q
                params = {'nTrees': nTrees, 'max_depth': 10, 'min_samples_split': 10 if 'VR' in model_name else 20, 
                          'train_quantiles': train_quantiles, 'nominal_quantiles': ls_q_get, 'use_LOO':True}
    
                model = UQ(type_tree=type_tree, nested_set=nested_set, type_conformal=type_conformal, ope_in_leaves='standard', params=params)
                if use_alpha:
                    interval, sample2predset, widths, coverages = model.conformal_oob(x_train, y_train, x_test, y_test, alpha)
                    with open(root + f'{folder_name}/{ntrain}_{name_dataset}_{ite}_widths.pkl', 'wb') as handle:
                        pickle.dump(widths, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    with open(root + f'{folder_name}/{ntrain}_{name_dataset}_{ite}_coverages.pkl', 'wb') as handle:
                        pickle.dump(coverages, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    sample2predset, marginal_levels = model.get_quantile_estimate(x_train, y_train, x_test, y_test=y_test)
                    np.save(root + f'{folder_name}/{ntrain}_{name_dataset}_{ite}_marginal_levels.npy', marginal_levels)

                np.save(root + f'{folder_name}/{ntrain}_{name_dataset}_{ite}_y_test.npy', y_test)

    
                with open(root + f'{folder_name}/{ntrain}_{name_dataset}_{ite}_params.pkl', 'wb') as handle:
                    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(root + f'{folder_name}/{ntrain}_{name_dataset}_{ite}_sample2predset.pkl', 'wb') as handle:
                    pickle.dump(sample2predset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        

# Run the process in parallel
with ProcessPoolExecutor() as executor:
    executor.map(process_dataset, datasets)
