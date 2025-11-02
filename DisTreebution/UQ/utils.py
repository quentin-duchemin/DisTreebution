import numpy as np
import pandas as pd
import os

def filter_dict(dict, keys_to_keep):
    if dict == {}:
        return {}
    else:
        return {k: dict.get(k, 0) for k in keys_to_keep}

def crps_from_quantiles(quantiles, tau, y):
    """
    Compute CRPS from predicted quantiles.
    
    Parameters:
    - quantiles: List or numpy array of predicted quantiles.
    - tau: List or numpy array of corresponding quantile levels (e.g., [0.1, 0.2, ..., 0.9]).
    - y: The observed value.
    
    Returns:
    - crps: The computed CRPS score.
    """
    quantiles = np.array(quantiles)
    tau = np.array(tau)
    
    # Compute the absolute errors
    errors = np.abs(quantiles - y)
    
    # Compute the weights as differences in tau levels
    weights = np.diff(np.insert(tau, 0, 0))  # Add 0 at the beginning for correct differences
    
    # Compute weighted sum
    crps = np.sum(weights * errors)
    
    return crps


def compute_crps(sample2predset, y_test, ls_q):
    res = 0
    for i, ytesti in enumerate(y_test):
        res += crps_from_quantiles([sample2predset[i][iq] for iq in range(len(ls_q))], ls_q, ytesti) / len(y_test)
    return res

def load_dataset(root_path, dataset):
    if dataset == 'abalone':
        # Predicting the age of abalone from physical measurements.
        names = ['Sex', 'Length', 'Diameter',
                 'Height',
                 'Whole weight',
                 'Shucked weight',
                 'Viscera weight',
                 'Shell weight',
                 'Rings'
                ]
        df = pd.read_csv(os.path.join(root_path, 'abalone/abalone.data'), header=None)
        df.columns = names
        df['Sex'] = df['Sex'].apply(lambda x: 0 if x=='M' else 1) 
        X = df.loc[:, df.columns != 'Rings'].to_numpy()
        y = df.loc[:, 'Rings'].to_numpy()
    elif dataset == 'gpu':
        df = pd.read_csv(os.path.join(root_path, 'gpu_kernel_performance/sgemm_product.csv'), sep=',')
        y = df.loc[:, [col for col in df.columns if 'Run' in col]].mean(axis=1).to_numpy().astype(float)
        X = df.loc[:, [col for col in df.columns if not('Run' in col)]].to_numpy().astype(float)
    elif dataset == 'gas_turbine':
        # The dataset can be well used for predicting turbine energy yield (TEY) using ambient variables as features.
        df = pd.read_csv(os.path.join(root_path, 'gas_turbine/gt_2015.csv'), sep=',')
        y = df.loc[:, 'TEY'].to_numpy()
        X = df.loc[:, df.columns != 'TEY'].to_numpy()
    elif dataset == 'combined_cycle_power_plant':
        # Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP)  of the plant.
        df = pd.read_csv(os.path.join(root_path, 'combined_cycle_power_plant/Folds5x2_pp.csv'))
        y = df.loc[:, 'PE'].to_numpy()
        X = df.loc[:, df.columns != 'PE'].to_numpy()
    elif dataset == 'red_wine':
        # Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP)  of the plant.
        df = pd.read_csv(os.path.join(root_path, 'wine/winequality-red.csv'), sep=';')
        y = df.loc[:, 'quality'].to_numpy()
        X = df.loc[:, df.columns != 'quality'].to_numpy()
    elif dataset == 'white_wine':
        # Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP)  of the plant.
        df = pd.read_csv(os.path.join(root_path, 'wine/winequality-white.csv'), sep=';')
        y = df.loc[:, 'quality'].to_numpy()
        X = df.loc[:, df.columns != 'quality'].to_numpy()
    return X, y