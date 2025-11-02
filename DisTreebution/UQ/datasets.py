import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import numpy as np    
import math


def level2idx(n,q):
    return max(0,min(math.ceil(n*q)-1,n-1))

def sine_skew(n=100, x=None,  levels=None):
    if levels is None:
        repli = 1
    else:
        repli = 100
        
    if x is None:
        x = np.random.uniform(-1,1,n)
    else:
        n = x.shape[0]
    x = np.tile(x.reshape(-1,1),(1,repli))

    g = np.random.normal(0,1,(n,repli))
    xi = g*(g<=0) + 7*g*(g>0)
    y = 5 * np.sin(8*x) + (0.2 + 3*x**3)*xi
    if not(levels is None):
        y = np.sort(y, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = y[:,level2idx(repli,q)]
        return x.reshape(-1,1), y.reshape(-1), true_quantile
    else:
        return x.reshape(-1,1), y.reshape(-1)

    
def griewank(n=100, x=None,  levels=None):
    if levels is None:
        repli = 1
    else:
        repli = 100
    if x is None:
        x = np.zeros((n,2))
        x[:,0] = np.random.uniform(-5,5,n)
        x[:,1] = np.random.uniform(-3,3,n)
    
    n = x.shape[0]
    x = np.tile(x.reshape(n,2,1),(1,1,repli))
    g = np.random.normal(0,1,(n,repli))
    xi = g*(g<=0) + 5*g*(g>0)
    G = np.sum(x**2, axis=1)/4000 - np.cos(x[:,0,:])*np.cos(x[:,1,:]/np.sqrt(2))+1
    y = G*xi
    
    if not(levels is None):
        y = np.sort(y, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = y[:,level2idx(repli,q)]
        return x, y.reshape(-1), true_quantile
    else:
        return x[:,:,0], y.reshape(-1)
    
    

def michalewicz(n=100, x=None,  levels=None):
    if levels is None:
        repli = 1
    else:
        repli = 5000
    if x is None:
        x = np.random.uniform(0,4,n)
    else:
        n = x.shape[0]
    x = np.tile(x.reshape(-1,1),(1,repli))
    g = np.random.normal(0,1, (n,repli))
    xi = 3*g*(g<=0) + 6*g*(g>0)
    res = -2*np.sin(x)*(np.sin(x**2/np.pi))**30 
    res -= (0.1*np.cos(np.pi*x/10)**3 / np.abs(-np.sin(x)*np.sin(x**2/np.pi)**30+2) ) * xi**2
    if not(levels is None):
        res = np.sort(res, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = res[:,level2idx(repli,q)]
        return x.reshape(-1,1), res.reshape(-1), true_quantile
    else:
        return x.reshape(-1,1), res.reshape(-1)


def candes_sesia(n=100, d=4, x=None, levels=None):
    import pandas as pd
    if x is None:
        x = np.random.uniform(0,1,(n,d))
    d = x.shape[1]
    beta = np.ones(d)
    n = x.shape[0]
    def f(a):
        return (2*np.sin(np.pi*a)+np.pi*a)
    y = f(x@beta) + np.random.normal(0,1,n)*np.sqrt(1+(x@beta)**2)
        
    if not(levels is None):
        repli = 1000
        for i in range(repli):
            xtemp, ytemp = candes_sesia(d=d, x=x)
            if i == 0:
                y = ytemp.reshape(-1,1)
            else:
                y = np.concatenate((y,ytemp.reshape(-1,1)),axis=1) 
        y = np.sort(y, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = y[:,level2idx(repli,q)]
        return x, y, true_quantile
    else:
        return x, y.reshape(-1)
    

def bimodal(n=100, x=None,  levels=None):
    if x is None:
        x = np.random.uniform(-1.5,1.5,n)
    n = x.shape[0]
    def f(a):
        return ((a-1)**2)*(a+1)      
    def g(a):
        return 2*(a>=-0.5)*np.sqrt(np.abs(a+0.5))
    def sigma(a):
        return np.sqrt(1/4+np.abs(a))
            
    u = np.random.rand(n)
    x = x.reshape(-1)
    y = (u>=0.5)*np.random.normal(f(x)-g(x),sigma(x)) + (u<0.5)*np.random.normal(f(x)+g(x),sigma(x))
    if not(levels is None):
        repli = 1000
        n = x.shape[0]
        for i in range(repli):
            xtemp, ytemp = bimodal(x=x)
            if i == 0:
                y = ytemp.reshape(-1,1)
            else:
                y = np.concatenate((y,ytemp.reshape(-1,1)),axis=1) 
        y = np.sort(y, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = y[:,level2idx(repli,q)]
        return x, y, true_quantile
    else:
        return x.reshape(-1,1), y.reshape(-1) 

def asymmetric(n=100, x=None,  levels=None):
    if x is None:
        x = np.random.uniform(-5,5,n)
    n = x.shape[0]
    x = x.reshape(-1)
    y = 5*x + np.random.gamma(1+2*np.abs(x),1+2*np.abs(x))
    if not(levels is None):
        repli = 1000
        n = x.shape[0]
        all_x = []
        for i in range(repli):
            xtemp, ytemp = asymmetric(x=x)
            if i == 0:
                y = ytemp.reshape(-1,1)
            else:
                y = np.concatenate((y,ytemp.reshape(-1,1)),axis=1) 
        y = np.sort(y, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = y[:,level2idx(repli,q)]
        return x, y, true_quantile
    else:
        return x.reshape(-1,1), y.reshape(-1)    
    
    
def homoscedastic(n=100, x=None,  levels=None):
    if x is None:
        x = np.random.uniform(-5,5,n)
    n = x.shape[0]
    x = x.reshape(-1)
    y = np.random.normal(x,np.ones(n))
    if not(levels is None):
        repli = 1000
        n = x.shape[0]
        all_x = []
        for i in range(repli):
            xtemp, ytemp = homoscedastic(x=x)
            if i == 0:
                y = ytemp.reshape(-1,1)
            else:
                y = np.concatenate((y,ytemp.reshape(-1,1)),axis=1) 
        y = np.sort(y, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = y[:,level2idx(repli,q)]
        return x, y, true_quantile
    else:
        return x.reshape(-1,1), y.reshape(-1)
    
def heteroscedastic(n=100, x=None,  levels=None):
    if x is None:
        x = np.random.uniform(-5,5,n)
    x = x.reshape(-1)
    y = np.random.normal(x,1+np.abs(x))
    if not(levels is None):
        repli = 1000
        n = x.shape[0]
        all_x = []
        for i in range(repli):
            xtemp, ytemp = heteroscedastic(x=x)
            if i == 0:
                y = ytemp.reshape(-1,1)
            else:
                y = np.concatenate((y,ytemp.reshape(-1,1)),axis=1) 
        y = np.sort(y, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = y[:,level2idx(repli,q)]
        return x, y, true_quantile
    else:
        return x.reshape(-1,1), y.reshape(-1)
    
    
def sales(n=1000, x=None,  levels=None):
    import pandas as pd
    if x is None:
        # Features
        advertising_expenditure = np.random.uniform(100, 1000, size=n)
        competition = np.random.uniform(0, 1, size=n)
        price = np.random.uniform(50, 200, size=n)
        seasonality = np.random.choice([0, 1], size=(n, 4))

        # True relationship (nonlinear)
        true_sales = 1000 * advertising_expenditure**0.5 + 500 * competition + 10 * price - 200 * seasonality[:, 0] + \
                     50 * seasonality[:, 1] - 100 * seasonality[:, 2] + 150 * seasonality[:, 3]

        # Introduce heavy-tailed noise (t-distribution)
        noise = np.random.standard_t(df=3, size=n)

        # Sales with noise
        y_sales = true_sales + noise

        # Create DataFrame
        data = pd.DataFrame({
            'AdvertisingExpenditure': advertising_expenditure,
            'Competition': competition,
            'Price': price,
            'Seasonality1': seasonality[:, 0],
            'Seasonality2': seasonality[:, 1],
            'Seasonality3': seasonality[:, 2],
            'Seasonality4': seasonality[:, 3],
            'Sales': y_sales
        })
        x = data[[col for col in data.columns if col!='Sales']].to_numpy()
        y = data['Sales'].to_numpy()
    else:
        n = x.shape[0]
        true_sales = 1000 * x[:,0]**0.5 + 500 * x[:,1] + 10 * x[:,2] - 200 * x[:,3] + \
                     50 * x[:,4] - 100 * x[:,5] + 150 * x[:,6]

        # Introduce heavy-tailed noise (t-distribution)
        noise = np.random.standard_t(df=3, size=n)

        # Sales with noise
        y = true_sales + noise
        
    if not(levels is None):
        repli = 1000
        n = x.shape[0]
        for i in range(repli):
            xtemp, ytemp = sales(x=x)
            if i == 0:
                y = ytemp.reshape(-1,1)
            else:
                y = np.concatenate((y,ytemp.reshape(-1,1)),axis=1) 
        y = np.sort(y, axis=1)
        true_quantile = np.zeros((n,len(levels)))
        for i_q, q in enumerate(levels):
            true_quantile[:,i_q] = y[:,level2idx(repli,q)]
        return x, y, true_quantile
    else:
        return x, y.reshape(-1)