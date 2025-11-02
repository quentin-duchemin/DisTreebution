import numpy as np
import math
from QRT.MinMaxHeap import *



def entropies_Quadratic(order, y, use_LOO=True):
    # order: order of the features


    # y sorted using the order of the feature
    ysort = y[order]
    entropy = [0]
    
    
    entropy.append(0)

    for idx in range(1,len(order)): 
#         if use_LOO and (min([idx,len(order)-1-idx])>=2):
        if use_LOO and (idx>=2):
            # use of Mallows Cp
            sigma2 = np.var(ysort[:idx+1], ddof=1) # unbiased variance estimate
            entropy.append(np.mean((ysort[:idx+1]-np.mean(ysort[:idx+1]))**2) + 2*sigma2/(idx+1))
        else:
            entropy.append(np.mean((ysort[:idx+1]-np.mean(ysort[:idx+1]))**2))
    return np.array(entropy)
