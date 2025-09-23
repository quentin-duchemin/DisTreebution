import numpy as np
import math
from QRT.MinMaxHeap import *

def level2idx(n,q):
    return max(0,min(math.ceil(n*q)-1,n-1))


def get_entropy(values, quantiles):
    vals = np.sort(values)
    res = 0
    n = len(vals)
    tot_sum = np.sum(vals)
    cum_sum = np.cumsum(vals)
    for i_q, q in enumerate(quantiles):
        floor_qn = level2idx(n, q)
        res += vals[floor_qn] * ( (floor_qn+1)/n-q)
        res += tot_sum * q / n
        res -= cum_sum[floor_qn] / n
    return res


def get_entropy_multiquantiles(heap2sum, heaps, quantiles, n):
    res = 0
    tot_sum = np.sum(heap2sum)
    cum_sum = np.cumsum(heap2sum)
    for i_q, q in enumerate(quantiles):
        iqm, max_heap_i_q = get_max_heaps_minus(i_q, heaps)
        res += max_heap_i_q * ((level2idx(n,q)+1)/n-q)
        res += tot_sum * q / n
        res -= cum_sum[i_q] / n
    return res

def get_max_heaps_minus(i_qp, heaps):
    res = None
    while (i_qp>=0 and (res is None)):
        res = heaps[i_qp].peekmax()
        if res is None:
            i_qp -= 1
    return i_qp, res


def get_min_heaps_plus(i_qp, heaps):
    res = None
    while (i_qp<len(heaps) and (res is None)):
        res = heaps[i_qp].peekmin()
        if res is None:
            i_qp += 1
    return i_qp, res


def OLD_LOO_quantiles(y, entropy, quantiles):
    n = len(y)
    res = 0
    for q in quantiles:    
        idx_star = int(n*q)
        idx_m = int((n-1)*q)
        bool_idx_m_lower_i = 1*(idx_m<np.arange(0,n))
        res += bool_idx_m_lower_i * ( (idx_m==idx_star)*(-q*y+y[idx_star]*q) + (idx_m!=idx_star)*(-q*y+y[idx_star]*(1-idx_star+q*n)+y[idx_m]*(idx_m-q*(n-1))))
        res += (1-bool_idx_m_lower_i) * ( (-q*y+y+y[idx_star]*(-idx_star+q*n)+y[idx_m+1]*(idx_m-q*(n-1))))
    return ((entropy*n+np.mean(res))/(n-1))


    
def LOO_quantiles(y, entropy, quantiles, heaps):
    n = len(y)
    HLOO = entropy*n
    for i_q, q in enumerate(quantiles):
        rstar = level2idx(n,q)
        i_qstar, y_rstar = get_max_heaps_minus(i_q, heaps)
        
        i_qp, y_rstarp1 = get_min_heaps_plus(i_q+1, heaps)
        if y_rstarp1 is None:
            y_rstarp1 = y_rstar
        
        if heaps[i_qstar].size==1:
            if i_q!=0:
                i_qp, y_rstarm1 = get_max_heaps_minus(i_q-1, heaps)
            else:
                y_rstarm1 = y_rstar
        elif heaps[i_qstar].size==2:
            y_rstarm1 = heaps[i_qstar].peekmin()
        else:
            y_rstarm1 = min([heaps[i_qstar].a[1], heaps[i_qstar].a[2]])

        if rstar==level2idx(n-1,q):
            HLOO += (1-q)*(rstar+1)*(y_rstarp1-y_rstar)
        else:
            HLOO += q*(n-(rstar+1)+1)*(y_rstar-y_rstarm1)
    return HLOO/n

    
    
def entropies_MultiQuantiles(order, y, quantiles, use_LOO=True):
    # order: order of the features
    
    heaps = []
    for i in range(len(quantiles)+1):
        heaps.append(MinMaxHeap())

    # y sorted using the order of the feature
    ysort = y[order]
    entropy = [0]
    
    heap2sum = np.zeros(len(heaps))
            
    heap2sum[0] = ysort[0]
    heap2size = np.zeros(len(heaps))
    heap2size[0] = 1
    heaps[0].insert(ysort[0])
    entropy.append(get_entropy([ysort[0]], quantiles))

    for idx in range(1,len(order)):
        y_c = ysort[idx]
        for i_q, q in enumerate(quantiles):
            max_heap_i_q = heaps[i_q].peekmax()
            if np.sum(heap2size[:i_q+1])==level2idx(idx+1,q)+1:
                if not(max_heap_i_q is None):
                    if y_c <= max_heap_i_q:
                        max_heap_i_q = heaps[i_q].popmax()
                        heap2sum[i_q] -= max_heap_i_q
                        heaps[i_q].insert(y_c)
                        heap2sum[i_q] += y_c
                        y_c = max_heap_i_q
            else:
                i_q_min, min_heap_i_q_plus = get_min_heaps_plus(i_q+1, heaps)
                if min_heap_i_q_plus is None:
                    heaps[i_q].insert(y_c)
                    heap2sum[i_q] += y_c
                    heap2size[i_q] += 1
                    y_c = None
                    break
                  
                else:
                    if y_c<=min_heap_i_q_plus:
                        heaps[i_q].insert(y_c)
                        heap2sum[i_q] += y_c
                        heap2size[i_q] += 1
                        min_heap_i_q_plus = heaps[i_q_min].popmin()
                        y_c = min_heap_i_q_plus
                        heap2sum[i_q_min] -= min_heap_i_q_plus
                        heap2size[i_q_min] -= 1
                    else:
                        min_heap_i_q_plus = heaps[i_q_min].popmin()
                        heap2sum[i_q_min] -= min_heap_i_q_plus
                        heap2size[i_q_min] -= 1
                        heaps[i_q].insert(min_heap_i_q_plus)
                        heap2sum[i_q] += min_heap_i_q_plus
                        heap2size[i_q] += 1
        if not(y_c is None):
            heaps[-1].insert(y_c)
            heap2sum[-1] += y_c
            heap2size[-1] += 1
        
        
        entropy_idx = get_entropy_multiquantiles(heap2sum, heaps, quantiles, idx+1)
#         if use_LOO and (min([idx,len(order)-1-idx])>=2):
        if use_LOO and (idx>=2):
            entropy.append(LOO_quantiles(ysort[:idx+1], entropy_idx, quantiles, heaps))
        else:
            entropy.append(entropy_idx)

    return np.array(entropy)
