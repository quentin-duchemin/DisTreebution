import numpy as np
from CRPStreeC.FenwickTree import FenwickTree
from CRPStreeC.WBTree import WBTree


def entropies_CRPS(order, y, use_LOO=True):
    # order: order of the features
    N = len(y)

    # y sorted using the order of the feature
    ysort = y[order]
    argsort = np.argsort(ysort)

    pos = np.zeros(N, dtype=np.int32)
    i = np.arange(N, dtype = np.int32)
    np.put(pos, argsort, i) # pos[argsort[i]] = i 

    
    # Use of a Weight Balanced Tree: Time complexity O(n x log(n))
    tree=WBTree()
    ranks = []
    for el in ysort:
        a,b = tree.add(el)
        ranks.append(a+1)
        

    ls_y = [ysort[0]]
    h = 0
    fenwick_tree_up = FenwickTree(N)
    fenwick_tree_down = FenwickTree(N)

    fenwick_tree_up.add(pos[0], ysort[0])
    fenwick_tree_down.add(N-pos[0]-1, ysort[0])



    # Total weighted sum
    Wup = ysort[0]
    Wdown = ysort[0]
    # Total sum
    S = ysort[0]

    # index for which the weighted fenwick trees are valid
    idx_ref_weighted_ft = pos[0]
    hup, hlow = 0, 0
    
    # Entropy with empty list and with one element
    entropy = [0,0]


    for idx in range(1,len(order)):
        n = idx
        # Update cumulative sum
        Sold = S
        S = S + ysort[idx]
        # compute cumulative sum up to current position before adding the new element
        cum_sum_0_old = fenwick_tree_up.prefix_sum(pos[idx])
        # compute weighted cumulative sum up to current position before adding the new element
        cum_sum_new_end = Sold - cum_sum_0_old
        # Update weighted cumulative sum
        Wup_old = Wup
        Wup = Wup + ranks[idx] * ysort[idx] + cum_sum_new_end #+ ranks[idx] * ysort[idx]


        Wdown_old = Wdown
        Wdown = (n+2) * S - Wup

        hup = hup - 2*S
        hup = hup + 2*cum_sum_0_old
        hup = hup + 2*Wup
        hup = hup + ((ranks[idx]-1) * (ranks[idx]-2)) * ysort[idx]

        hlow = hlow  + 2*(n+1)*cum_sum_0_old
        hlow = hlow + ((n-ranks[idx]+1) * (n-ranks[idx]+2) ) * ysort[idx]

        entropy.append((hup - hlow)/(n+1)**3)
            
        idx_ref_weighted_ft = pos[idx]
        fenwick_tree_up.add(pos[idx], ysort[idx])
        fenwick_tree_down.add(N-pos[idx]-1, ysort[idx])
        
        if (use_LOO and (idx>=2)):
            entropy[-1] *= (n+1)**2 / (n**2)

    return np.array(entropy)

