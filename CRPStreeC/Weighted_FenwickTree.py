import numpy as np
class Weighted_FenwickTree(object):
    """
    A data structure for maintaining cumulative (prefix) sums.
    (aka "binary indexed tree")
    Incrementing a value is O(log n).
    Calculating a cumulative sum is O(log n).
    Retrieving a single frequency is a special case of calculating a cumulative
    sum, and is thus O(log n).
    """
    def __init__(self, n):
        """ Initializes n frequencies to zero. """
        self._n = n
        self._v = [0] * n
        self.idx2minrange = [0] * (n+1)
        self.idx2maxrange = [0] * (n+1)
        for i in range(1,n+1):
            self.idx2minrange[i] = i - (i & -i )
            self.idx2maxrange[i] = i
        
    def first_add(self, idx, k):
        idx += 1
        while idx <= self._n:
            self._v[idx - 1] += k
            idx += idx & -idx

    def __len__(self):
        return self._n

    def prefix_sum(self, stop):
        """ Returns sum of first elements (sum up to *stop*, exclusive). """
        if stop < 0 or stop > self._n:
            raise IndexError()
        _sum = 0
        while stop > 0:
            _sum += self._v[stop - 1]
            stop &= stop - 1
        return _sum

    def range_sum(self, start, stop):
        """ Returns sum from start (inclusive) to stop (exclusive). """
        if start < 0 or start > self._n:
            raise IndexError()
        if stop < start or stop > self._n:
            raise IndexError()
        if stop==start:
            result = 0
        else:
            result = self.prefix_sum(stop)
            if start > 0:
                result -= self.prefix_sum(start)
        return result

    def __getitem__(self, idx):
        return self.range_sum(idx, idx + 1)

    def frequencies(self):
        """ Retrieves all frequencies in O(n). """
        _frequencies = [0] * self._n
        for idx in range(1, self._n + 1):
            _frequencies[idx - 1] += self._v[idx - 1]
            parent_idx = idx + (idx & -idx)
            if parent_idx <= self._n:
                _frequencies[parent_idx - 1] -= self._v[idx - 1]
        return _frequencies

    
    def update_up(self, idx, rank, k, idx_ref, ft_weighted_up, ft_weighted_down, fenwick, n):
        N = self._n
        minrange = self.idx2minrange[idx]
        maxrange = self.idx2maxrange[idx]
        if maxrange<=idx_ref:
            sup1 = ft_weighted_up.range_sum(minrange,idx)
            sup2 = ft_weighted_up.range_sum(idx,maxrange)
            s = fenwick.range_sum(idx-1,maxrange)
            if rank==-1:
                s = 0
            return (sup1 + sup2 + k*rank + s)
        elif minrange<=idx_ref-1:
            sup = ft_weighted_up.range_sum(minrange,idx_ref)
            s = fenwick.range_sum(idx_ref,maxrange)
            sdown =  -ft_weighted_down.range_sum(N-maxrange,N-idx_ref) + (n+1) * s
            s = fenwick.range_sum(idx-1,maxrange)
            if rank==-1:
                s = 0
            return (sup + sdown + k*rank + s)
        else:
            u = N-maxrange
            s = fenwick.range_sum(minrange,maxrange)
            sdown = -ft_weighted_down.range_sum(N-maxrange,N-minrange) + (n+1) * s
            s = fenwick.range_sum(idx-1,maxrange)
            if rank==-1:
                s = 0
            return (sdown + k*rank + s)
        
    def get_list2add_lazy_up(self, idx, rank, k, idx_ref, ft_weighted_up, ft_weighted_down, fenwick, n):
        """ Adds k to idx'th element (0-based indexing). """
        if idx < 0 or idx >= self._n:
            raise IndexError()
        idx += 1
        idx_ref += 1
        N = self._n
        list2add = {}
        while idx <= self._n:
            list2add[idx-1] = self.update_up(idx, rank, k, idx_ref, ft_weighted_up, ft_weighted_down, fenwick, n)
            idx += idx & -idx        
        return list2add
    
    def get_list2add_up(self, idx_ref_old, rank, k, idx_ref_new, ft_weighted_up, ft_weighted_down, fenwick, n):
        idx_ref_old += 1
        idx_ref_new += 1
        N = self._n
        list2add = {}
        LSIDX = []
        idx = idx_ref_new + 1
        while idx <= self._n:
            LSIDX.append(idx)
            idx += idx & -idx
            
#         for idx in range(idx_ref_old, idx_ref_new):
#             list2add[idx-1] = self.update_up(idx, -1, 0, idx_ref_old, ft_weighted_up, ft_weighted_down, fenwick, n)
        idx = idx_ref_new
        while idx <= self._n:
            list2add[idx-1] = self.update_up(idx, rank, k, idx_ref_old, ft_weighted_up, ft_weighted_down, fenwick, n)
            idx += idx & -idx 
        return list2add
    
    def update_down(self, idx, rank, k, idx_ref, ft_weighted_down, ft_weighted_up, fenwick_down, n):
        N = self._n
        minrange = self.idx2minrange[idx]
        maxrange = self.idx2maxrange[idx]
        if idx_ref>=maxrange:
            sup1 = ft_weighted_down.range_sum(minrange,idx)
            sup2 = ft_weighted_down.range_sum(idx,maxrange)
            s = fenwick_down.range_sum(idx-1,maxrange)
            if rank==-1:
                s = 0
            return (sup1 + sup2 + k*rank + s)
        elif idx_ref-1>=minrange:
            sdown = ft_weighted_down.range_sum(minrange, idx_ref)
            u = N-idx_ref-1
            s = fenwick_down.range_sum(idx_ref, maxrange)
            sup = -ft_weighted_up.range_sum(N-maxrange,N-idx_ref) + (n+1) * s
            s = fenwick_down.range_sum(idx-1,maxrange)
            if rank==-1:
                s = 0
            return(sup + sdown + k*rank + s)
        else:
            u = N-maxrange
            s = fenwick_down.range_sum(minrange,maxrange)
            sdown = -ft_weighted_up.range_sum(N-maxrange,N-minrange) + (n+1) * s

            s = fenwick_down.range_sum(idx-1,maxrange)
            if rank==-1:
                s = 0
            return(sdown + k*rank + s)

    def get_list2add_lazy_down(self, idx, rank, k, idx_ref, ft_weighted_down, ft_weighted_up, fenwick_down, n):
        """ Adds k to idx'th element (0-based indexing). """
        if idx < 0 or idx >= self._n:
            raise IndexError()
        idx += 1
        idx_ref += 1
        N = self._n
        list2add = {}
        while idx <= self._n:
            list2add[idx-1] = self.update_down(idx, rank, k, idx_ref, ft_weighted_down, ft_weighted_up, fenwick_down, n)
            idx += idx & -idx 
        return list2add

    def get_list2add_down(self, idx_ref_old, rank, k, idx_ref_new, ft_weighted_down, ft_weighted_up, fenwick_down, n):
        idx_ref_old += 1
        idx_ref_new += 1
        N = self._n
        list2add = {}
        LSIDX = []
        idx = idx_ref_new + 1
        while idx <= self._n:
            LSIDX.append(idx)
            idx += idx & -idx
#         for idx in range(idx_ref_old, idx_ref_new):
#             list2add[idx-1] = self.update_down(idx, -1, 0, idx_ref_old, ft_weighted_down, ft_weighted_up, fenwick_down, n)
        idx = idx_ref_new
        while idx <= self._n:
            list2add[idx-1] = self.update_down(idx, rank, k, idx_ref_old, ft_weighted_down, ft_weighted_up, fenwick_down, n)
            idx += idx & -idx 
        return list2add

    def update_from_list(self, list2add):
        for key, value in list2add.items():
            self._v[key] = value

    def __setitem__(self, idx, value):
        # It's more efficient to use add directly, as opposed to
        # __setitem__, since the latter calls __getitem__.
        self.add(idx, value - self[idx])

    def init(self, frequencies):
        """ Initialize in O(n) with specified frequencies. """
        if len(frequencies) != self._n:
            raise ValueError()
        for idx in range(self._n):
            self._v[idx] = frequencies[idx]
        for idx in range(1, self._n + 1):
            parent_idx = idx + (idx & -idx) # parent in update tree
            if parent_idx <= self._n:
                self._v[parent_idx - 1] += self._v[idx - 1]

    def __eq__(self, other):
        return isinstance(other, Weighted_FenwickTree) and self._n == other._n and self._v == other._v
    
    def check_correctness(self, vals, idx_ref):
        for i in range(self._n):
            minrange = self.idx2minrange[i+1]
            maxrange = self.idx2maxrange[i+1]
            if maxrange<=idx_ref:
                s = np.sum([vals[j] for j in range(minrange, maxrange)])
    #            assert self._v[i]==s, "weighted cum sum error: {0}  {1}".format(str(self._v[i]), s)
                if abs(self._v[i]-s)>=1e-4:
                    print("weighted cum sum error: {0}  {1} {2}  {3}".format(str(self._v[i]), s, minrange, maxrange))