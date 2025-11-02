class FenwickTree(object):
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

    def add(self, idx, k):
        """ Adds k to idx'th element (0-based indexing). """
        if idx < 0 or idx >= self._n:
            raise IndexError()
        idx += 1
        while idx <= self._n:
            self._v[idx - 1] += k
            idx += idx & -idx

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
        return isinstance(other, FenwickTree) and self._n == other._n and self._v == other._v