def get_entropy(values):
    h = 0
    n = len(values)
    for i in range(1,n+1):
        h += (i-1)*i*(values[i-1]-values[n-i])
    return h/(n)**3

def get_hup(values):
    h = 0
    n = len(values)
    for i in range(1,n+1):
        h += (i-1)*(2*n-3*i+2)*(values[i-1])
    return h

def get_hup2(n, rank, ynew, values):
    Sold = np.sum(values)
    cum_sum_weight_0_old = np.sum([(i+1)*el for i ,el in enumerate(values[:rank-1])])
    cum_sum_0_old = np.sum([el for i ,el in enumerate(values[:rank-1])])
    Wup_old = np.sum([(i+1)*el for i ,el in enumerate(values)])
    print('HUP2', Sold, cum_sum_weight_0_old, cum_sum_0_old, Wup_old)
    hup = 2*(n+3)*Sold
    hup = hup - (2*n+8)*cum_sum_0_old
    hup = hup + 6*cum_sum_weight_0_old
    hup = hup - 4*Wup_old
    hup = hup + ((rank-1) * (2*n-3*rank+4) ) * ynew
    return hup

def get_hlow(values):
    h = 0
    n = len(values)
    for i in range(1,n+1):
        h += (i-1)*(2*n-3*i+2)*(values[n-i])
    return h