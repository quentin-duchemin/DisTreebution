def entropies_bruteforce(N):
    #np.random.seed(1)
    y = np.random.normal(0,1,N)
    #y = np.cumsum(np.ones(N))
    import random
    #random.seed(3)
    random.shuffle(y)
    import time
    curr = time.time()



    # order of the features
    order = [i for i in range(N)]

    # y sorted using the order of the feature
    ysort = [y[idx] for idx in order]
    argsort = np.argsort(ysort)
    pos = np.zeros(N, dtype=int)
    for i in range(N):
        pos[argsort[i]] = i

    ranks = [1]
    yranks = [ysort[0]]
    for i in range(1,N):
        k=0
        while (ysort[i]>yranks[k]):
            k += 1
            if k==len(yranks):
                break
        ranks.append(k+1)
        yranks.insert(k,ysort[i])
        a= get_entropy(yranks)

    final_time = time.time()
    print("Execution time =", (final_time-curr)/60)

    return final_time - curr


if False:
    ysort = np.sort(y)

    n = len(y)
    s1 = 0
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                s1 += 2*(ysort[j]-ysort[i])

    for j in range(n):
        for k in range(j+1,n):
            for i in range(k+1,n):
                s1 += 2*(ysort[k]-ysort[i])

    print(s1)

    s2 = 0
    for i in range(1,n+1):
        s2 += (i-1)*(2*n-3*i+2)*(ysort[i-1]+ysort[n-i])
    print(s2, s2/n**3)