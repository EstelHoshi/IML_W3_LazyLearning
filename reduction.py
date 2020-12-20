import math
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from classification import (kNNAlgorithm, _sanitize, _dist_1norm, _dist_2norm,
                            _dist_chebyshev, _vote_majority, _vote_inverse,
                            _vote_sheppard)


def reductionKNNAlgorithm(X, y, algorithm='drop2', **kwargs):
    X = _sanitize(X)
    y = _sanitize(y)

    algorithm = algorithm.lower()

    if algorithm == 'drop2':
        return DROP(X, y, v=2, **kwargs)

    elif algorithm == 'drop3':
        return DROP(X, y, v=3, **kwargs)

    elif algorithm == 'snn':
        return SNN(X, y)

    elif algorithm == 'enn':
        return ENN(X, y, **kwargs)
        
    else:
        return X, y


def DROP(T, y, distance='2-norm', v=2, n_proc=2, **knn_kwargs):

    dist_func = {
        '1-norm': _dist_1norm,
        '2-norm': _dist_2norm,
        'chebyshev': _dist_chebyshev
    }[distance]

    knn_kwargs['distance'] = distance
    knn_kwargs['n_proc'] = 1
    k = knn_kwargs['k']
    
    if v == 3: # DROP3
        T, y = ENN(T, y, **knn_kwargs)

    n = T.shape[0]
    S = np.ones(n, dtype=np.bool)
    A = [list() for _ in range(n)]

    def single_knn_min_enemy_dist(idx):
        dists = dist_func(T, T[idx])
        sorted_i = np.argsort(dists)
        knn_result = sorted_i[1:k+2]
        min_enemy_dist = dists[sorted_i][y[sorted_i] != y[idx]][0] #  np.min(dists[y != y[idx]])
        return knn_result, min_enemy_dist

    def single_update_assoc(a):
        dists = dist_func(T, T[a])
        sorted_i = np.argsort(dists)
        ini = int(S[a]) # if is still present, ignore distance to itself
        knn[a, :] = sorted_i[S[sorted_i]][ini:k+1+ini] # k+1 dist to points still present
        A[knn[a, -1]].append(a)

    with ThreadPool(processes=n_proc) as pool:
        # compute k+1 nn for each sample (list of nn) and distance to nearest enemy
        res = pool.map(single_knn_min_enemy_dist, range(n))
        knn = np.array([r[0] for r in res])

        # create list of associates
        for i, indices in enumerate(knn):
            for j in indices:
                A[j].append(i)

        # compute order of removal with distances to nearest enemies
        enemy_dists = [r[1] for r in res]
        order_of_removal = np.argsort(enemy_dists)[::-1]

        for i in order_of_removal:
            # count associates of i correctly classified
            assoc = A[i]
            with_mask = np.unique(knn[assoc, :k])
            y_pred_with = kNNAlgorithm(T[assoc], T[with_mask], y[with_mask], **knn_kwargs)
            with_ = np.count_nonzero(y[assoc] == y_pred_with)

            # count associates of i correctly classified without i
            assoc_knn = knn[assoc, :]   # this time get k+1 nn from each assoc
            without_mask = np.unique(assoc_knn[assoc_knn != i]) # and remove i
            y_pred_without = kNNAlgorithm(T[assoc], T[without_mask], y[without_mask], **knn_kwargs)
            without_ = np.count_nonzero(y[assoc] == y_pred_without)

            if without_ >= with_:
                # remove i and update associates with their next nn
                S[i] = False
                pool.map(single_update_assoc, assoc)
            
    return T[S], y[S]


def ENN(X, y, **knn_kwargs):
    knn_kwargs['n_proc'] = 2
    y_pred = kNNAlgorithm(X, X, y, offset=1, **knn_kwargs)
    return X[y_pred == y], y[y_pred == y]



def SNN(X_train, y_train): #SNN
    #X_train = X_train.to_numpy()
    #X_test = X_test.to_numpy()
    #y_train = y_train.to_numpy()
    #y_train = y_train.astype(int)               # ground truth
    n = len(y_train)                            # original number of instances

    A = np.zeros([len(X_train), len(X_train)])  # initialize matrix A
    for i in range(n):
        a = y_train - y_train[i]
        id = np.where(a == 0)[0]                # first criteria for RN
        A[id,i] = 1                             # symmetric matrix

    d = np.zeros([len(X_train), len(X_train)])  # initialize distance matrix
    for i in range(n):
        d[:, i] = np.sum(np.abs(X_train - X_train[i, :]), axis=1)
        d[i,i] = math.inf

    for i in range(n):                          # for each column in A
        d0 = d[:,i]/abs((A[:,i]-1))
        mind0 = np.min(d0)                      # it always be the minimum distance of the instance in another class
        pn = d[:,i] - mind0
        zero_rows = np.where(pn >= 0)           # second criteria for RN not meet
        A[zero_rows,i] = 0                      # change the 1 for a 0

    # A = np.zeros([8,8])                       # test with a known matrix A
    # A[0,0] = 1
    # A[1,1] = 1
    # A[2, 2] = 1
    # A[3, 3] = 1
    # A[4, 4] = 1
    # A[5, 5] = 1
    # A[6, 6] = 1
    # A[7, 7] = 1
    # A[0,5] = 1
    # A[1,6] = 1
    # A[3,1] = 1
    # A[3,4] = 1
    # A[4,6] = 1
    # A[5,2] = 1
    # A[6,3] = 1
    # A[7,3] = 1
    # #print(A)

    A = np.int8(A)                                 # I was hoping this would reduce the computation time but it seems not
    id_A = np.array(range(len(A)))

    S = [] #subset
    change = True
    while change == True and np.shape(A)[1] > 0:
        change = False
        ##step1##
        col_del = []
        row_del = []
        for i in range(np.shape(A)[1]):            # for each column in the binary matrix A
            RN_i = np.where(A[:,i] == 1)[0] #RN of column i
            if len(RN_i) == 1:  #if its the single RN of column i
                change = True
                S.append(id_A[RN_i])
                col2app = np.where(A[RN_i,:] == 1)[1]               # search if there is this RN is a RN of another column
                for c2a in col2app:
                    col_del.append(c2a)
                row_del.append(RN_i)

        A = np.delete(A, row_del, 0)                        # delete rows
        A = np.delete(A, col_del, 1)                        # delete columns

        id_A = np.delete(id_A,row_del)
        # print(np.shape(A))

        ##step2##
        rows = np.shape(A)[0]
        for j in range(np.shape(A)[0]):
            # print(j)
            if A[j,0]!=2:   #if j not in row_del:
                for k in range(j+1,rows):
                    if A[k,0] !=2: # if k not in j_no:
                        row_dif = A[j,:] - A[k,:]#np.subtract(A[j,:],A[k,:]) #A[j,:] - A[k,:]
                        if not(1 in row_dif):
                            change = True
                            A[j,0] = 2
                            break
                        elif not (-1 in row_dif):
                            change = True
                            A[k,0] = 2

        id_A = np.delete(id_A, np.where(A[:, 0] == 2))
        A = np.delete(A,np.where(A[:,0]==2),0)

        ##step3##
        for i in range(np.shape(A)[1]):
            if A[0,i]!=2:   #if j not in row_del:
                for k in range(i+1,np.shape(A)[1]):
                    if A[0,k] !=2: # if k not in j_no:
                        row_dif = A[:,i] - A[:,k]#np.subtract(A[j,:],A[k,:]) #A[j,:] - A[k,:]
                        if not(-1 in row_dif):
                            change = True
                            A[0,i] = 2
                            break
                        elif not (1 in row_dif):
                            change = True
                            A[0,k] = 2

        A = np.delete(A,np.where(A[0,:]==2),1)

    for i in id_A:
        S.append(i)

    X_train2 = np.zeros([len(S),np.shape(X_train)[1]])
    y_train2 = np.zeros([len(S),1])
    for j in range(len(S)):
        X_train2[j,:] = X_train[S[j],:]
        y_train2[j,0] = y_train[S[j]]

    return X_train2, y_train2.ravel()
