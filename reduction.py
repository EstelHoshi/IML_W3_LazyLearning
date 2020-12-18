import numpy as np
import math
import pandas as pd
from classification import kNNAlgorithm, kNNAlgorithm_Estel
#import numba


def reductionKNNAlgorithm():
    pass

def reductionKNNAlgorithm_Estel(X_train,y_train,X_test,K, similarity, policy, weighting): #SNN
    #X_train = X_train.to_numpy()
    #X_test = X_test.to_numpy()
    #y_train = y_train.to_numpy()
    y_train = y_train.astype(int)               # ground truth
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
        print(np.shape(A))

        ##step2##
        rows = np.shape(A)[0]
        for j in range(np.shape(A)[0]):
            print(j)
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

    y_test = kNNAlgorithm_Estel(X_train2, y_train2, X_test, K, similarity, policy, weighting)

    return y_test