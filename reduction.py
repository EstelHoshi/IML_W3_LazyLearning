import numpy as np
import math
import pandas as pd
from classification import kNNAlgorithm, kNNAlgorithm_Estel

def reductionKNNAlgorithm():
    pass

def reductionKNNAlgorithm_Estel(X_train,y_train,X_test,k, similarity, policy, weighting): #SNN
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_train = y_train.astype(int)

    n = len(y_train)
    # d = np.zeros([len(X_train), len(X_train)])
    # for i in range(n):
    #     d[:, i] = np.sum(np.abs(X_train - X_train[i, :]), axis=1)
    #
    # print("start")
    # A = np.zeros([len(X_train), len(X_train)])
    # for j in range(n):
    #     print(j)
    #     for i in range(n):
    #         if y_train[i] == y_train[j]:
    #             A[j,i] = 1
    #             k = 0
    #             while k < n:
    #                 if y_train[j] != y_train[k]:
    #                     if d[i,j] <= d[i,k]:
    #                         A[j,i] = 0
    #                         k = n
    #                 k = k +1
    #
    # print(A)


    # A = np.zeros([len(X_train), len(X_train)])
    # for i in range(n):
    #     a = y_train - y_train[i]
    #     id = np.where(a == 0)[0]   #firts criteria for RN
    #     A[id,i] = 1 #simetric matrix
    # print("1")
    # d = np.zeros([len(X_train), len(X_train)])
    # for i in range(n):
    #     d[:, i] = np.sum(np.abs(X_train - X_train[i, :]), axis=1)
    #     d[i,i] = math.inf
    # print("1")
    #
    # for i in range(n): #for each column of A
    #     d0 = d[:,i]/abs((A[:,i]-1))
    #     mind0 = np.min(d0) #it always be the minimum distance of the instance in another class
    #     pn = d[:,i] - mind0
    #     zero_rows = np.where(pn < 0)
    #     A[zero_rows,i] = 0



    #Data reduction

    A = np.zeros([8,8])
    A[0,0] = 1
    A[1,1] = 1
    A[2, 2] = 1
    A[3, 3] = 1
    A[4, 4] = 1
    A[5, 5] = 1
    A[6, 6] = 1
    A[7, 7] = 1
    A[0,5] = 1
    A[1,6] = 1
    A[3,1] = 1
    A[3,4] = 1
    A[4,6] = 1
    A[5,2] = 1
    A[6,3] = 1
    A[7,3] = 1
    print(A)
    id_A = np.array(range(len(A)))
    #print(A[[0:1,2:-1],:])
    print(":)")
    #A = np.delete(A, [0,0,0,3], 1) dpm
    print(A)
    S = [] #subset
    change = True
    while change == True and np.shape(A)[1] > 0:
        change = False
        #Step 1
        col_del = []
        row_del = []

        for i in range(np.shape(A)[1]): #for each column in the binary matrix A
            #col = np.sum(A[:,i],axis = 0) #Sum
            RN_i = np.where(A[:,i] == 1)[0] #RN of column i
            print(RN_i)
            if len(RN_i) == 1:  #if its the single RN of column i
                change = True
                print("entra")
                #!!!!!!!!!!!!!!!!!!!!!!!!!S.append(X_train[RN_i,:])  #add to the subset the instance
                S.append(id_A[RN_i])
                col_del.append(np.where(A[RN_i,:] == 1)[1]) #search if there is this RN is a RN of another column
                print(col_del)
                row_del.append(RN_i)
        A = np.delete(A, col_del, 1) #delate columns
        A = np.delete(A, row_del, 0)  # delate row
        id_A = np.delete(id_A,row_del)
        print(A)
        #Step 2
        print("step2")
        row_del = []
        for j in range(np.shape(A)[0]):
            print("-----")
            print(row_del)
            j_no = row_del + [j]
            print(j_no)
            A_dif = A - A[j,:]  #substract each row j form matrix A
            A_dif = np.delete(A_dif, j_no, 0)  #delate the itself substract and the already delated rows
            boolarr = np.sum((A_dif == -1)*1,axis = 1) #check if row j can be delated
            print(A_dif == -1)
            print((A_dif == -1)*1)
            print(boolarr)
            print(np.prod(boolarr))
            if (np.prod(boolarr) == 0):
                change = True
                row_del.append(j)
                print(row_del)
                #A = np.delete(A, row_del[-1], 0)
        A = np.delete(A, row_del, 0)
        id_A = np.delete(id_A, row_del)
        print(A)
        # Step 3
        col_del = []
        print("step3)")
        for i in range(np.shape(A)[1]):
            print("----")
            print(i)
            print(A[:,i])
            A_dif = A - A[:,i].reshape(np.shape(A)[0],1)
            print(A_dif)
            i_no = col_del + [i]
            print(i_no)
            A_dif = np.delete(A_dif, i_no, 1)
            print(A_dif)
            boolarr = np.sum((A_dif == 1) * 1, axis=0)
            print(boolarr)
            if (np.prod(boolarr) == 0):
                change = True
                col_del.append(i)
        A = np.delete(A, col_del, 1)


    print(A)
    print(id_A)
    for i in id_A:
        #!!!!!!!!!!!!!!!!!!!!!!!S.append(X_train[i,:])
        S.append(i)

    print(S)
    #print(np.shape(S))



    #y_test = kNNAlgorithm_Estel(X_train, y_train, X_test, k, similarity, policy, weighting)
    return 0