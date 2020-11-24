import numpy as np
import pandas as pd
from scipy import stats

def kNNAlgorithm():
    pass

def kNNAlgorithm_Estel(X_train,y_train,X_test,k, similarity, policy, weighting):
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()

    d = np.zeros([len(X_train), len(X_test)])
    if similarity == 'minkowski1':
        for i in range(len(X_test)):
            d[:, i] = np.sum(np.abs(X_train - X_test[i, :]), axis=1)
    elif similarity == 'minkowski2':
        for i in range(len(X_test)):
            #d[:, i] = np.sum(np.power((X_train - X_test[i, :]),p), axis=1)
            d[:, i] = np.sum(np.square(X_train - X_test[i, :]), axis=1)
    else:
        print("not done yet")




    id_k = np.zeros([k, len(X_test)]).astype(int)
    y_k = np.copy(id_k)
    d_df = pd.DataFrame(d)
    d_k = np.zeros([k, len(X_test)])
    for i in range(len(X_test)):
        d_k[:,i] = d_df[i].nsmallest(k).values
        id_k[:,i] = d_df[i].nsmallest(k).index
        y_k[:,i] = y_train[id_k[:,i]][:, 0]

    print(id_k)
    print(y_k)
    print(d_k)


    y_test = np.zeros([1, len(X_test)])
    if policy == 'majority':
        y_test = stats.mode(y_k)[0]
    if policy == 'inverse':
        d_k_inv = 1 / d_k
        print(d_k_inv)
        d_k_inv_w = np.zeros([k, len(X_test)])
        for i in range(len(X_test)):
            for j in range(k):
                B = (y_k[:, i] == y_k[j, i]).astype(int)
                d_k_inv_w[j, i] = np.sum(d_k_inv[:, i] * B)
            print(d_k_inv_w[:, i])
            print(y_k[:,i])
            best_k = np.argmax(d_k_inv_w[:, i])
            print(y_k[best_k,i])
            y_test[0,i] = y_k[best_k,i]

        print(d_k[:,1])

    if policy == 'sheppard':
        print(np.exp(d_k))
        d_k_exp = np.exp(d_k)
        print(d_k_exp)
        d_k_inv_w = np.zeros([k, len(X_test)])
        for i in range(len(X_test)):
            for j in range(k):
                B = (y_k[:, i] == y_k[j, i]).astype(int)
                d_k_inv_w[j, i] = np.sum(d_k_exp[:, i] * B)
            print(d_k_inv_w[:, i])
            print(y_k[:,i])
            best_k = np.argmax(d_k_inv_w[:, i])
            print(y_k[best_k,i])
            y_test[0,i] = y_k[best_k,i]

        print(d_k[:,1])

    return np.transpose(y_test)
