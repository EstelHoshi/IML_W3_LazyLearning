import numpy as np
import pandas as pd
from scipy import stats
from ReliefF import ReliefF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def kNNAlgorithm():
    pass

def kNNAlgorithm_Estel(X_train,y_train,X_test,k, similarity, policy, weighting):
    #X_train = X_train.to_numpy()
    #X_test = X_test.to_numpy()
    #y_train = y_train.to_numpy()

    #y_train = y_train.astype(int)
    #y_train = y_train.reshape(len(y_train))
    print(np.shape(y_train))
    print(np.shape(X_train))
    print(np.shape(X_test))

    #1.-FEATURE PREPROCESSING
    if weighting == 'SelectKBest':
        test = SelectKBest(score_func=chi2, k=np.shape(X_test)[1])
        fit = test.fit(X_train, y_train.astype(int))
        weighted_features = fit.scores_
        weighted_features = weighted_features/np.max(weighted_features)
        X_train = X_train * weighted_features
        X_test = X_test * weighted_features
    # elif weighting == 'ReliefF':
    #     y_train2 = y_train.reshape(len(y_train))
    #     fs = ReliefF(n_neighbors=k, n_features_to_keep=np.shape(X_test)[1])
    #     x_train_fs = fs.fit_transform(X_train, y_train2)
    #     features_weight = fs.feature_scores
    #     print("AAAA")
    #     print(features_weight)




    #2.-SIMILARITY METRIC
    # Compute distnace between each test data to each training data
    d = np.zeros([len(X_train), len(X_test)]) #for each train instance we store the distance to each test intance
    if similarity == 'minkowski1':  #Manhattan distance
        for i in range(len(X_test)):
            d[:, i] = np.sum(np.abs(X_train - X_test[i, :]), axis=1)
    elif similarity == 'minkowski2': #Euclidian distance
        for i in range(len(X_test)):
            d[:, i] = np.sqrt(np.sum(np.square(X_train - X_test[i, :]), axis=1))
    elif similarity == 'chebyshev':
        for i in range(len(X_test)):
            d[:,i] = np.max(np.abs(X_train - X_test[i,:]), axis = 1)



    id_k = np.zeros([k, len(X_test)]).astype(int) #index (in the training dataset) of each NN intance
    y_k = np.copy(id_k)                           #label of each the NN instance
    d_df = pd.DataFrame(d)                        #matrix of distance into a data frame
    d_k = np.zeros([k, len(X_test)])              #distance from each NN to each X test (id_k rows stored in d_df)
    for i in range(len(X_test)):                  #searching for each column in d (distances from a given test to all train)
        d_k[:,i] = d_df[i].nsmallest(k).values    #distance of the NN to each test value
        id_k[:,i] = d_df[i].nsmallest(k).index    #position of the NN
        y_k[:,i] = y_train[id_k[:,i]][:, 0]       #label of the NN

    print(id_k)
    print(y_k)
    print(d_k)

    #3.-TEST LABEL
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

    y_test = y_test.reshape(np.shape(y_test)[1],1)

    return y_test

def reductionKNNAlgorithm(i):

    return i
