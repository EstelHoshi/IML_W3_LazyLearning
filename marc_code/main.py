import time
import numpy as np
import pandas as pd
import pre_processing
import utils
import kNN
import editedNN
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import ttest_ind, ttest_rel, wilcoxon, friedmanchisquare


def main_kropt(k, p, v, w, edition):
    folds = sorted(os.listdir('datasetsCBR/kropt'))
    fold_tuple = [(folds[2*i+1], folds[2*i]) for i in range(10)]

    scaler, le, weights = pre_processing.obtain_pp('datasetsCBR/kropt/kropt.fold.000000.train.arff',
                                                   'datasetsCBR/kropt/kropt.fold.000000.test.arff',
                                                   weighting=w)
    accuracy_fold = []
    accuracy_sk_fold = []
    for fold in fold_tuple:
        X_train, y_train = pre_processing.apply_pp(fold[0], le, scaler, weights)
        X_test, y_test = pre_processing.apply_pp(fold[1], le, scaler, weights)

        if edition == 'ENN':
            X_train_edited, y_train_edited = editedNN.ENN(X_train, y_train, k, p)
            lazy = kNN.kNNAlgorithm(k, p, X_train_edited, X_test, y_train_edited)
        elif edition == 'ENNTh':
            X_train_edited, y_train_edited = editedNN.ENNTh(X_train, y_train, k, p, 1)
            lazy = kNN.kNNAlgorithm(k, p, X_train_edited, X_test, y_train_edited)
        else:
            lazy = kNN.kNNAlgorithm(k, p, X_train, X_test, y_train)

        if v == 'max':
            y_pred = lazy.max_voting()
        elif v == 'invd':
            y_pred = lazy.inverse_distance_voting()
        elif v == 'sheppard':
            y_pred = lazy.sheppard_voting()
        else:
            y_pred = lazy.max_voting()

        neigh = KNeighborsClassifier(n_neighbors=k, p=p)
        neigh.fit(X_train, y_train)
        y_pred_sk = neigh.predict(X_test)

        # Save fold accuracy
        accuracy_fold.append(accuracy_score(y_test, y_pred))
        accuracy_sk_fold.append(accuracy_score(y_test, y_pred_sk))

    # Obtain average accuracy
    accuracy = sum(accuracy_fold) / len(accuracy_fold)
    accuracy_sk = sum(accuracy_sk_fold) / len(accuracy_sk_fold)

    return accuracy, accuracy_sk


def main_satimage_v2(k, p, v, w, edition):
    folds = sorted(os.listdir('datasetsCBR/satimage'))
    fold_tuple = [(folds[2 * i + 1], folds[2 * i]) for i in range(10)]

    scaler, le, weights, X, Y = pre_processing.obtain_pp('datasetsCBR/satimage/satimage.fold.000000.train.arff',
                                                         'datasetsCBR/satimage/satimage.fold.000000.test.arff',
                                                         weighting=w)

    if edition == 'ENN':
        edited_X, edited_Y = editedNN.get_ENN_pp(X, Y, k, p)

    accuracy_fold = []
    accuracy_sk_fold = []
    time_fold_pp = []
    time_fold_test = []

    for fold in fold_tuple:
        start_time_pp = time.time()
        X_train, y_train = pre_processing.apply_pp(fold[0], le, scaler, weights)
        X_test, y_test = pre_processing.apply_pp(fold[1], le, scaler, weights)

        if edition == 'ENN':
            X_train_edited, y_train_edited = editedNN.apply_ENN(edited_X, X_train, y_train)
            lazy = kNN.kNNAlgorithm(k, p, X_train_edited, X_test, y_train_edited)
        elif edition == 'ENNTh':
            X_train_edited, y_train_edited = editedNN.ENNTh(X_train, y_train, k, p, 1)
            lazy = kNN.kNNAlgorithm(k, p, X_train_edited, X_test, y_train_edited)
        else:
            lazy = kNN.kNNAlgorithm(k, p, X_train, X_test, y_train)

        # Save preprocessing time
        time_fold_pp.append(float(time.time() - start_time_pp))

        start_time_test = time.time()
        if v == 'max':
            y_pred = lazy.max_voting()
        elif v == 'invd':
            y_pred = lazy.inverse_distance_voting()
        elif v == 'sheppard':
            y_pred = lazy.sheppard_voting()
        else:
            y_pred = lazy.max_voting()

        # Save test time
        time_fold_test.append(float(time.time() - start_time_test))
        print("--- %s seconds ---" % (time.time() - start_time_test))

        if p == 'inf':
            neigh = KNeighborsClassifier(n_neighbors=k, p=99999)
        else:
            neigh = KNeighborsClassifier(n_neighbors=k, p=p)
        neigh.fit(X_train, y_train)
        y_pred_sk = neigh.predict(X_test)

        # Save fold accuracy
        accuracy_fold.append(accuracy_score(y_test, y_pred))
        accuracy_sk_fold.append(accuracy_score(y_test, y_pred_sk))

    # Obtain average accuracy and times
    accuracy = sum(accuracy_fold)/len(accuracy_fold)
    accuracy_sk = sum(accuracy_sk_fold)/len(accuracy_sk_fold)
    time_pp = sum(time_fold_pp) / len(time_fold_pp)
    time_test = sum(time_fold_test) / len(time_fold_test)

    print(accuracy)

    return accuracy, accuracy_sk, time_pp, time_test, accuracy_fold


def grid_search():
    ks = [1, 3, 5, 7]
    ps = [1, 2, 'inf']
    vs = ['max', 'invd', 'sheppard']
    ws = ['uniform', 'mutual_info', 'chi2', 'relieff']

    columns = ['k', 'p', 'v', 'w', 'accuracy', 'accuracy_sk', 'time_pp', 'time_test']
    data = np.zeros((1, 8))
    df_results = pd.DataFrame(data=data, columns=columns)

    for k in ks:
        for p in ps:
            for v in vs:
                for w in ws:
                    print(k, p, v, w)
                    accuracy, accuracy_sk, time_pp, time_test, accuracy_fold = main_satimage_v2(k, p, v, w, 0)
                    data = np.array([k, p, v, w, accuracy, accuracy_sk, time_pp, time_test])[np.newaxis, :]
                    df_results_aux = pd.DataFrame(data=data, columns=columns)
                    df_results = pd.concat([df_results, df_results_aux], axis=0)

    df_results.to_csv('satimage_grid_search_results_minmax_weight_normalization.csv', index=False)
    print(df_results)


def statistical_test():
    accuracy_folds_top1 = np.asarray(main_satimage_v2(5, 1, 'sheppard', 'mutual_info', 0)[4])
    accuracy_folds_top2 = np.asarray(main_satimage_v2(3, 1, 'sheppard', 'mutual_info', 0)[4])
    accuracy_folds_top3 = np.asarray(main_satimage_v2(5, 1, 'sheppard', 'uniform', 0)[4])
    accuracy_folds_top4 = np.asarray(main_satimage_v2(3, 1, 'max', 'mutual_info', 0)[4])
    accuracy_folds_top5 = np.asarray(main_satimage_v2(5, 1, 'sheppard', 'relieff', 0)[4])
    accuracy_folds_top6 = np.asarray(main_satimage_v2(3, 1, 'invd', 'relieff', 0)[4])
    accuracy_folds_top7 = np.asarray(main_satimage_v2(5, 1, 'invd', 'mutual_info', 0)[4])
    accuracy_folds_top8 = np.asarray(main_satimage_v2(5, 1, 'invd', 'uniform', 0)[4])
    accuracy_folds_top9 = np.asarray(main_satimage_v2(5, 1, 'invd', 'relieff', 0)[4])
    accuracy_folds_top10 = np.asarray(main_satimage_v2(3, 1, 'sheppard', 'relieff', 0)[4])

    # Perform Friedman test (multi classifier)
    # Null Hypothesis: performance under all the classifiers follows same distribution
    fried, p_friedman = friedmanchisquare(accuracy_folds_top1, accuracy_folds_top2, accuracy_folds_top3,
                                          accuracy_folds_top4, accuracy_folds_top5, accuracy_folds_top6,
                                          accuracy_folds_top7, accuracy_folds_top8, accuracy_folds_top9,
                                          accuracy_folds_top10)

    print('\n')
    print('Friedman test among top10 classifiers')
    print('p values: ', str(p_friedman))

    if p_friedman < 0.05:
        print('Reject Null Hypothesis: not all the performances are under same distribution')

        top10 = [accuracy_folds_top2, accuracy_folds_top3, accuracy_folds_top4, accuracy_folds_top5,
                 accuracy_folds_top6, accuracy_folds_top7, accuracy_folds_top8, accuracy_folds_top9,
                 accuracy_folds_top10]

        for i, classifier in enumerate(top10):
            wilc, p_wilcoxon = wilcoxon(accuracy_folds_top1, classifier)

            print('\n')
            print('Wilcoxon test: ' + 'top1 vs. top' + str(i + 2))
            print('p values: ', str(p_wilcoxon))
            if p_wilcoxon < 0.05:
                print('Reject Null Hypothesis: not all the performances are under same distribution')
            else:
                print('Confirmed Null Hypothesis: no difference statistical significant among different classifiers ')

    else:
        print('Confirmed Null Hypothesis: no difference statistical significant among different classifiers ')


if __name__ == '__main__':
    start_time = time.time()
    #main_satimage_v2(5, 1, 'max', 'mutual_info', 'ENN')
    grid_search()
    print("--- %s seconds ---" % (time.time() - start_time))


