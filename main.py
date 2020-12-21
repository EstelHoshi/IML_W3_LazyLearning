import click
from click import Choice
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

from datasetsCBR import load_credita, load_satimage, K_FOLDS
from classification import kNNAlgorithm
from reduction import reductionKNNAlgorithm
from model_selection import GridSearchCV, cross_validate
from analysis import (param_best_model, global_best_model, get_isolated_param,
                      get_model, get_numpy_folds, get_folds_model, hypothesis_test)


@click.group()
def cli():
    pass


# -----------------------------------------------------------------------------------------
# Run kNN with the specified parameters
@cli.command('kNN', help='Run kNN with the specified parameters')
@click.option('-d', '--dataset', type=Choice(['satimage', 'credita']), default='satimage',
              help='Dataset name')
@click.option('-k', type=int, default=5, help='Value k for the nearest neighours to consider')
@click.option('-s', '--similarity', type=Choice(['1-norm', '2-norm', 'chebyshev']), default='2-norm', 
              help='Distance / similarity function')
@click.option('-p', '--policy', type=Choice(['majority', 'inverse', 'sheppard']), default='majority',
              help='Policy for deciding the solution of a query')
@click.option('-w', '--weighting', type=Choice(['uniform', 'mutual_info', 'relief']), default='uniform',
              help='Method to weight features')
@click.option('-r', '--reduction', type=Choice(['no', 'drop2', 'drop3', 'snn', 'enn']), default='no',
              help='Method to reduce the number of instances')
def kNN(dataset, k, similarity, policy, weighting, reduction):
    if dataset == 'credita':
        cv_splits = load_credita(weighting=weighting)

    elif dataset == 'satimage':
        cv_splits = load_satimage(weighting=weighting)

    # perform reduction
    if reduction != 'no':
        # reduce each fold separately
        saved_inst_count = 0

        t0 = time.time()
        for i in tqdm(range(len(cv_splits))):   # progress bar
            X_train, X_test, y_train, y_test = cv_splits[i]
            X_redc, y_redc = reductionKNNAlgorithm(X_train, y_train, k=k, algorithm=reduction,
                                                   distance=similarity, policy=policy)
            cv_splits[i] = X_redc, X_test, y_redc, y_test
            saved_inst_count += X_redc.shape[0]

        # print performance measures
        reduce_efficiency = (time.time() - t0) / len(cv_splits)
        print('reduction efficiency: {}s'.format(round(reduce_efficiency, 2)))

        avg_redc = saved_inst_count / len(cv_splits)
        n = X_train.shape[0] + X_test.shape[0]
        print('reduction in storage: {}/{} ({}%)'.format(round(avg_redc, 2), n, round(avg_redc * 100 / n, 2)))
        

    stats = cross_validate(cv_splits, k=k, distance=similarity, policy=policy)

    print('accuracy:', round(stats[0], 6))
    print(f'efficiency: {round(stats[1], 6)}s')


# -----------------------------------------------------------------------------------------
# Perform a grid search over all possible combinations of parameters for kNN
@cli.command('gridsearch', help='Perform a grid search over all possible combinations of '
             'parameters for kNN')
@click.option('-d', '--dataset', type=Choice(['satimage', 'credita']), default='credita',
              help='Dataset name')
@click.option('-o', '--out', type=str, default=None, help='Output file name')
@click.option('-cv', type=str, default='yes', help='Whether to compute the average over k-folds or '
              'consider each fold as a separate dataset')
def gridsearch(dataset, out, cv):
    if dataset == 'credita':
        cv_splits_all = []
        cv_splits_all.append(load_credita(weighting=None))
        cv_splits_all.append(load_credita(weighting='mutual_info'))
        cv_splits_all.append(load_credita(weighting='relief'))

    elif dataset == 'satimage':
        cv_splits_all = []
        cv_splits_all.append(load_satimage(weighting=None))
        cv_splits_all.append(load_satimage(weighting='mutual_info'))
        cv_splits_all.append(load_satimage(weighting='relief'))

    if cv == 'no':
        # flatten, squeeze so as to perform gridsearch on each fold separately
        cv_splits_all = [[split] for cv_splits in cv_splits_all for split in cv_splits]

    param_grid = {
        'k': [1, 3, 5, 7],
        'distance': ['1-norm', '2-norm', 'chebyshev'],
        'policy': ['majority', 'inverse', 'sheppard']
    }

    # run gridsearch and accumulate results
    history = pd.DataFrame()

    with tqdm(total=len(cv_splits_all)) as pbar: # progress bar

        for split in cv_splits_all:
            h = GridSearchCV(split, param_grid)
            history = history.append(h, ignore_index=True)
            pbar.update()

    # tag results with corresponding weighting methods
    m = len(history) // 3
    history.loc[:m*1, 'weighting'] = 'uniform'
    history.loc[m*1:m*2, 'weighting'] = 'mutual_info'
    history.loc[m*2:, 'weighting'] = 'relief'

    if cv == 'no':
        # tag fold number: first 36 to 0, next 36 to 1, ...
        c = 4 * 3 * 3   # number of results per one fold
        history['fold'] = dataset + '-' +  ((history.index // c) % K_FOLDS).astype(str)

    # print best
    history = history.sort_values('accuracy')
    print('\nbest:')
    print(history.iloc[-1])

    # save history df
    if out is not None:
        if not out.endswith('.csv'):
            out += '.csv'
        history.to_csv(out, index=False)


# -----------------------------------------------------------------------------------------
# Test all reduction algorithms for a specific kNN model
@cli.command('test-reduction', help='Test all reduction algorithms for a specific kNN model')
@click.option('-d', '--dataset', type=Choice(['satimage', 'credita']), default='credita',
              help='Dataset name')
@click.option('-k', type=int, default=3, help='Value k for the nearest neighours to consider')
@click.option('-s', '--similarity', type=Choice(['1-norm', '2-norm', 'chebyshev']), default='2-norm', 
              help='Distance / similarity function')
@click.option('-p', '--policy', type=Choice(['majority', 'inverse', 'sheppard']), default='majority',
              help='Policy for deciding the solution of a query')
@click.option('-w', '--weighting', type=Choice(['uniform', 'mutual_info', 'relief']), default='uniform',
              help='Method to weight features')
@click.option('-o', '--out', type=str, default=None, help='Output file name')
@click.option('-cv', type=str, default='yes', help='Whether to compute the average over k-folds or '
              'consider each fold as a separate dataset')
def test_reduction(dataset, k, similarity, policy, weighting, out, cv):
    if dataset == 'credita':
        cv_splits = load_credita(weighting=weighting)

    elif dataset == 'satimage':
        cv_splits = load_satimage(weighting=weighting)

    # perform reduction
    history = pd.DataFrame()
    reductions = ['no', 'enn', 'snn', 'drop2']
    pbar = tqdm(total=len(reductions) * len(cv_splits))

    for reduction in reductions:
        partial_hist = pd.DataFrame()

        # run reduction reduction on each fold and store performance metrics
        for X_train, X_test, y_train, y_test in cv_splits:   # progress bar
            stats = pd.Series(dtype=np.float)

            t0 = time.time()
            X_redc, y_redc = reductionKNNAlgorithm(X_train, y_train, k=k, algorithm=reduction,
                                                    distance=similarity, policy=policy)
            stats['reduce_efficiency'] = time.time() - t0
            stats['percentage'] = X_redc.shape[0] * 100 / X_train.shape[0]

            acc, eff = cross_validate([(X_redc, X_test, y_redc, y_test)], k=k, distance=similarity, policy=policy)
            stats['accuracy'] = acc
            stats['efficiency'] = eff

            partial_hist = partial_hist.append(stats, ignore_index=True)
            pbar.update()

        if cv == 'no':
            # tag result of each fold with a separate name
            partial_hist['dataset'] = [dataset + '-' + str(i) for i in range(len(cv_splits))]

        else:
            # average results across folds
            partial_hist = partial_hist.mean()
            partial_hist['dataset'] = dataset

        partial_hist['algorithm'] = reduction
        history = history.append(partial_hist, ignore_index=True)

    pbar.close()

    # print best
    history = history.sort_values('accuracy')
    print('\nbest:')
    print(history.iloc[-1])

    # save history df
    if out is not None:
        if not out.endswith('.csv'):
            out += '.csv'
        history.to_csv(out, index=False)


# -----------------------------------------------------------------------------------------
# Perform a statistical analysis comparing the best model for each isolated parameter
@cli.command('analyze', help='Perform a statistical analysis comparing the best model for each '
             'isolated parameter using the results obtained from the other commands')
@click.option('-g', '--gridsearch-results-folds', type=str, help='Path to file with gridsearch results'
              ' by folds')
@click.option('-G', '--gridsearch-results-cv', type=str, help='Path to file with gridsearch results '
              'with cross-validation')
@click.option('-r', '--reduction-results-folds', type=str, help='Path to file with reduction results '
              'by folds')
def analyze(gridsearch_results_folds, gridsearch_results_cv, reduction_results_folds):
    # Read Results
    df_gs_cv = pd.read_csv(gridsearch_results_cv)
    df_gs_folds = pd.read_csv(gridsearch_results_folds)
    # df_reduction_cv = pd.read_csv('results_credita_reduction.csv')
    df_reduction_folds = pd.read_csv(reduction_results_folds)

    # Best Model per each parameter in isolation [k, policy, distance, weighting]
    df_gs_cv_k = param_best_model(get_isolated_param(df_gs_cv, 'k'), 'k', 'accuracy', 'efficiency')
    df_gs_cv_pol = param_best_model(get_isolated_param(df_gs_cv, 'policy'), 'policy', 'accuracy', 'efficiency')
    df_gs_cv_dist = param_best_model(get_isolated_param(df_gs_cv, 'distance'), 'distance', 'accuracy', 'efficiency')
    df_gs_cv_w = param_best_model(get_isolated_param(df_gs_cv, 'weighting'), 'weighting', 'accuracy', 'efficiency')

    df_best_k = global_best_model(df_gs_cv_k, 'accuracy')
    df_best_pol = global_best_model(df_gs_cv_pol, 'accuracy')
    df_best_dist = global_best_model(df_gs_cv_dist, 'accuracy')
    df_best_w = global_best_model(df_gs_cv_w, 'accuracy')

    # Best overall model
    df_gs_cv_best = global_best_model(df_gs_cv, 'accuracy')

    # Folds associated to best models per each parameter [k, policy, distance, weighting]
    params = ['k', 'policy', 'distance', 'weighting']
    df_gs_folds_k = get_folds_model(df_gs_folds, df_gs_cv_k, params).sort_values(by=['k', 'fold'])
    df_gs_folds_pol = get_folds_model(df_gs_folds, df_gs_cv_pol, params).sort_values(by=['k', 'fold'])
    df_gs_folds_dist = get_folds_model(df_gs_folds, df_gs_cv_dist, params).sort_values(by=['k', 'fold'])
    df_gs_folds_w = get_folds_model(df_gs_folds, df_gs_cv_w, params).sort_values(by=['k', 'fold'])

    # Folds associated to the best models (combined and isolated)
    df_gs_folds_best = get_folds_model(df_gs_folds, df_gs_cv_best, params).sort_values(by=['k', 'fold'])
    df_gs_folds_k_best = get_folds_model(df_gs_folds, df_best_k, params).sort_values(by=['k', 'fold'])
    df_gs_folds_pol_best = get_folds_model(df_gs_folds, df_best_pol, params).sort_values(by=['k', 'fold'])
    df_gs_folds_dist_best = get_folds_model(df_gs_folds, df_best_dist, params).sort_values(by=['k', 'fold'])
    df_gs_folds_w_best = get_folds_model(df_gs_folds, df_best_w, params).sort_values(by=['k', 'fold'])

    # Hypothesis test

    # Hypothesis test per each parameter in isolation
    accuracy_folds_k = get_numpy_folds(get_model(df_gs_folds_k), 'model', 'accuracy')
    accuracy_folds_pol = get_numpy_folds(get_model(df_gs_folds_pol), 'model', 'accuracy')
    accuracy_folds_dist = get_numpy_folds(get_model(df_gs_folds_dist), 'model', 'accuracy')
    accuracy_folds_w = get_numpy_folds(get_model(df_gs_folds_w), 'model', 'accuracy')

    print('\n> BEST K\n')
    hypothesis_test(accuracy_folds_k)

    print('> BEST POLICY\n')
    hypothesis_test(accuracy_folds_pol)

    print('> BEST DISTANCE\n')
    hypothesis_test(accuracy_folds_dist)

    print('> BEST WEIGHTING\n')
    hypothesis_test(accuracy_folds_w)

    # Best kNN algorithm
    best_models = [get_model(df_gs_folds_best), get_model(df_gs_folds_k_best), get_model(df_gs_folds_pol_best),
                   get_model(df_gs_folds_dist_best), get_model(df_gs_folds_w_best)]
    df_best_models = pd.concat(best_models, axis=0)
    accuracy_folds_best = get_numpy_folds(df_best_models, 'model', 'accuracy')

    print('\n> BEST KNN\n')
    hypothesis_test(accuracy_folds_best)

    # Comparison Best vs. with reduction algorithms
    df_reduction_folds = df_reduction_folds.sort_values(by=['algorithm', 'dataset'])
    accuracy_folds_reduced = get_numpy_folds(df_reduction_folds, 'algorithm', 'accuracy')
    print('> BEST REDUCTION\n')
    hypothesis_test(accuracy_folds_reduced)



# -----------------------------------------------------------------------------------------
# Plot a graph comparing a specified parameter across different models measuring some metrics
@cli.command('compare', help='Plot a graph comparing a specified parameter across different '
             'models measuring some metrics')
@click.option('-p', '--parameter', type=Choice(['k', 'distance', 'policy', 'weighting', 'algorithm']),
              help='Parameter to compare')
@click.option('-m', '--metrics', type=str, multiple=True, help='Metrics to evaluate')
@click.option('-f', '--results-file', type=str, help='Path to file with results to compare')
@click.option('-t', '--title', type=str, default=None, help='Title of the plot')
@click.option('-x', '--x-label', type=str, default=None, help='Name of x axis')
@click.option('-y', '--y-label', type=str, default=None, help='Name of y axis')
def compare(parameter, metrics, results_file, title, x_label, y_label):
    df_res = pd.read_csv(results_file)
    metrics = list(metrics)

    values = df_res.groupby(parameter).mean().sort_values(metrics[0])
    print(values)

    ax = plt.axes()

    x = values.index
    for metric in metrics:
        y = values[metric]
        plt.bar(x, y, label=metric)

    if x_label is None:
        plt.xlabel(parameter)
    else:
        plt.xlabel(x_label)

    if y_label is None:
        plt.ylabel(metrics[0])
    else:
        plt.ylabel(y_label)

    if title is None:
        plt.title(f'Evaluating parameter \'{parameter}\'')
    else:
        plt.title(title)
        
    plt.xticks(np.unique(x))
    ax.yaxis.grid(color='0.85')
    ax.set_axisbelow(True)
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    cli()