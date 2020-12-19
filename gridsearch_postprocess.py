import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, ttest_ind
import itertools


def param_best_model(data, param, metric, tie_metric='no'):
    # Best accuracy grouped for each value in the evaluated parameter
    df_param = data.groupby(param)[metric].max().reset_index()
    df_param = pd.merge(data, df_param, how='inner', on=[param, metric])

    # Check ties:
    num_params = df_param[metric].nunique()
    num_results = df_param[metric].count()

    # In case of tie, break it using tie metric
    if tie_metric != 'no' and num_results > num_params:
        df_untie = df_param.groupby(param)[tie_metric].min().reset_index()
        df_param = pd.merge(df_param, df_untie, how='inner', on=[param, tie_metric])

    return df_param


def get_folds_model(data_folds, data_param, params):
    df_folds = pd.merge(data_folds, data_param[params], how='inner', on=params)

    return df_folds


def global_best_model(data, metric):
    # Best global accuracy of the given data
    max_metric = data[metric].max()
    df_best = data[data[metric] == max_metric]

    return df_best


def get_model(data):
    data['model'] = data['k'].astype(str) + '-' + data['distance'].astype(str) +\
                    '-' + data['policy'].astype(str) + '-' + data['weighting'].astype(str)
    return data


def get_numpy_folds(data_folds, param, metric):
    param_list = sorted(data_folds[param].unique())
    metric_dict = {}

    for par in param_list:
        metric_dict[par] = data_folds[data_folds[param] == par][metric].to_numpy()

    return metric_dict


def hypothesis_test(models):
    # Strategy:
    # 1) perform a Friedman Test among multiple classifiers
    # 2) Check Null Hypothesis with p-values
    # 3) If Null Hypothesis rejected, perform Wilcoxon test 1-1
    model_list = list(models.values())

    if len(model_list) == 3:
        s, p = friedmanchisquare(model_list[0], model_list[1], model_list[2])
    else:
        s, p = friedmanchisquare(model_list[0], model_list[1], model_list[2], model_list[3])

    print('Friedman Test')
    print('p-value = ', p)
    if p < 0.05:
        print('Reject Null Hypothesis: not all the performances are under same distribution')

        print('\n')
        print('Wilcoxon Test: ')
        models_comb = generate_combinations(models)
        for comb in models_comb:
            print(comb)
            s_wx, p_wx = wilcoxon(models[comb[0]], models[comb[1]])
            print('p_value = ', p_wx)
            if p_wx < 0.05:
                print('Reject Null Hypothesis: statistical significant difference among model performance')
            else:
                print('Confirmed Null Hypothesis: no statistical significant difference among different classifiers ')

    else:
        print('Confirmed Null Hypothesis: no statistical significant difference among different classifiers ')

    print('-----------------------------')
    print('\n')


def generate_combinations(param):
    generator = itertools.combinations(param.keys(), 2)

    combs = []
    for comb in generator:
        combs.append(comb)

    return combs

# Read Results
df_gs_cv = pd.read_csv('results_credita_cv.csv')
df_gs_folds = pd.read_csv('results_credita_folds.csv')
df_reduction_cv = pd.read_csv('results_credita_reduction.csv')
df_reduction_folds = pd.read_csv('results_credita_reduction_folds.csv')


# Best Model per each parameter [k, policy, distance, weighting]
df_gs_cv_k = param_best_model(df_gs_cv, 'k', 'accuracy', 'efficiency')
df_gs_cv_pol = param_best_model(df_gs_cv, 'policy', 'accuracy', 'efficiency')
df_gs_cv_dist = param_best_model(df_gs_cv, 'distance', 'accuracy', 'efficiency')
df_gs_cv_w = param_best_model(df_gs_cv, 'weighting', 'accuracy', 'efficiency')

# Best overall model
df_gs_cv_best = global_best_model(df_gs_cv, 'accuracy')

# Folds associated to best models per each parameter [k, policy, distance, weighting]
params = ['k', 'policy', 'distance', 'weighting']
df_gs_folds_k = get_folds_model(df_gs_folds, df_gs_cv_k, params).sort_values(by=['k', 'fold'])
df_gs_folds_pol = get_folds_model(df_gs_folds, df_gs_cv_pol, params).sort_values(by=['k', 'fold'])
df_gs_folds_dist = get_folds_model(df_gs_folds, df_gs_cv_dist, params).sort_values(by=['k', 'fold'])
df_gs_folds_w = get_folds_model(df_gs_folds, df_gs_cv_w, params).sort_values(by=['k', 'fold'])

# Hypothesis test

# Best kNN algorithm
accuracy_folds_k = get_numpy_folds(get_model(df_gs_folds_k), 'model', 'accuracy')
accuracy_folds_pol = get_numpy_folds(get_model(df_gs_folds_pol), 'model', 'accuracy')
accuracy_folds_dist = get_numpy_folds(get_model(df_gs_folds_dist), 'model', 'accuracy')
accuracy_folds_w = get_numpy_folds(get_model(df_gs_folds_w), 'model', 'accuracy')

hypothesis_test(accuracy_folds_k)
hypothesis_test(accuracy_folds_pol)
hypothesis_test(accuracy_folds_dist)
hypothesis_test(accuracy_folds_w)

# Comparison Best vs. with reduction algorithms
df_reduction_folds = df_reduction_folds.sort_values(by=['algorithm', 'dataset'])
accuracy_folds_reduced = get_numpy_folds(df_reduction_folds, 'algorithm', 'accuracy')
hypothesis_test(accuracy_folds_reduced)