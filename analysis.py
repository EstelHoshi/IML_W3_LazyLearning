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


def get_isolated_param(data, param):
    if param == 'k':
        data = data[(data['policy'] == 'majority') & (data['distance'] == '2-norm')
                    & (data['weighting'] == 'uniform')]
    elif param == 'policy':
        data = data[(data['k'] == 5) & (data['distance'] == '2-norm')
                    & (data['weighting'] == 'uniform')]
    elif param == 'distance':
        data = data[(data['k'] == 5) & (data['policy'] == 'majority')
                    & (data['weighting'] == 'uniform')]
    elif param == 'weighting':
        data = data[(data['k'] == 5) & (data['policy'] == 'majority')
                    & (data['distance'] == '2-norm')]

    return data


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
    elif len(model_list) == 4:
        s, p = friedmanchisquare(model_list[0], model_list[1], model_list[2], model_list[3])
    else:
        s, p = friedmanchisquare(model_list[0], model_list[1], model_list[2], model_list[3], model_list[4])

    print('Friedman Test:')
    print('p-value = ', round(p, 6))
    if p < 0.05:
        print('Reject Null Hypothesis: not all the performances are under same distribution')

        print('\n')
        print('Wilcoxon Test: ')
        models_comb = generate_combinations(models)
        for comb in models_comb:
            print(comb)
            s_wx, p_wx = wilcoxon(models[comb[0]], models[comb[1]])
            print('p_value = ', round(p_wx, 6))
            if p_wx < 0.05:
                print('Reject Null Hypothesis: statistical significant difference among model performance\n')
            else:
                print('Confirmed Null Hypothesis: no statistical significant difference among different classifiers\n')

    else:
        print('Confirmed Null Hypothesis: no statistical significant difference among different classifiers ')

    print('_____________________________')
    print('\n')


def generate_combinations(param):
    generator = itertools.combinations(param.keys(), 2)

    combs = []
    for comb in generator:
        combs.append(comb)

    return combs