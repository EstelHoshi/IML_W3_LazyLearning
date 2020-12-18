import pandas as pd


def best_model_parameter(data, param, metric):
    # Best accuracy grouped for each value in the evaluated parameter
    df_param = data.groupby(param)[metric].max().reset_index()
    df_param = pd.merge(data, df_param, how='inner', on=[param, metric])
    df_param = df_param[df_param['k'] != 0]

    return df_param


def best_model(data, metric):
    # Best global accuracy of the given data
    max_metric = data[metric].max()
    df_best = data[data[metric] == max_metric]

    return df_best

# Generate data
df_gs_wo_relieff = pd.read_csv('satimage_grid_search_results.csv')
df_gs_relieff = pd.read_csv('satimage_relieff_grid_search_results.csv')

df_gs_all = pd.concat([df_gs_wo_relieff, df_gs_relieff], axis=0)
df_gs_all = df_gs_all.sort_values(by='accuracy', ascending=False)


df_gs_minmax = pd.read_csv('satimage_grid_search_results_minmax_weight_normalization.csv')
df_gs_minmax = df_gs_minmax.sort_values(by='accuracy', ascending=False)


# Best Model per each K parameter
df_gs_all_K = best_model_parameter(df_gs_all, 'k', 'accuracy')

# Best Model per each voting
df_gs_all_vote = best_model_parameter(df_gs_all, 'v', 'accuracy')

# Best Model per each distance
df_gs_all_dist = best_model_parameter(df_gs_all, 'p', 'accuracy')

# Best Model per each weighting strategy
df_gs_all_w = best_model_parameter(df_gs_all, 'w', 'accuracy')

# Best overall model
df_gs_all_best = best_model(df_gs_all, 'accuracy')
