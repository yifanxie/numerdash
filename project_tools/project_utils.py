import numpy as np
import pandas as pd
import os
import pickle
import time
from contextlib import contextmanager
from importlib import reload
import re
from project_tools import project_config, project_utils, numerapi_utils
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint, random
import itertools
import scipy
from scipy.stats import ks_2samp
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
import datetime
import json
from collections import OrderedDict
from os import listdir
from os.path import isfile, join, isdir
import glob
import numerapi
import itertools
import io
import requests
from pathlib import Path
from scipy.stats.mstats import gmean
from typing import List, Dict


napi = numerapi.NumerAPI() #verbosity="info")


def get_time_string():
    """
    Generate a time string representation of the time of call of this function.
    :param None
    :return: a string that represent the time of the functional call.
    """
    now = datetime.datetime.now()
    now = str(now.strftime('%Y%m%d%H%M'))
    return now


def reload_project():
    """
    utility function used during experimentation to reload various model when required, useful for quick experiment iteration
    :return: None
    """
    reload(project_config)
    reload(project_utils)
    reload(numerapi_utils)

@contextmanager
def timer(name):
    """
    utility timer function to check how long a piece of code might take to run.
    :param name: name of the code fragment to be timed
    :yield: time taken for the code to run
    """
    t0 = time.time()
    print('[%s] in progress' % name)
    yield
    print('[%s] done in %.6f s' %(name, time.time() - t0))



def load_data(pickle_file):
    """
    load pickle data from file
    :param pickle_file: path of pickle data
    :return: data stored in pickle file
    """
    load_file = open(pickle_file, 'rb')
    data = pickle.load(load_file)
    return data


def pickle_data(path, data, protocol=-1, timestamp=False, verbose=True):
    """
    Pickle data to specified file
    :param path: full path of file where data will be pickled to
    :param data: data to be pickled
    :param protocol: pickle protocol, -1 indicate to use the latest protocol
    :return: None
    """
    file = path
    if timestamp:
        base_file = os.path.splitext(file)[0]
        time_str = '_' + get_time_string()
        ext = os.path.splitext(os.path.basename(file))[1]
        file = base_file + time_str + ext

    if verbose:
        print('creating file %s' % file)

    save_file = open(file, 'wb')
    pickle.dump(data, save_file, protocol=protocol)
    save_file.close()


def save_json(path, data, timestamp=False, verbose=True, indent=2):
    """
    Save data to Json format
    :param path: full path of file where data will be pickled to
    :param data: data to be pickled
    :param timestamp: if true, the timestamp will be saved as part of the file name
    :param verbose: if true, print information about file creation
    :param indent: specify the width of the indent in the resulted Json file
    :return: None
    """
    file = path
    if timestamp:
        base_file = os.path.splitext(file)[0]
        time_str = '_' + get_time_string()
        ext = os.path.splitext(os.path.basename(file))[1]
        file = base_file + time_str + ext
    if verbose:
        print('creating file %s' % file)
    outfile = open(file, 'w')
    json.dump(data, outfile, indent=indent)
    outfile.close()


def load_json(json_file):
    """
    load data from Json file
    :param json_file: path of json file
    :return: data stored in json file as python dictionary
    """
    load_file = open(json_file)
    data = json.load(load_file)
    load_file.close()
    return data


def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)



def glob_folder_filelist(path, file_type='', recursive=True):
    """
    utility function that walk through a given directory, and return list of files in the directory
    :param path: the path of the directory
    :param file_type: if not '', this function would only consider the file type specified by this parameter
    :param recursive: if True, perform directory walk-fhrough recursively
    :return absfile: a list containing absolute path of each file in the directory
    :return base_files: a list containing base name of each file in the directory
    """
    if path[-1] != '/':
        path = path +'/'
    abs_files = []
    base_files = []
    patrn = '**' if recursive else '*'
    glob_path = path + patrn
    matches = glob.glob(glob_path, recursive=recursive)
    for f in matches:
        if os.path.isfile(f):
            include = True
            if len(file_type)>0:
                ext = os.path.splitext(f)[1]
                if ext[1:] != file_type:
                    include = False
            if include:
                abs_files.append(f)
                base_files.append(os.path.basename(f))
    return abs_files, base_files


def dir_compare(pathl, pathr):
    files_pathl = set([f for f in listdir(pathl) if isfile(join(pathl, f))])
    files_pathr = set([f for f in listdir(pathr) if isfile(join(pathr, f))])
    return list(files_pathl-files_pathr), list(files_pathr-files_pathl)




def lr_dir_sync(pathl, pathr):
    files_lrddiff, files_rldiff = project_utils.dir_compare(pathl, pathr)
    for f in files_lrddiff:
        scr = pathl + f
        dst = pathr + f
        print('copying file %s' % scr)
        copyfile(scr, dst)



def copy_file_with_time(src_file, dst_file_name, des_path):
    basename = os.path.splitext(os.path.basename(dst_file_name))[0]
    ext_name = os.path.splitext(os.path.basename(dst_file_name))[1]
    timestr = get_time_string()
    des_name = '%s%s_%s%s' % (des_path, basename, timestr, ext_name)
    # print(des_name)
    copyfile(src_file, des_name)





def find_filesfromfolder(target_dir, containtext):
    absnames, basenames = glob_folder_filelist(target_dir)
    result_filelist = []
    for absname, basename in zip(absnames, basenames):
        if containtext in basename:
            result_filelist.append(absname)
    # result_filelist = [f for f in total_filelist if containtext in f]
    return result_filelist


def cp_files_with_prefix(src_path, dst_path, prefix, ext):
    abs_file_list, base_file_list = get_folder_filelist(src_path, file_type=ext)
#     print(abs_file_list)
    for src_file, base_file in zip(abs_file_list, base_file_list):
        dst_file = dst_path + prefix + base_file
        copyfile(src_file, dst_file)
    return None



def mv_files_with_prefix(src_path, dst_path, prefix, ext):
    abs_file_list, base_file_list = get_folder_filelist(src_path, file_type=ext)
#     print(abs_file_list)
    for src_file, base_file in zip(abs_file_list, base_file_list):
        dst_file = dst_path + prefix + base_file
        move(src_file, dst_file)
    return None



def empty_folder(path):
    if path[-1]!='*':
        path = path + '*'
    files = glob.glob(path)
    for f in files:
        os.remove(f)


def rescale(n, range1, range2):
    if n>range1[1]: #or n<range1[0]:
        n=range1[1]
    if n<range1[0]:
        n=range1[0]
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]



def rmse(y_true, y_pred):
    """
    RMSE (Root Mean Square Error) evaluation function
    :param y_true: label values
    :param y_pred: prediction values
    :return:  RMSE value of the input prediction values, evaluated against the input label values
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))




def str2date(date_str, dateformat='%Y-%m-%d'):
    """
    convert an input string in specified format into datetime format
    :param date_str: the input string with certain specified format
    :param dateformat: the format of the string which is used by the strptime function to do the type converson
    :return dt_value: the datetime value that is corresponding to the input string and the specified format
    """
    dt_value = datetime.datetime.strptime(date_str, dateformat)
    return dt_value


def isnotebook():
    """
    Determine if the current python file is a jupyter notebook (.ipynb) or a python script (.py)
    :return: return True if the the current python file is a jupyter notebook, otherwise return False
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False



def list_intersection(left, right):
    """
    take two list as input, conver them into sets, calculate the intersection of the two sets, and return this as a list
    :param left: the first input list
    :param right: the second input list
    :return: the intersection set of elements for both input list, as a list
    """
    left_set = set(left)
    right_set = set(right)
    return list(left_set.intersection(right_set))


def list_union(left, right):
    """
    take two list as input, conver them into sets, calculate the union of the two sets, and return this as a list
    :param left: the first input list
    :param right: the second input list
    :return: the union set of elements for both input list, as a list
    """
    left_set = set(left)
    right_set = set(right)
    return list(left_set.union(right_set))


def list_difference(left, right):
    """
    take two list as input, conver them into sets, calculate the difference of the first set to the second set, and return this as a list
    :param left: the first input list
    :param right: the second input list
    :return: the result of difference set operation on elements for both input list, as a list
    """
    left_set = set(left)
    right_set = set(right)
    return list(left_set.difference(right_set))


def is_listelements_identical(left, right):
    equal_length = (len(left)==len(right))
    zero_diff = (len(list_difference(left,right))==0)
    return equal_length & zero_diff




def np_corr(a, b):
    """
    take two numpy arrays, and compute their correlation
    :param a: the first numpy array input
    :param b: the second numpy array input
    :return: the correlation between the two input arrays
    """
    return pd.Series(a).corr(pd.Series(b))



def list_sort_values(a, ascending=True):
    """
    sort the value of a list in specified order
    :param a: the input list
    :param ascending: specified if the sorting is to be done in ascending or descending order
    :return: the input list sorted in the specified order
    """
    return pd.Series(a).sort_values(ascending=ascending).tolist()


def get_rank(data):
    """
    convert the values of a list or array into ranked percentage values
    :param data: the input data in the form of a list or an array
    :return: the return ranked percentage values in numpy array
    """
    ranks = pd.Series(data).rank(pct=True).values
    return ranks



def plot_feature_corr(df, features, figsize=(10,10), vmin=-1.0):
    """
    plot the pair-wise correlation matrix for specified features in a dataframe
    :param df: the input dataframe
    :param features: the list of features for which correlation matrix will be plotted
    :param figsize: the size of the displayed figure
    :param vmin: the minimum value of the correlation to be included in the plotting
    :return: the pair-wise correlation values in the form of pandas dataframe, the figure will be plotted during the operation of this function.
    """
    val_corr = df[features].corr().fillna(0)
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(val_corr, vmin=vmin, square=True)
    return val_corr


def decision_to_prob(data):
    """
    convert output value of a sklearn classifier (i.e. ridge classifier) decision function into probability
    :param data: output value of decision function in the form of a numpy array
    :return: value of probability in the form of a numpy array
    """
    prob = np.exp(data) / np.sum(np.exp(data))
    return prob


def np_describe(a):
    """
    provide overall statistic description of an input numpy value using the Describe method of Pandas Series
    :param a: the input numpy array
    :return: overall statistic description
    """
    return pd.Series(a.flatten()).describe()


def ks_2samp_selection(train_df, test_df, pval=0.1):
    """
    use scipy ks_2samp function to select features that are statistically similar between the input train and test dataframe.
    :param train_df: the input train dataframe
    :param test_df: the input test dataframe
    :param pval: the p value threshold use to decide which features to be selected. Only features with value higher than the specified p value will be selected
    :return train_df: the return train dataframe with selected features
    :return test_df: the return test dataframe with selected features
    """
    list_p_value = []
    for i in train_df.columns.tolist():
        list_p_value.append(ks_2samp(train_df[i], test_df[i])[1])
    Se = pd.Series(list_p_value, index=train_df.columns.tolist()).sort_values()
    list_discarded = list(Se[Se < pval].index)
    train_df = train_df.drop(columns=list_discarded)
    test_df = test_df.drop(columns=list_discarded)
    return train_df, test_df



def df_balance_sampling(df, class_feature, minor_class=1, sample_ratio=1):
    """
    :param df:
    :param class_feature:
    :param minor_class:
    :param sample_ratio:
    :return:
    """
    minor_df = df[df[class_feature] == minor_class]
    major_df = df[df[class_feature] == (1 - minor_class)].sample(sample_ratio * len(minor_df))

    res_df = minor_df.append(major_df)
    res_df = res_df.sample(len(res_df)).reset_index(drop=True)
    return res_df


def prob2acc(label, probs, p=0.5):
    """
    calculate accuracy score  for probability predictions  with given threshold, as part of the process, the input probability predictions will be converted into discrete binary predictions
    :param label: labels used to evaluate accuracy score
    :param probs: probability predictions for which accuracy score will be calculated
    :param p: the threshold to be used for convert probabilites into discrete binary values 0 and 1
    :return acc: the computed accuracy score
    :return preds: predictions in discrete binary value
    """

    preds = (probs >= p).astype(np.uint8)
    acc = accuracy_score(label, preds)
    return acc, preds



def np_pearson(t,p):
    vt = t - t.mean()
    vp = p - p.mean()
    top = np.sum(vt*vp)
    bottom = np.sqrt(np.sum(vt**2)) * np.sqrt(np.sum(vp**2))
    res = top/bottom
    return res


def df_get_features_with_str(df, ptrn):
    """
    extract list of feature names from a data frame that contain the specified regular expression pattern
    :param df: the input dataframe of which features name to be analysed
    :param ptrn: the specified regular expression pattern
    :return: list of feature names that contained the specified regular expression
    """
    return [col for col in df.columns.tolist() if len(re.findall(ptrn, col)) > 0]


def df_fillna_with_other(df, src_feature, dst_feature):
    """
    fill the NA values of a specified feature in a dataframe with values of another feature from the same row.
    :param df: the input dataframe
    :param src_feature: the specified feature of which NA value will be filled
    :param dst_feature: the feature of which values will be used
    :return: a dataframe with the specified feature's NA value being filled by values from the "dst_feature"
    """
    src_vals = df[src_feature].values
    dst_vals = df[dst_feature].values
    argwhere_nan = np.argwhere(np.isnan(dst_vals)).flatten()
    dst_vals[argwhere_nan] = src_vals[argwhere_nan]
    df[dst_feature] = dst_vals
    return df



def plot_prediction_prob(y_pred_prob):
    """
    plot probability prediction values using histrogram
    :param y_pred_prob: the probability prediction values to be plotted
    :return: None, the plot will be plotted during the operation of the function.
    """
    prob_series = pd.Series(data=y_pred_prob)
    prob_series.name = 'prediction probability'
    prob_series.plot(kind='hist', figsize=(15, 5), bins=50)
    plt.show()
    print(prob_series.describe())





def df_traintest_split(df, split_var, seed=None, train_ratio=0.75):
    """
    perform train test split on a specified feature on a given dataframe wwith specified train ratio. Unique value of the specified feature will only present on either the resulted train or the test dataframe
    :param df: the input dataframe to be split
    :param split_var: the feature to be used as unique value to perform the split
    :param seed: the random used to facilitate the train test split
    :param train_ratio: the ratio of data to be split into the resulted train dataframe.
    :return train_df: the resulted train dataframe after the split
    :return test_df: the resulted test dataframe after the split
    """
    sv_list = df[split_var].unique().tolist()
    train_length = int(len(sv_list) * train_ratio)
    train_siv_list = pd.Series(df[split_var].unique()).sample(train_length, random_state=seed)
    train_idx = df.loc[df[split_var].isin(train_siv_list)].index.values
    test_idx = df.iloc[df.index.difference(train_idx)].index.values
    train_df = df.loc[train_idx].copy().reset_index(drop=True)
    test_df = df.loc[test_idx].copy().reset_index(drop=True)
    return train_df, test_df



# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df, verbose=True, exceiptions=[]):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    np_input = False
    if isinstance(df, np.ndarray):
        np_input = True
        df = pd.DataFrame(data=df)

    start_mem = df.memory_usage().sum() / 1024 ** 2
    col_id = 0
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        if verbose: print('doing %d: %s' % (col_id, col))
        col_type = df[col].dtype
        try:
            if (col_type != object) & (col not in exceiptions):
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        #                         df[col] = df[col].astype(np.float16)
                        #                     elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        #             else:
        #                 df[col] = df[col].astype('category')
        #                 pass
        except:
            pass
        col_id += 1
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    if np_input:
        return df.values
    else:
        return df



def get_xgb_featimp(model):
    imp_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    imp_dict = {}
    try:
        bst = model.get_booster()
    except:
        bst = model
    feature_names = bst.feature_names
    for impt in imp_type:
        imp_dict[impt] = []
        scores = bst.get_score(importance_type=impt)
        for feature in feature_names:
            if feature in scores.keys():
                imp_dict[impt].append(scores[feature])
            else:
                imp_dict[impt].append(np.nan)
    imp_df = pd.DataFrame(index=bst.feature_names, data=imp_dict)
    return imp_df


def get_df_rankavg(df):
    idx = df.index
    cols = df.columns.tolist()
    rankavg_dict = {}
    for col in cols:
        rankavg_dict[col]=df[col].rank(pct=True).tolist()
    rankavg_df = pd.DataFrame(index=idx, columns=cols, data=rankavg_dict)
    rankavg_df['rankavg'] = rankavg_df.mean(axis=1)
    return rankavg_df.sort_values(by='rankavg', ascending=False)


def get_list_gmean(lists):
    out = np.zeros((len(lists[0]), len(lists)))
    for i in range(0, len(lists)):
        out[:,i] = lists[i]
    gmean_out = gmean(out, axis=1)
    return gmean_out



def generate_nwise_combination(items, n=2):
    return list(itertools.combinations(items, n))


def pairwise_feature_generation(df, feature_list, operator='addition', verbose=True):
    feats_pair = generate_nwise_combination(feature_list, 2)
    result_df = pd.DataFrame()
    for pair in feats_pair:
        if verbose:
            print('generating %s of %s and %s' % (operator, pair[0], pair[1]))
        if operator == 'addition':
            feat_name = pair[0] + '_add_' + pair[1]
            result_df[feat_name] = df[pair[0]] + df[pair[1]]
        elif operator == 'multiplication':
            feat_name = pair[0] + '_mulp_' + pair[1]
            result_df[feat_name] = df[pair[0]] * df[pair[1]]
        elif operator == 'division':
            feat_name = pair[0] + '_div_' + pair[1]
            result_df[feat_name] = df[pair[0]] / df[pair[1]]
    return result_df


def try_divide(x, y, val=0.0):
    """
    try to perform division between two number, and return a default value if division by zero is detected
    :param x: the number to be used as dividend
    :param y: the number to be used as divisor
    :param val: the default output value
    :return: the output value, the default value of val will be returned if division by zero is detected
    """
    if y != 0.0:
        val = float(x) / y
    return val


def series_reverse_cumsum(a):
    return a.fillna(0).values[::-1].cumsum()[::-1]


def get_array_sharpe(values):
    return values.mean()/values.std()


#### NumerDash specific functions ###

def calculate_rounddailysharpe_dashboard(df, lastround, earliest_round, score='corr'):
    if score=='corr':
        target = 'corr_sharpe'
    elif score == 'corr_pct':
        target = 'corr_pct_sharpe'
    elif score=='mmc':
        target = 'mmc_sharpe'
    elif score=='mmc_pct':
        target = 'mmc_pct_sharpe'
    elif score=='corrmmc':
        target = 'corrmmc_sharpe'
    elif score=='corr2mmc':
        target = 'corr2mmc_sharpe'
    elif score=='cmavg_pct':
        target = 'cmavgpct_sharpe'
    elif score=='c2mavg_pct':
        target = 'c2mavcpct_sharpe'

    mean_feat = 'avg_sharpe'
    sos_feat = 'sos'
    df = df[(df['roundNumber'] >= earliest_round) & (df['roundNumber'] <= lastround)]
    res = df.groupby(['model', 'roundNumber', 'group'])[score].apply(
        lambda x: get_array_sharpe(x)).reset_index(drop=False)
    res = res.rename(columns={score: target}).sort_values('roundNumber', ascending=False)
    res = res.pivot(index=['model', 'group'], columns='roundNumber', values=target)
    res.columns.name = ''
    cols = [i for i in res.columns[::-1]]
    res = res[cols]
    res[mean_feat] = res[cols].mean(axis=1)
    res[sos_feat] = res[cols].apply(lambda x: get_array_sharpe(x), axis=1)
    res = res.drop_duplicates(keep='first').sort_values(by=sos_feat, ascending=False)
    res.reset_index(drop=False, inplace=True)
    return res[['model', 'group', sos_feat, mean_feat]+cols]



def groupby_agg_execution(agg_recipies, df, verbose=True):
    result_dfs = dict()
    for groupby_cols, features, aggs in agg_recipies:
        group_object = df.groupby(groupby_cols)
        groupby_key = '_'.join(groupby_cols)
        if groupby_key not in list(result_dfs.keys()):
            result_dfs[groupby_key] = pd.DataFrame()
        for feature in features:
            rename_col = feature
            for agg in aggs:
                if isinstance(agg, dict):
                    agg_name = list(agg.keys())[0]
                    agg_func = agg[agg_name]
                else:
                    agg_name = agg
                    agg_func = agg
                if agg_name=='count':
                    groupby_aggregate_name = '{}_{}'.format(groupby_key, agg_name)
                else:
                    groupby_aggregate_name = '{}_{}_{}'.format(groupby_key, feature, agg_name)
                verbose and print(f'generating statistic {groupby_aggregate_name}')
                groupby_res_df = group_object[feature].agg(agg_func).reset_index(drop=False)
                groupby_res_df = groupby_res_df.rename(columns={rename_col: groupby_aggregate_name})
                if len(result_dfs[groupby_key]) == 0:
                    result_dfs[groupby_key] = groupby_res_df
                else:
                    result_dfs[groupby_key][groupby_aggregate_name] = groupby_res_df[groupby_aggregate_name]
    return result_dfs


def get_latest_round_id():
    try:
        all_competitions = numerapi_utils.get_competitions()
        latest_comp_id = all_competitions[0]['number']
    except:
        print('calling api unsuccessulf, using downloaded data to get the latest round')
        local_data = load_data(project_config.DASHBOARD_MODEL_RESULT_FILE)
        latest_comp_id = local_data['roundNumber'].max()
    return int(latest_comp_id)
#     except:

latest_round = get_latest_round_id()




def update_numerati_data(url=project_config.NUMERATI_URL, save_path=project_config.FEATURE_PATH):
    content = requests.get(url).content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))
    save_file = os.path.join(save_path, 'numerati_data.pkl')
    pickle_data(save_file, data)
    return data




def get_model_group(model_name):
    cat_name = 'other'
    if model_name in project_config.MODEL_NAMES+project_config.NEW_MODEL_NAMES:
        cat_name = 'yx'
    elif model_name in project_config.TOP_LB:
        cat_name = 'top_corr'
    elif model_name in project_config.IAAI_MODELS:
        cat_name = 'iaai'
    elif model_name in project_config.ARBITRAGE_MODELS:
        cat_name = 'arbitrage'
    elif model_name in project_config.MCV_MODELS:
        cat_name = 'mcv'
    # elif model_name in project_config.MM_MODELS:
    #     cat_name = 'mm'
    elif model_name in project_config.BENCHMARK_MODELS:
        cat_name = 'benchmark'
    elif model_name in project_config.TP3M:
        cat_name = 'top_3m'
    elif model_name in project_config.TP1Y:
        cat_name = 'top_1y'
    return cat_name


def get_dashboard_data_status():
    dashboard_data_tstr = 'NA'
    nmtd_tstr = 'NA'
    try:
        dashboard_data_t = datetime.datetime.utcfromtimestamp(os.path.getctime(project_config.DASHBOARD_MODEL_RESULT_FILE))
        dashboard_data_tstr = dashboard_data_t.strftime(project_config.DATETIME_FORMAT2)
    except Exception as e:
        print(e)
        pass
    try:
        nmtd_t = datetime.datetime.utcfromtimestamp(os.path.getctime(project_config.NUMERATI_FILE))
        nmtd_tstr = nmtd_t.strftime(project_config.DATETIME_FORMAT2)
    except Exception as e:
        print(e)
        pass
    return dashboard_data_tstr, nmtd_tstr














