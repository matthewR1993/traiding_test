import gc
import argparse

import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib

from features import *


parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str)
parser.add_argument('--train_data_path', type=str)
parser.add_argument('--predict_data_path', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()


def load_data(path):
    with h5py.File(path) as f:
        ob_gr = f['OB']
        ask = np.array(ob_gr['Ask'])
        askv = np.array(ob_gr['AskV'])
        bid = np.array(ob_gr['Bid'])
        bidv = np.array(ob_gr['BidV'])
        ob_ts = np.array(ob_gr['TS'])

    # TODO: delete that
    n = 100000
    ask = ask[:n]
    askv = askv[:n]
    bid = bid[:n]
    bidv = bidv[:n]
    ob_ts = ob_ts[:n]

    return ask, askv, bid, bidv, ob_ts


def generate_features(ask, askv, bid, bidv, ob_ts):
    mid_price = (np.max(bid, axis=1) + np.min(ask, axis=1)) * 0.5
    time_sec = (ob_ts * 0.001).astype(int)

    idx_list = []

    cutoff = 300
    horizont = 30

    for n in range(0, len(time_sec) - cutoff):
        idx_start = n
        idx_end = np.max(np.argwhere(time_sec[n:n + cutoff] - time_sec[n] <= horizont)) + n
        idx_list.append((idx_start, idx_end))

    res_ = []

    for n in range(len(idx_list)):
        idx = idx_list[n]
        x = time_sec[idx[1]] - time_sec[idx[0]]
        res_.append(x)

    ret_arr = []
    for n in range(len(idx_list)):
        idx = idx_list[n]
        ret_ = (mid_price[idx[1]] / mid_price[idx[0]]) - 1
        ret_arr.append(ret_)

    ret_arr = ret_arr + [0] * (mid_price.shape[0] - len(ret_arr))
    ret_arr = np.array(ret_arr)

    tail_size = 2000

    feat_list = []

    lags = [50, 75, 100, 250, 500]

    for n in range(len(mid_price)):
        idx_label = n
        feat_dict = {}
        for m in lags:
            if n >= tail_size:
                feat_dict[f'return_lagged_{m}'] = ret_arr[idx_label - m]
            else:
                feat_dict[f'return_lagged_{m}'] = np.nan

        feat_list.append(feat_dict)

    feat_df = pd.DataFrame(feat_list)

    del feat_list
    gc.collect()

    feat_dict = {}

    for k in feature_mapping.keys():
        for window in [30, 100]:
            feat_dict[f'{k}_window_{window}'] = np.nan_to_num(feature_mapping[k](mid_price, window))

    for window in [1, 5, 10, 25, 50, 100]:
        feat_dict[f'bid_ask_volume_cumm_diff_window_{window}'] = bid_ask_volume_cumm_diff(askv, bidv, window)
        feat_dict[f'bid_ask_max_diff_window_{window}'] = bid_ask_max_diff(askv, bidv, window)

    feat_df = pd.concat([feat_df, pd.DataFrame(feat_dict)], axis=1)
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feat_df, ret_arr


def prepare_train_test(feat_df, ret_arr):
    # shuffle
    feat_df['label'] = ret_arr
    feat_df = feat_df.sample(frac=1)

    # Removing correlated features.
    X_train_df = feat_df.drop(columns=['label'])
    _, feature_pairs = handling_correlation(X_train_df, threshold=0.8)
    features_to_remove = remove_corr_from_pairs(feature_pairs)
    print('features_to_remove:', features_to_remove)

    selected_features = list(set(list(feat_df.drop(columns=['label']).columns)) - set(features_to_remove))
    selected_features.sort()
    print('selected_features:', len(selected_features), selected_features)

    feat_df = feat_df[selected_features + ['label']]

    del X_train_df
    gc.collect()

    X = feat_df.drop(columns=['label']).values
    feat_cols = feat_df.drop(columns=['label']).columns
    y = feat_df['label'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    gc.collect()

    X_train = pd.DataFrame(X_train, columns=feat_cols)
    X_val = pd.DataFrame(X_val, columns=feat_cols)

    gc.collect()

    # Removing outliers in train.
    return_treshold = 0.02

    X_train['label'] = y_train
    X_train = X_train[X_train['label'].abs() < return_treshold]
    y_train = X_train['label'].values
    X_train = X_train.drop(columns=['label'])

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val):
    hyper_params = {
        "learning_rate": 0.9,
        "max_depth": 10,
        "num_iterations": 4000,
        "metric": 'mse'
    }

    model = lgb.LGBMRegressor(**hyper_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=lambda y_true, y_pred: [custom_r2(y_true, y_pred)],
        eval_names=['train', 'val'],
        early_stopping_rounds=300,
        verbose=-1
    )

    y_pred = model.predict(X_val)
    score_list = []
    coef_arr = np.linspace(0.5, 5, 100)

    for coef in coef_arr:
        score = r2_score(y_val * coef, y_pred)
        score_list.append(score)

    opt_coef = coef_arr[np.argmax(score_list)]
    print('opt_coef:', opt_coef)
    print('opt_score:', opt_score)

    return model, opt_coef


if args.run_mode == 'train':
    ask, askv, bid, bidv, ob_ts = load_data(path)
    feat_df, ret_arr = generate_features(ask, askv, bid, bidv, ob_ts)
    X_train, y_train, X_val, y_val = prepare_train_test(feat_df, ret_arr)
    model, opt_coef  = train_model(X_train, y_train, X_val, y_val)

    save_obj = {
        'model': model,
        'scale_coef': opt_coef
    }
    joblib.dump(save_obj, 'model.data')

elif args.run_mode == 'predict':
    # TODO
    model_data = joblib.load('')

    ask, askv, bid, bidv, ob_ts = load_data(path)
    feat_df, ret_arr = generate_features(ask, askv, bid, bidv, ob_ts)

    X = feat_df.values
    returns = model_data['model'].predict(X) * model_data['scale_coef']

    with h5py.File("result.h5", "w") as f:
        grp = f.create_group("Return")
        grp.create_dataset("TS", data=ob_ts)
        grp.create_dataset("Res", data=returns)

else:
    raise ValueError
