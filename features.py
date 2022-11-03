import numpy as np
from sklearn.metrics import r2_score


def rolling_window(a: np.ndarray, window: int) -> np.ndarray:
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    # len(x) == len(output)
    y = np.empty(window - 1)
    y[:] = np.nan
    return np.append(y, np.sum(rolling_window(x, window), axis=1))


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    # len(x) == len(output)
    y = np.empty(window - 1)
    y[:] = np.nan
    return np.append(y, np.mean(rolling_window(x, window), axis=1))


def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    # len(x) == len(output)
    y = np.empty(window - 1)
    y[:] = np.nan
    return np.append(y, np.min(rolling_window(x, window), axis=1))


def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    # len(x) == len(output)
    y = np.empty(window - 1)
    y[:] = np.nan
    return np.append(y, np.max(rolling_window(x, window), axis=1))


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    # len(x) == len(output)
    y = np.empty(window - 1)
    y[:] = np.nan
    return np.append(y, np.std(rolling_window(x, window), axis=1, ddof=1))


def pct_change(x: np.ndarray) -> np.ndarray:
    # len(x) == len(output)
    return np.append(np.nan, np.diff(x) / x[:-1])


def pct_change_window(x: np.ndarray, window: int) -> np.ndarray:
    y = x / np.roll(x, window) - 1
    y[:window] = np.nan
    return y


def diff(x: np.ndarray, window: int) -> np.ndarray:
    y = x - np.roll(x, window)
    y[:window] = np.nan
    return y


def get_macd(price: np.ndarray, window: int) -> np.ndarray:
    # macd:  (Mov average Price of 2 ticks / Mov average Price of window ticks) - 1
    # f'macd_{window}'
    return (rolling_mean(price, 2) / rolling_mean(price, window)) - 1


def get_support(price: np.ndarray, window: int) -> np.ndarray:
    # support: (current_price / Max price of this window) - 1
    # f'spt_{window}'
    return (price / rolling_max(price, window)) - 1


def get_resistance(price: np.ndarray, window: int) -> np.ndarray:
    #  resistance: (current_price / Min price of this window) - 1
    # f'rst_{window}'
    return (price / rolling_min(price, window)) - 1


def get_dist_from_min(price: np.ndarray, window: int) -> np.ndarray:
    sqrt_window = int(np.sqrt(window))
    volatility_sqrt_window = rolling_std(price, sqrt_window)
    expected_price_change_from_volatility = volatility_sqrt_window * sqrt_window
    res = (price - rolling_min(price, window)) / expected_price_change_from_volatility
    res[np.isinf(res)] = np.nan
    return np.log(res + 1)


def get_dist_from_max(price: np.ndarray, window: int) -> np.ndarray:
    sqrt_window = int(np.sqrt(window))
    volatility_sqrt_window = rolling_std(price, sqrt_window)
    expected_price_change_from_volatility = volatility_sqrt_window * sqrt_window
    res = (rolling_max(price, window) - price) / expected_price_change_from_volatility
    res[np.isinf(res)] = np.nan
    return np.log(res + 1)


def get_range(price: np.ndarray, window: int) -> np.ndarray:
    # range: ((Max Price window - Min Price window) / Min Price window)
    # f'range_{window}'
    return (rolling_max(price, window) / rolling_min(price, window)) / rolling_min(price, window)


def get_velocity(price: np.ndarray, window: int) -> np.ndarray:
    # displacement = Percentage Change of Price w.r.t. its Price window ticks ago.
    # velocity: displacement / window
    # f'velocity_{window}'
    return np.abs(pct_change_window(price, window)) / window


def get_acceleration(price: np.ndarray, window: int) -> np.ndarray:
    # acceleration: Percentage change of velocity w.r.t. its previous value.
    # f'accl_{window}'
    res = pct_change(get_velocity(price, window))
    res[np.isinf(res)] = np.nan
    return np.log(res + 2)


def get_high_low_pct_change_window(price: np.ndarray, window: int) -> np.ndarray:
    # (highest_high_price_window ÷ lowest_low_price_window) -1  (Note that this will always be a +ve value)
    # If the sign of open_close_pct_change_window_size is -ve then multiple this value by -1, otherwise leave it as is
    # This essentially gives us the range of price in percentage for this window.
    # The sign helps with direction. E.g. The qualification with -ve sign if the close was lower
    # than open should help identify 3 - i.e. super support.
    # A positive sign may help identify 4 i.e. super resistance.
    high = rolling_max(price, window)
    low = rolling_min(price, window)
    x = pct_change_window(price, window)
    x[x < 0] = -1
    x[x >= 0] = 1
    return (high / low - 1) * x


def get_price_displacement_window(price: np.ndarray, window: int) -> np.ndarray:
    # (close_price - open_price) ÷ (From n = 1 .. w Σ Abs( Price(n) - Price(n-1)) - where n = 0 based çindex of the price
    # This helps in determining displacement of the price w.r.t. (Open - Close) within the range (High - Low):
    # It will oscillate b/w:  -1…………….. 0 …………….. 1. Values closer to zero are going to be away from 3 or 4
    disp = rolling_sum(np.abs(diff(price, 1)), window)
    return diff(price, window) / disp


def pe_pct_change_HH_window(price: np.ndarray, window: int) -> np.ndarray:
    # Percentage change of the Highest High Price in a window w.r.t. its previous value.
    return pct_change(rolling_max(price, window))


def pe_pct_change_LL_window(price: np.ndarray, window: int) -> np.ndarray:
    # Percentage change of the Lowest Low Price in a window w.r.t. its previous value.
    return pct_change(rolling_min(price, window))


def get_open_close_pct_change_window(price: np.ndarray, window: int) -> np.ndarray:
    # (current_price ÷ price_window_ticks_ago) - 1
    return pct_change_window(price, window)


def get_roc_high_low_pct_change_window(price: np.ndarray, window: int) -> np.ndarray:
    # Percentage change of high_low_pct_change_window w.r.t. it’s previous value
    # This helps with identifying if a range is contracting or expanding.
    return pct_change(get_high_low_pct_change_window(price, window))


def get_std_price_percentage_change_window(price: np.ndarray, window: int) -> np.ndarray:
    # Standard deviation of price_pct_change for a given window
    # This provides us with rough estimate of the volatility for a window
    return rolling_std(pct_change(price), window)


def get_pct_change_mov_avg_window(price: np.ndarray, window: int) -> np.ndarray:
    # Percentage change of the moving average w.r.t. to its own previous value (Could potentially help with identifying trend)
    # (current_mov_avg_price_window ÷ previous_mov_average_price_window) - 1
    return pct_change(rolling_mean(price, window))


def bid_ask_volume_cumm_diff(askv, bidv, window):
    if window == 1:
        ask_sum = np.sum(askv, axis=1)
        bid_sum = np.sum(bidv, axis=1)
        return bid_sum - ask_sum

    if window > 1:
        bid_sum = rolling_sum(np.sum(askv, axis=1), window)
        ask_sum = rolling_sum(np.sum(bidv, axis=1), window)
        return bid_sum - ask_sum


def bid_ask_max_diff(askv, bidv, window):
    if window == 1:
        max_ask = np.max(askv, axis=1) - np.min(askv, axis=1)
        max_bid = np.max(bidv, axis=1) - np.min(bidv, axis=1)
        return max_ask - max_bid

    if window > 1:
        max_ask = rolling_max(np.max(askv, axis=1), window)
        max_bid = rolling_max(np.max(bidv, axis=1), window)
        return max_ask - max_bid


feature_mapping = {
    'pct_change': pct_change_window,
    'macd': get_macd,
    'support': get_support,
    'resistance': get_resistance,
    'dist_from_min': get_dist_from_min,
    'dist_from_max': get_dist_from_max,
    'range': get_range,
    'velocity': get_velocity,
    'acceleration': get_acceleration,
    'high_low_pct_change': get_high_low_pct_change_window,
    'price_displacement': get_price_displacement_window,
    'pe_pct_change_HH': pe_pct_change_HH_window,
    'pe_pct_change_LL': pe_pct_change_LL_window,
    'open_close_pct_change': get_open_close_pct_change_window,
    'roc_high_low_pct_change': get_roc_high_low_pct_change_window,
    'std_price_percentage_change': get_std_price_percentage_change_window,
    'pct_change_mov_avg': get_pct_change_mov_avg_window
}


def handling_correlation(features_df, threshold):
    feature_pairs = []
    corr_features = set()
    corr_matrix = features_df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                pair_name = f'{corr_matrix.columns[i]}____{corr_matrix.columns[j]}'
                # print(pair_name, abs(corr_matrix.iloc[i, j]))
                feature_pairs.append((pair_name, abs(corr_matrix.iloc[i, j])))
                colname = corr_matrix.columns[i]
                corr_features.add(colname)

    return list(corr_features), feature_pairs


def remove_corr_from_pairs(feature_pairs):
    feat_to_keep = {}
    all_features = []

    for fp in feature_pairs:
        x = fp[0].split('____')
        feat_1, feat_2 = x[0], x[1]
        if feat_1 not in feat_to_keep and feat_2 not in feat_to_keep:
            feat_to_keep[feat_1] = 1
            all_features.append(feat_1)
            all_features.append(feat_2)

    all_features = list(set(all_features))
    features_to_keep = list(feat_to_keep.keys())
    feat_to_remove = list(set(all_features) - set(features_to_keep))
    return feat_to_remove


def custom_r2(y_true, y_pred):
    return 'R2', r2_score(y_true, y_pred), True
