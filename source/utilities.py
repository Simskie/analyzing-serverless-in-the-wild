import pandas as pd
import numpy as np
import pmdarima as pm
import numpy as np



def compute_simulation(data_invoc, keep_alive_window, prewarm_window):
    """
    Run a serverless simulation of function invocations given a policy with keep-alive and prewarm windows
    """

    unique_apps = data_invoc['HashApp'].unique()

    app_settings = pd.DataFrame({
        'HashApp': unique_apps,
        'pre_warm': prewarm_window,
        'keep_alive': keep_alive_window
    })
    df_melted = data_invoc.melt(id_vars='HashApp', var_name='Minute', value_name='Invocation')
    df_melted['Minute'] = df_melted['Minute'].astype(int)
    
    new_rows = pd.DataFrame({
        'HashApp': unique_apps,
        'Minute': 0,
        'Invocation': -1
    })

    df_melted = pd.concat([new_rows, df_melted], ignore_index=True)

    df_melted['NonZeroInvocation'] = df_melted['Invocation'] != 0

    df_invocations = df_melted[df_melted['NonZeroInvocation']].copy()
    df_invocations.sort_values(['HashApp', 'Minute'], inplace=True)
    
    df_invocations['TimeDiff'] = df_invocations.groupby('HashApp')['Minute'].diff()
    df_invocations = df_invocations[df_invocations['Invocation'] != -1]

    df_invocations = df_invocations.merge(app_settings, on='HashApp', how='left')

    df_invocations['pre_warm'] = df_invocations['pre_warm'].fillna(0)
    df_invocations['keep_alive'] = df_invocations['keep_alive'].fillna(0)

    df_invocations['FirstInvocation'] = df_invocations.groupby('HashApp')['Minute'].rank(method='first') == 1

    df_invocations['ColdStart'] = (
        df_invocations['FirstInvocation'] |
        (
            (df_invocations['TimeDiff'] > (df_invocations['pre_warm'] + df_invocations['keep_alive'])) |
            (
                (df_invocations['TimeDiff'] <= df_invocations['pre_warm']) &
                (df_invocations['TimeDiff'] > 1)
            )
        )
)
    
    df_invocations['KeepAlive'] = df_invocations.apply(
        lambda row: min(
            row['TimeDiff'] - 1, 
            row['keep_alive'] if pd.notnull(row['TimeDiff']) else 0
        ) if (row['TimeDiff'] > 1) and (row['TimeDiff'] <= (row['pre_warm'] + row['keep_alive'])) else 0,
        axis=1
    )
    
    app_metrics = df_invocations.groupby('HashApp').agg(
        TotalInvocations=('NonZeroInvocation', 'sum'),
        ColdStartCount=('ColdStart', 'sum'),
        TotalKeepAliveTime=('KeepAlive', 'sum')
    ).reset_index()

    app_metrics['ColdStartPercentage'] = (
        app_metrics['ColdStartCount'] / app_metrics['TotalInvocations'] * 100
    )

    app_metrics['TotalDuration'] = app_metrics['TotalInvocations']

    app_metrics['WastedMemoryRatio'] = app_metrics['TotalKeepAliveTime'] / (app_metrics['TotalKeepAliveTime'] + app_metrics['TotalDuration'])

    return app_metrics


def compute_cv(hist):
    mean_bin_count = np.nanmean(hist, axis=1)
    std_bin_count = np.nanstd(hist, axis=1)
    
    cv = np.zeros_like(mean_bin_count)
    
    valid = (mean_bin_count != 0) & (~np.isnan(mean_bin_count))
    
    cv = np.divide(std_bin_count, mean_bin_count, out=cv, where=valid)
    
    return cv


def compute_percentiles(hist, pctl_lower, pctl_upper):
    cdf = np.cumsum(hist, axis=1)

    denominator = cdf[:, -1][:, np.newaxis]

    denominator[denominator == 0] = 1

    normalized_cdf = cdf / denominator

    percentile_lower = np.array([np.searchsorted(row, pctl_lower / 100) for row in normalized_cdf])
    percentile_upper = np.array([np.searchsorted(row, pctl_upper / 100) for row in normalized_cdf])

    return percentile_lower, percentile_upper


def compute_inter_invocation_times(data):
    num_rows, num_cols = data.shape

    col_indices = np.arange(num_cols)

    non_zero_mask = data != 0

    positions = [col_indices[non_zero_mask[row]] for row in range(num_rows)]

    gap_lengths = []
    max_gaps = 0

    for idx, pos in enumerate(positions):
        if pos.size > 1:
            gaps = np.diff(pos)
            gap_lengths.append(gaps.tolist())
            max_gaps = max(max_gaps, len(gaps))
        else:
            gap_lengths.append([])
    
    padded_gap_lengths = np.array([
        gaps + [0]*(max_gaps - len(gaps)) for gaps in gap_lengths
    ])

    return padded_gap_lengths


def compute_histogram(data):

    non_zero_data = data[data != 0]
    
    if non_zero_data.size == 0:
        counts = np.zeros((data.shape[0], 0), dtype=int)
        return counts

    min_value = 1
    max_value = np.max(data)
    num_bins = int(max_value - min_value + 1)
    bins = np.linspace(min_value, max_value + 1, num_bins + 1)

    flat_data = data.flatten()
    mask = flat_data != 0

    bin_indices = np.digitize(flat_data[mask], bins) - 1

    row_indices = np.repeat(np.arange(data.shape[0]), data.shape[1])[mask]

    counts = np.zeros((data.shape[0], num_bins), dtype=int)

    np.add.at(counts, (row_indices, bin_indices), 1)

    return counts


def compute_oob_ratio(arr, idx_threshold):
    total_sum = np.sum(arr, axis=1)

    outside_indices = np.arange(arr.shape[1]) >= idx_threshold

    outside_values = arr[:, outside_indices]

    outside_sum = np.sum(outside_values, axis=1)

    ratio = np.zeros_like(total_sum, dtype=float)

    valid_mask = total_sum != 0

    ratio[valid_mask] = outside_sum[valid_mask] / total_sum[valid_mask]

    return ratio


def train_test_arima(train, test):
    total_cold_starts = 0
    total_function_duration = 0
    total_keep_alive_time = 0
    apps = train['HashApp']

    for app in apps:
        train_app = train[train['HashApp'] == app]
        train_app = train_app.drop(columns=['HashApp'])
        train_app = train_app.to_numpy().flatten()
        test_app = test[test['HashApp'] == app]
        test_app = test_app.drop(columns=['HashApp'])
        test_app = test_app.to_numpy().flatten()

        model = pm.auto_arima(train_app)

        n_periods = len(test_app)
        preds = model.predict(n_periods=n_periods)

        predicted_iits = np.array(preds)

        actual_iits = test_app
        t_actual = np.cumsum(actual_iits)

        t_predicted = np.cumsum(predicted_iits)

        app_cold_starts = 0
        app_function_duration = len(actual_iits) + 1
        app_keep_alive_time = 0

        for i in range(n_periods):
            pred_iit = predicted_iits[i]
            t_p = t_predicted[i]
            t_a = t_actual[i]

            pre_warm_start = t_p - pred_iit * 0.85
            pre_warm_end = t_p - pred_iit * 0.15
            keep_alive_start = pre_warm_end
            keep_alive_end = t_p + pred_iit * 0.15

            app_keep_alive_time += (keep_alive_end - pre_warm_start)

            if pre_warm_start <= t_a <= keep_alive_end:
                pass
            else:
                app_cold_starts += 1

        total_cold_starts += app_cold_starts
        total_function_duration += app_function_duration
        total_keep_alive_time += app_keep_alive_time


    return total_cold_starts, total_function_duration, total_keep_alive_time