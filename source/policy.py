import pandas as pd
import numpy as np
import source.utilities as utilities


def hybrid_policy(data, config):

    oob_threshold = config['oob_threshold']
    histogram_threshold = config['histogram_threshold']*60
    cv_threshold = config['cv_threshold']
    pctl_lower = config['pctl_lower']
    pctl_upper = config['pctl_upper']

    iit = utilities.compute_inter_invocation_times(data)
    hist = utilities.compute_histogram(iit)

    oob_ratio = utilities.compute_oob_ratio(hist, histogram_threshold)

    hist = hist[:,:histogram_threshold]

    percentile_lower, percentile_upper = utilities.compute_percentiles(hist, pctl_lower, pctl_upper)

    pre_warm = percentile_lower
    keep_alive = percentile_upper - percentile_lower

    df = pd.DataFrame(hist)

    df['PreWarm'] = pre_warm
    df['KeepAlive'] = keep_alive + 1

    df['Policy'] = 'Arima'
    df['OOBRatio'] = oob_ratio
    df.loc[df['OOBRatio'] <= oob_threshold, 'Policy'] = "Histogram"

    cv = utilities.compute_cv(hist)
    df['CV'] = cv
    df.loc[(df['CV'] < cv_threshold) & (df['Policy'] != 'Arima'), 'Policy'] = 'Fixed'

    df = df[['PreWarm', 'KeepAlive', 'Policy', 'OOBRatio', 'CV']]

    return df
