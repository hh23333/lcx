import pandas as pd


def click_counting(x, bin_column):
    clicks = pd.Series(
        x[x['click'] > 0][bin_column].value_counts(), name='clicks')
    no_clicks = pd.Series(
        x[x['click'] < 1][bin_column].value_counts(), name='no_clicks')

    counts = pd.DataFrame([clicks, no_clicks]).T.fillna('0')
    counts['total'] = counts['clicks'].astype(
        'int64') + counts['no_clicks'].astype('int64')

    return counts


def bin_counting(counts):
    counts['N+'] = counts['clicks'].astype('int64').divide(
        counts['total'].astype('int64'))
    counts['N-'] = counts['no_clicks'].astype('int64').divide(
        counts['total'].astype('int64'))
    counts['log_N+'] = counts['N+'].divide(counts['N-'])

    #    If we wanted to only return bin-counting properties, we would filter here
    bin_counts = counts.filter(items=['N+', 'N-', 'log_N+'])
    return counts, bin_counts


def cate_dist_reg(d):
    if d == 0:
        d = 0
    elif d <= 1:
        d = 1
    elif d <= 2:
        d = 2
    elif d <= 4:
        d = 3
    elif d <= 6:
        d = 4
    else:
        d = 5
    return d
