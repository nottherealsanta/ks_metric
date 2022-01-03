import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer


def ks_table(
    y_true,
    y_pred,
    n_bins=10,
    ret_ks=False
):
    """
    Build and return Gains Table

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    n_bins : int, default=10
        Number of bins for the ks_table
    ret_ks : bool, default=False
        If True, returns ks value along with ks_table.

    Returns
    -------
    ks_table : DataFrame

    ks : float
        Returned only if `ret_ks` is True.
    """

    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    df = pd.DataFrame()
    df['score'] = y_pred
    df['bad'] = y_true
    df['good'] = 1 - y_true

    df['bucket'] = pd.qcut(df.score.rank(method='first'), n_bins)

    grouped = df.groupby('bucket', as_index=False)

    ks_table = pd.DataFrame()
    ks_table['min_score'] = grouped.min().score
    ks_table['max_score'] = grouped.max().score
    ks_table['n_bads'] = grouped.sum().bad
    ks_table['n_goods'] = grouped.sum().good
    ks_table['n_total'] = ks_table.n_bads + ks_table.n_goods

    ks_table['odds'] = (ks_table.n_goods /
                        ks_table.n_bads).apply('{0:.2f}'.format)
    ks_table['bad_rate'] = (
        ks_table.n_bads / ks_table.n_total).apply('{0:.2%}'.format)
    ks_table['good_rate'] = (
        ks_table.n_goods / ks_table.n_total).apply('{0:.2%}'.format)

    count_bads = df.bad.sum()
    count_goods = df.good.sum()
    ks_table['cs_bads'] = ((ks_table.n_bads / count_bads).cumsum()
                           * 100).apply('{0:.2f}'.format)
    ks_table['cs_goods'] = ((ks_table.n_goods / count_goods).cumsum()
                            * 100).apply('{0:.2f}'.format)
    ks_table['sep'] = np.abs(np.round(
        ((ks_table.n_bads / count_bads).cumsum() - (ks_table.n_goods / count_goods).cumsum()), 4) * 100)

    def flag(x): return '<--' if x == ks_table.sep.max() else ''
    ks_table['KS'] = ks_table.sep.apply(flag)

    ks = ks_table['sep'].max()

    if ret_ks:
        return ks_table, ks
    else:
        return ks_table


def ks_score(
    y_true,
    y_pred,
    n_bins=10,
):
    """
    returns KS value 

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    Returns
    -------

    ks : float

    """

    _, ks = ks_table(y_true, y_pred, ret_ks=True)

    return ks

ks_scorer = make_scorer(ks_score, greater_is_better=True)