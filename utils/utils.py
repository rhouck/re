import math

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pykalman import KalmanFilter
from sklearn import linear_model as lm
from IPython.display import Image 
from sklearn.externals.six import StringIO
from sklearn.externals.six import StringIO  
import pydot
from sklearn import tree
from matplotlib import pyplot as plt
import seaborn as sns

from settings import *

## general tools
#

def stack_and_align(dfs, cols=None):

    def format(df):
        try:
            df.index.levels
        except:
            df = df.stack()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df

    df = pd.concat([format(df) for df in dfs], axis=1).dropna(how='all')

    if cols:
        df.columns = cols
    
    return df

def get_row_percentile(s, ts=False):
    if not ts:
        s = s.unstack().rank(axis=1)
        s = s.div(s.max(axis=1), axis='rows')
        s = s.stack()
    else:
        s = s.rank()
        s = s.div(s.max())
    return s


## ts transformations
#


def xs_z_score_winsorize(df, threshold=3):
    m = df.mean(axis=1)
    std = df.std(axis=1)
    
    limmit = m + std * threshold
    limmit = pd.DataFrame(np.repeat(np.array([limmit.values]).T, df.shape[1], axis=1), 
                         index=df.index, 
                         columns=df.columns)
    df = pd.concat([df.stack(), limmit.stack()], axis=1).min(axis=1).unstack()
    
    limmit = m - std * threshold
    limmit = pd.DataFrame(np.repeat(np.array([limmit.values]).T, df.shape[1], axis=1), 
                         index=df.index, 
                         columns=df.columns)
    df = pd.concat([df.stack(), limmit.stack()], axis=1).max(axis=1).unstack()

    return df


def ts_score(df, panel=True):
    
    def ts(df, panel):
        hl = TS_HALFLIFE
        min_per = 12
        if panel:
             hl = hl * df.index.levels[1].shape[0]
             min_per = min_per * df.index.levels[1].shape[0]
        m = pd.ewma(df, halflife=hl, min_periods=min_per)
        std = pd.ewmstd(df, halflife=hl, min_periods=min_per)
        return (df - m) / std
    
    if panel:
        return ts(df, panel)
    else:
        cols = df.columns
        for c in cols:
            d = df[c].unstack()
            d = ts(d, panel)
            df.loc[:,c] = d.stack()
        return df

def get_panel_z_scores(df):
    m = df.stack().mean()
    std = df.stack().std()
    z = (df - m) / std
    return z

def z_score_to_value(z_scores, values):
    m = values.mean()
    std = values.std()
    return (z_scores * std) + m

def get_cum_return(df):  
    
    df = df.applymap(lambda x: x + 1.)
    
    #df = (df / df.shift())#.replace(np.nan, 0)
    
    # # drop outliers - greater than 'outlier_threshold' std moves
    # #
    # z = get_z_scores(df)
    # z_max = z.applymap(lambda x: abs(x)).max()
    # outliers = z_max[z_max > outlier_threshold].index
    # df.drop(outliers, axis=1, inplace=True) 
    
    df = df.cumprod()  
    df = df.applymap(lambda x: x - 1.)
    return df

def get_forward_return(df, periods):    
    df = ((df.shift(-periods) / df) - 1.).dropna(how='all')
    # # drop outliers - greater than 3 std moves
    # #
    # z = get_z_scores(df)
    # z_max = z.applymap(lambda x: abs(x)).max()
    # outliers = z_max[z_max > 3].index
    # df.drop(outliers, axis=1, inplace=True)    
    return df

def kalman_ma(df, transition_covariance=.01):
    
    df_new = pd.DataFrame()
    
    for c in df.columns:
        
        # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices = [1],
                          observation_matrices = [1],
                          initial_state_mean = df.ix[0,c],
                          initial_state_covariance = 1,
                          observation_covariance=1,
                          transition_covariance=transition_covariance)

        # Use the observed values of the price to get a rolling mean
        state_means, _ = kf.filter(df[c].values)
        df_new[c] = state_means[:,0]
    
    df_new.index = df.index
    
    return df_new

def capped_transformation(func, px, px_ca, px_us):
    """apply a transformation 'func' to each time series and suppress outliers"""
    px = func(px)
    px_ca = func(px_ca)
    px_us = func(px_us)
    
    px_us = px_us.map(lambda x: 3 if x > 3 else x)
    px_ca = px_ca.map(lambda x: 3 if x > 3 else x)
    px = px.applymap(lambda x: 3 if x > 3 else x)
    
    return px, px_ca, px_us


## general empirics
# 

def get_betas(df, s, per=1, fwd=False):
    
    if fwd:
        df = (df.shift(-per) / df - 1.).dropna(how='all')
        s = (s.shift(-per) / s - 1.).dropna(how='all')
    else:
        df = (df / df.shift(per) - 1.).dropna(how='all')
        s = (s / s.shift(per) - 1.).dropna(how='all')

    clf = lm.LinearRegression(fit_intercept=True)
    
    betas = []
    for c in df.columns:
        d = pd.DataFrame({'X': s.values, 'y': df[c].values}).dropna() 
        try:
            clf.fit(d[['X']], d['y'])
            betas.append({'model': c, 'beta':  clf.coef_[0]})
        except:
            pass
    return pd.DataFrame(betas).set_index('model')['beta']

def get_beta_neutral_returns(df, s):

    clf = lm.LinearRegression(fit_intercept=True)
    for c in df.columns:
        X = pd.DataFrame(s.values)
        y = df[c].values
        clf.fit(X, y)
        df[c] =  y - clf.predict(X)
    return df
    
def gen_quintile_ts(df, rank_col, display_col, agg='sum'):
    
    rank = get_row_percentile(df[rank_col])
    rank.name = 'rank'
    
    df = stack_and_align([df, rank])
    
    quint = {}
    for i in range(2, 12, 2):
        i = i/10.
        
        d = df[(df['rank']>(i-.2)) & (df['rank']<i)][display_col].unstack()
        
        if agg=='sum':
            d = d.sum(axis=1)
        elif agg=='mean':
            d = d.mean(axis=1)
        
        quint[i] = d

    return pd.DataFrame(quint)


def gen_quintile_flat(df, rank_col, display_col, agg='sum', ts=False):
    
    rank = get_row_percentile(df[rank_col], ts=ts)
    rank.name = 'rank'
    
    df = stack_and_align([df, rank])
    
    quint = {}
    for i in range(2, 12, 2):
        i = i/10.
        
        d = df[(df['rank']>(i-.2)) & (df['rank']<i)][display_col]
        
        if agg=='sum':
            d = d.sum()
        elif agg=='mean':
            d = d.mean()
        
        quint[i] = d

    return pd.Series(quint)    


def lead_lag_corr(ind, dep, rng=range(-52,52,4)):
    data = []
    for i in reversed(rng):
        df_aligned = stack_and_align([ind.shift(i), dep]).dropna()
        c = df_aligned.corr().values[0,1]
        data.append({'per': i, 'corr': c})
    corr = pd.DataFrame(data)
    corr.set_index('per', inplace=True)
    return corr


def simple_ols(X, y, fit_intercept=True): 
    if fit_intercept:
        X = sm.add_constant(X)
    model = sm.OLS(y,X)
    results = model.fit()
    try:
        f_test = results.f_test(np.identity(2))
    except:
        f_test = None
        
    return {'params': results.params,
            'tvalues': results.tvalues,
            #'t_test': results.t_test([0,0]),
            'f_test': f_test
           }


def tree_vis(clf):
    #fn = ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(10)])
    fn = 'tree'
    fn = 'data/trees/{0}.png'.format(fn)
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(fn) 
    return Image(filename=fn)


def avg_rank_accuracy(df_res):
    d = gen_quintile_ts(df_res, 'pred', 'tar', agg='mean')
    d = get_row_percentile(d.stack()).unstack()
    d = d.fillna(method='ffill').dropna()
    return kalman_ma(d)


def get_cum_perforance(df):
    return get_cum_return(df.applymap(lambda x: math.pow(x + 1., (1./RET_PER)) - 1.))


def get_sharpe_ratio(df, rfr=0.0):
    ind = df.index
    q = ind.shape[0] / ((ind[-1] - ind[0]).days / 365.)

    an_ret = df.mean() * q
    an_vol = df.std() * math.sqrt(q)
    sharpe = (an_ret - rfr) / an_vol
    
    return sharpe

def get_xs_corr(sig, tar):
    corr = (stack_and_align([get_row_percentile(sig), get_row_percentile(tar)], 
                            cols=('sig', 'tar'))
            .corr().loc['sig', 'tar'])
    return corr














