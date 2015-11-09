import datetime
import os
import time
import random
import string
import math

import pandas as pd
import Quandl
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


QUANDL_API_KEY = '11Uh5euqzE625yn6n5QG'
RET_PER = 24
FIG_WIDTH = 8
FIG_HEIGHT = 4
TARGET_SERIES = 'A'
TARGET_INDICATOR = 'hoods'


## load quandl meta data
#
def load_regions():
    df = pd.read_csv('data/metro_codes.csv', delimiter='|', dtype='str')
    df.columns = df.columns.str.lower()
    df = df.ix[:0] # only load region representing entire us
    return df

def load_states():
    df = pd.read_csv('data/state_codes.csv', delimiter='|', dtype='str')
    df.columns = df.columns.str.lower()
    return df

def load_counties():
    df = pd.read_csv('data/county_codes.csv', delimiter='|', dtype='str')
    main_cols = df.columns[0].split(',')
    df = df[df.ix[:,0].map(lambda x: True if len(x.split(','))==len(main_cols) else False)]
    df = pd.concat([df.ix[:,0].apply(lambda x: pd.Series(x.split(','))), df.ix[:,1]], axis=1)
    df.columns = ['county', 'state', 'region',  'code']
    df = df[df.state=='CA']
    return df

def load_cities():
    df = pd.read_csv('data/city_codes.csv', delimiter='|', dtype='str')
    main_cols = df.columns[0].split(',')
    df = df[df.ix[:,0].map(lambda x: True if len(x.split(','))==len(main_cols) else False)]
    df = pd.concat([df.ix[:,0].apply(lambda x: pd.Series(x.split(','))), df.ix[:,1]], axis=1)
    df.columns = ['city', 'state',  'region', 'county', 'code']
    df = df[df.state=='CA']
    return df

def load_hoods():
    df = pd.read_csv('data/hood_codes.csv', delimiter='|', dtype='str')
    main_cols = df.columns[0].split(',')
    df = df[df.ix[:,0].map(lambda x: True if len(x.split(','))==len(main_cols) else False)]
    df = pd.concat([df.ix[:,0].apply(lambda x: pd.Series(x.split(','))), df.ix[:,1]], axis=1)
    df.columns = ['hood', 'city', 'state', 'region', 'county', 'code']
    df = df[df.state=='CA']
    return df

def load_indicators():
    indicator = pd.read_csv('data/indicator_codes.csv')
    indicator.columns = reversed(list(indicator.columns))
    return indicator


## quandl data
#

def scrape_quandl(area, indicator):
    """collects housing data from quandl zill api if available and not already collected - CA data only"""
    
    # check file exists
    fn = 'data/api_data/{0}_{1}.csv'.format(area.lower(), indicator.lower())
    if os.path.isfile(fn):
        print 'this data set is already collected'
        return
    
    # validate params
    areas = ('regions', 'states', 'counties', 'cities', 'hoods')
    if area not in areas:
        raise ValueError('area must be one of the following values: {0}'.format(areas))
    
    indicators = load_indicators()
    if indicator.upper() not in indicators.ix[:,1].values:
        raise ValueError('invalid inidicator value: {0}'.format(indicator.upper()))
    
    print 'lets scrape {0} / {1}'.format(area, indicator)
    
    if area == 'regions':
        codes = load_regions()
        area_api_category = 'M'
    elif area == 'states':
        codes = load_states()
        area_api_category = 'S'
    elif area == 'counties':
        codes = load_counties()
        area_api_category = 'CO'
    elif area == 'cities':
        codes = load_cities()
        area_api_category = 'C'
    else:
        codes = load_hoods()
        area_api_category = 'N'
        
    df_master = pd.DataFrame()
    for i in codes.code.values:    
        try:
            q =  "ZILL/{0}{1}_{2}".format(area_api_category, i, indicator.upper())
            df = Quandl.get(q, authtoken=QUANDL_API_KEY)
            df.columns = [i,]
            df_master = pd.concat([df_master, df], axis=1)
            time.sleep(.06)
        except Exception as err:
            if str(err) == 'Error Downloading! HTTP Error 429: Too Many Requests':
                print err
                return
            print('no data for code:\t{0}'.format(i))
            
    
    if df_master.shape[0]:
        df_master.to_csv(fn)
    
    print 'done'

def load_quandl_data(area, indicator):
    fn = 'data/api_data/{0}_{1}.csv'.format(area.lower(), indicator.lower())
    if not os.path.isfile(fn):
        print 'this data set has not been collected'
        return
    
    df = pd.read_csv(fn, parse_dates='Date', index_col='Date')
    df = df.resample('d', fill_method='bfill').resample('m')
    
    return df


def load_returns(indicator=TARGET_INDICATOR, series=TARGET_SERIES):
    """actual forward looking returns (using RET_PER)"""

    df = load_quandl_data(indicator, series)
    df = (df
          .fillna(method='bfill', limit=3)
          .fillna(method='ffill', limit=3)
          .dropna(axis=1))
    
    df = df.pct_change().dropna()

    df = get_cum_return(df) + 1.
    
    df = df.shift(-RET_PER) / df - 1.
    df.dropna(how='all', inplace=True)
        
    return df

def load_target(neutral=False):
    """target returns (beta neutral optional)"""

    tar = load_returns()

    if neutral:
        raise Exception('add logic to create market neutral betas')

    return tar
    # df = load_quandl_data(TARGET_INDICATOR,TARGET_SERIES)
    # df = (df
    #       .fillna(method='bfill', limit=3)
    #       .fillna(method='ffill', limit=3)
    #       .dropna(axis=1))
    
    # df = df.pct_change().dropna()

    # if neutral:
    #     mkt = load_quandl_data('states',TARGET_SERIES).ix[:,0]
        
    #     df = get_beta_neutral_returns(df, mkt.pct_change().dropna())
    
    # df = get_cum_return(df) + 1.
    
    # df = df.shift(-RET_PER) / df - 1.
    # df.dropna(how='all', inplace=True)
    
    # #df = get_z_scores(df)
    
    # return df



def load_series(series):
    
    px_us = load_quandl_data('regions', series).ix[:,0]
    px_ca = load_quandl_data('states', series).ix[:,0]
    px = load_quandl_data(TARGET_INDICATOR, series)

    px = (px.fillna(method='bfill', limit=3)
          .fillna(method='ffill', limit=3)
          .dropna(axis=1))
    
    return px, px_ca, px_us

 
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

def ts_score(df, panel=True):
    
    def ts(df):
        return  (df - df.mean()) / df.std()
    
    if panel:
        return ts(df)
    else:
        cols = df.columns
        for c in cols:
            d = df[c].unstack()
            d = ts(d)
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

def momentum(df, per=12):
    df = df.pct_change()
    #return pd.rolling_mean(df, window=per).dropna()
    return pd.ewma(df, halflife=per, min_periods=per).dropna(how='all')


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
    fn = ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(10)])
    fn = 'data/trees/{0}.png'.format(fn)
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(fn) 
    return Image(filename=fn)


def explore_series(px, px_ca, px_us, tar):
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(FIG_WIDTH*2, FIG_HEIGHT*3))
    px_us.plot(ax=axes[0,0], title='ca and us sig')
    px_ca.plot(ax=axes[0,0], title='hoods sig')
    px.plot(ax=axes[0,1], legend=False, alpha=.3)

    (lead_lag_corr(px, tar, rng=range(-52,52,4))
     .plot(kind='bar', title='lead lag corr', ax=axes[1,0]))#.axvline(0, linestyle='--', color='r'))

    df = stack_and_align([px, tar], cols=('sig','tar')).dropna()
    df = ts_score(df)
    sns.distplot(df['sig'], ax=axes[2,0]).set_title('sig dist')
    sns.regplot(df['sig'], df['tar'], ax=axes[2,1]).set_title('sig vs tar')

    clf = lm.LinearRegression()
    clf.fit(df[['sig']], df['tar'])
    print('int: {0}\tcoef: {1}'.format(clf.intercept_, clf.coef_[0]))


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


def build_model(clf, df):
    
    clf.fit(df[[c for c in df.columns if c != 'tar']], df['tar'])
    s = pd.Series(ts_score(clf.predict(df[[c for c in df.columns if c != 'tar']])), index=df.index)
    sns.jointplot(s, df['tar'], kind='reg')
    score = clf.score(df[[c for c in df.columns if c != 'tar']], df['tar'])

    ret = load_returns().stack().ix[df.index]
    # dif between avg ret and avg tar


    df_res = stack_and_align([df['tar'], s, ret], cols=('tar', 'pred', 'ret'))
    df_res['err'] = df_res['tar'] - df_res['pred']
    df_res['err2'] = df_res['err'].map(lambda x: x**2)
    avg_ret = df_res['tar'].unstack().mean(axis=1)
    df_res['avg_ret'] = pd.DataFrame({c: avg_ret for c in df_res.index.levels[1]}).stack()

    fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(FIG_WIDTH*2, FIG_HEIGHT*5))
    gen_quintile_flat(df_res, 'tar', 'pred', agg='mean', ts=False).plot(kind='bar', ax=axes[0,0], title='tar vs pred (xs)')
    gen_quintile_flat(df_res, 'tar', 'pred', agg='mean', ts=True).plot(kind='bar', ax=axes[1,0], title='tar vs pred (ts)')
    gen_quintile_flat(df_res, 'avg_ret', 'pred', agg='mean', ts=True).plot(kind='bar', ax=axes[2,0], title='mkt ret vs pred (ts)')
    gen_quintile_flat(df_res, 'tar', 'err2', agg='sum', ts=False).plot(kind='bar', ax=axes[0,1], title='tar vs err2 (xs)')
    gen_quintile_flat(df_res, 'tar', 'err2', agg='sum', ts=True).plot(kind='bar', ax=axes[1,1], title='tar vs err2 (ts)')
    gen_quintile_flat(df_res, 'avg_ret', 'err2', agg='sum', ts=True).plot(kind='bar', ax=axes[2,1], title='mkt ret vs err2 (ts)')
    gen_quintile_ts(df_res, 'pred', 'pred', agg='mean').plot(ax=axes[3,0], title='avg pred over time')
    gen_quintile_ts(df_res, 'tar', 'tar', agg='mean').plot(ax=axes[3,1], title='avg tar over time')
    
    avg_rank_accuracy(df_res).plot(ax=axes[4,0], title='avg pred rank accuracy')

    q = gen_quintile_ts(df_res, 'pred', 'ret', agg='mean')
    q['mkt'] = load_returns('states').ix[:,0]
    print('sharpe ratios:')
    print get_sharpe_ratio(q)
    get_cum_perforance(q).plot(ax=axes[4,1], title='continuously invested performance')
    
    return clf, df_res, score

















