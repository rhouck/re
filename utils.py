import datetime
import os
import time

import pandas as pd
import Quandl
import numpy as np
import statsmodels.api as sm
from pykalman import KalmanFilter
from sklearn import linear_model as lm


QUANDL_API_KEY = '11Uh5euqzE625yn6n5QG'
RET_PER = 24
FIG_WIDTH = 8
FIG_HEIGHT = 4

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

    df = pd.concat([format(df) for df in dfs], axis=1).dropna()

    if cols:
        df.columns = cols
    
    return df

def get_row_percentile(df):
    df = df.unstack().rank(axis=1)
    df = df.div(df.max(axis=1), axis='rows')
    return df.stack()


## general empirics
# 

def get_betas(df, s, per=1):
    df = (df / df.shift(per)).dropna(how='all')
    s = (s / s.shift(per)).dropna(how='all')
    clf = lm.LinearRegression(fit_intercept=False)
    
    betas = []
    for c in df.columns:
        d = pd.DataFrame({'X': df[c], 'y': s.values}).dropna() 
        try:
            clf.fit(d[['X']], d['y'])
            betas.append({'model': c, 'beta':  clf.coef_[0]})
        except:
            pass
    return pd.DataFrame(betas).set_index('model')['beta']

def gen_quintile_data(df, rank_col, display_col, agg='sum'):
    
    rank = get_row_percentile(df[rank_col])
    rank.name = 'rank'
    
    df = stack_and_align([df, rank])
    
    quint = pd.DataFrame()
    for i in range(2, 12, 2):
        i = i/10.
        
        d = df[(df['rank']>(i-.2)) & (df['rank']<i)][display_col].unstack()
        
        if agg=='sum':
            d = d.sum(axis=1)
        elif agg=='mean':
            d = d.mean(axis=1)
        
        quint[i] = d

    return quint

def ts_score(df):
    return  (df - df.mean()) / df.std()

def get_z_scores(df):
    m = df.stack().mean()
    std = df.stack().std()
    z = (df - m) / std
    return z

def z_score_to_value(z_scores, values):
    m = values.mean()
    std = values.std()
    return (z_scores * std) + m

def get_cum_return(df, outlier_threshold=3):  
    df = df.applymap(lambda x: x + 1.)
    df = (df / df.shift())#.replace(np.nan, 0)
    
    # drop outliers - greater than 'outlier_threshold' std moves
    #
    z = get_z_scores(df)
    z_max = z.applymap(lambda x: abs(x)).max()
    outliers = z_max[z_max > outlier_threshold].index
    df.drop(outliers, axis=1, inplace=True) 
    
    df = df.cumprod()  
    df = df.applymap(lambda x: x - 1.)
    return df

def get_forward_return(df, periods):    
    df = ((df.shift(-periods) / df) - 1.).dropna(how='all')
    # drop outliers - greater than 3 std moves
    #
    z = get_z_scores(df)
    z_max = z.applymap(lambda x: abs(x)).max()
    outliers = z_max[z_max > 3].index
    df.drop(outliers, axis=1, inplace=True)    
    return df

def lead_lag_corr(df_levels, df_returns, rng=range(-52,150,4)):
    data = []
    for i in reversed(rng):
        rec_ret = (df_levels / df_levels.shift(i) - 1.).dropna(how='all')
        df_aligned = stack_and_align(rec_ret, df_returns, cols=['past', 'fwd'])
        c = df_aligned.corr().values[0,1]
        dif = (df_aligned['fwd'] - df_aligned['past']).mean()
        data.append({'per': i, 'corr': c, 'dif': dif})
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

def kalman_ma(df, transition_covariance=.01):
    
    df_new = pd.DataFrame()
    
    # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=transition_covariance)

    for c in df.columns:
        # Use the observed values of the price to get a rolling mean
        state_means, _ = kf.filter(df[c].values)
        df_new[c] = state_means[:,0]
    
    df_new.index = df.index
    
    return df_new
