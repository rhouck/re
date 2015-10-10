import datetime

import pandas as pd
import Quandl
import numpy as np

## load quandl meta data
#

def load_counties():
    df = pd.read_csv('data/county_codes.csv', delimiter='|', dtype='str')
    main_cols = df.columns[0].split(',')
    df = df[df.ix[:,0].map(lambda x: True if len(x.split(','))==len(main_cols) else False)]
    df = pd.concat([df.ix[:,0].apply(lambda x: pd.Series(x.split(','))), df.ix[:,1]], axis=1)
    df.columns = ['county', 'state', 'region',  'code']
    return df

def load_cities():
    cities = pd.read_csv('data/city_codes.csv', delimiter='|', dtype='str')
    main_cols = cities.columns[0].split(',')
    cities = cities[cities.ix[:,0].map(lambda x: True if len(x.split(','))==len(main_cols) else False)]
    cities = pd.concat([cities.ix[:,0].apply(lambda x: pd.Series(x.split(','))), cities.ix[:,1]], axis=1)
    cities.columns = ['city', 'state',  'region', 'county', 'code']
    return cities

def load_hoods():
    hoods = pd.read_csv('data/hood_codes.csv', delimiter='|', dtype='str')
    main_cols = hoods.columns[0].split(',')
    hoods = hoods[hoods.ix[:,0].map(lambda x: True if len(x.split(','))==len(main_cols) else False)]
    hoods = pd.concat([hoods.ix[:,0].apply(lambda x: pd.Series(x.split(','))), hoods.ix[:,1]], axis=1)
    hoods.columns = ['hood', 'city', 'state', 'region', 'county', 'code']
    return hoods

def load_indicators():
    indicator = pd.read_csv('data/indicator_codes.csv')
    indicator.columns = reversed(list(indicator.columns))
    return indicator


## general empirics
# 

def get_z_scores(df):
    m = df.stack().mean()
    std = df.stack().std()
    z = (df - m) / std
    return z

def get_cum_return(df):  
    df = (df / df.shift())#.replace(np.nan, 0)
    # drop outliers - greater than 3 std moves
    #
    z = get_z_scores(df)
    z_max = z.applymap(lambda x: abs(x)).max()
    outliers = z_max[z_max > 3].index
    df.drop(outliers, axis=1, inplace=True) 
    df = df.cumprod()  
    return df

def get_forward_return(df, periods=24):    
    df = ((df.shift(-periods) / df) - 1.).dropna(how='all')
    # drop outliers - greater than 3 std moves
    #
    z = get_z_scores(df)
    z_max = z.applymap(lambda x: abs(x)).max()
    outliers = z_max[z_max > 3].index
    df.drop(outliers, axis=1, inplace=True)    
    return df

def lead_lag_corr(df_levels, df_returns):
    data = []
    for i in reversed(range(-36,36,2)):
        rec_ret = (df_levels.shift(i) / df_levels - 1.).dropna(how='all')
        df_aligned = pd.concat([rec_ret.stack().to_frame(), df_returns.stack().to_frame()], axis=1).dropna()
        df_aligned.columns = ['past', 'fwd']
        c = df_aligned.corr().values[0,1]
        dif = (df_aligned['fwd'] - df_aligned['past']).mean()
        data.append({'per': i, 'corr': c, 'dif': dif})
    corr = pd.DataFrame(data)
    corr.set_index('per', inplace=True)
    return corr