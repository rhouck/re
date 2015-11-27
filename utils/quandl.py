import os

import pandas as pd
import Quandl

from utils.utils import *


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
    fn = 'data/quandl_api/{0}_{1}.csv'.format(area.lower(), indicator.lower())
    if os.path.isfile(fn):
        print('this data set is already collected')
        return
    
    # validate params
    areas = ('regions', 'states', 'counties', 'cities', 'hoods')
    if area not in areas:
        raise ValueError('area must be one of the following values: {0}'.format(areas))
    
    indicators = load_indicators()
    if indicator.upper() not in indicators.ix[:,1].values:
        raise ValueError('invalid inidicator value: {0}'.format(indicator.upper()))
    
    print('lets scrape {0} / {1}'.format(area, indicator))
    
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
                print(err)
                return
            print('no data for code:\t{0}'.format(i))
            
    
    if df_master.shape[0]:
        df_master.to_csv(fn)
    
    print('done')

def load_quandl_data(area, indicator):
    fn = 'data/quandl_api/{0}_{1}.csv'.format(area.lower(), indicator.lower())
    if not os.path.isfile(fn):
        print('this data set has not been collected')
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
