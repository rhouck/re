import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from utils import quandl as ql
import utils.utils as ut
from settings import *


def draw_CA():
    """initialize a basemap centered on the California"""
    plt.figure(figsize=(FIG_WIDTH*3, FIG_HEIGHT*3))
    m = Basemap(projection='lcc', resolution='l',
                llcrnrlon=-125, urcrnrlon=-112,
                llcrnrlat=32, urcrnrlat=42,
                lat_0=38, lon_0=-120,
                area_thresh=10000)

    m.fillcontinents(color='white', lake_color='#eeeeee')
    m.drawstates(color='lightgray')
    m.drawcoastlines(color='lightgray')
    m.drawcountries(color='lightgray')
    m.drawmapboundary(fill_color='#eeeeee')
    return m


def load_pred_for_map():

    px = ql.load_quandl_data(TARGET_INDICATOR, TARGET_SERIES).stack()

    pred = pd.read_csv('data/processed/pred.csv', converters={'code': str})
    pred.Date = pd.to_datetime(pred.Date)
    pred = pred.set_index(['Date', 'code'])['pred']

    df = ut.stack_and_align([px, pred], cols=['px', 'pred'])
    df = df.dropna()
    df.index.levels[0].name = 'date'
    df.index.levels[1].name = 'code'

    geos = pd.read_csv('data/geo/{0}.csv'.format(TARGET_INDICATOR), converters={'code': str})
    df = (df.reset_index()
          .merge(geos[['code', 'lon', 'lat']], on='code')
          .set_index(['date', 'code']))

    if TARGET_INDICATOR == 'counties':
        quandl_codes = ql.load_counties()
    elif TARGET_INDICATOR == 'cities':
        quandl_codes = ql.load_cities()
    elif TARGET_INDICATOR == 'hoods':
        quandl_codes = ql.load_hoods()

    df = (df.reset_index()
          .merge(quandl_codes, on='code')
          .set_index(['date', 'code']))
    
    return df