import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def draw_CA():
    """initialize a basemap centered on the California"""
    plt.figure(figsize=(10, 10))
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
