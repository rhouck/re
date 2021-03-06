## model settings
#
RET_PER = 18
TS_HALFLIFE = 36
TARGET_SERIES = 'MVSF'
TARGET_INDICATOR = 'cities'

## plotting settings
# 
FIG_WIDTH = 8
FIG_HEIGHT = 4

## api keys
#
try: 
    from settings_local import GOOGLE_GEO_API_KEY, QUANDL_API_KEY
except:
    GOOGLE_GEO_API_KEY = None
    QUANDL_API_KEY = None
