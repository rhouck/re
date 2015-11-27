import os
import requests

import pandas as pd
import numpy as np

from utils.utils import *


def poll_google_geo_api(locs, api_key):
    q = ','.join(['+' + i.strip().replace(' ','+') for i in locs])[1:]
    q += ',+CA'
    payload = {'key': api_key, 'address': q}
    r = requests.get('https://maps.googleapis.com/maps/api/geocode/json?', params=payload)
    if r.status_code == 200:
        res = r.json()['results']
        return res
    else:
        raise Exception('unsuccessful request - status code {0}'.format(r.status_code))


def parse_google_geo_response(res):
    ac = res[0]['address_components']
    loc_type = ac[0]['types'][0]
    name = ', '.join([i['long_name'] for i in ac])
    loc = res[0]['geometry']['location']
    return {'lat': loc['lat'], 'lon': loc['lng'], 'type': loc_type, 'name': name}


def collect_google_geo_data(locs):
    """
    parameters: 
    locs: dict of location data with quandl code arrays

    Usage::
    hoods = load_hoods()
    cities = load_cities()   
    locs = {'cities': cities[['city', 'county', 'code']].values, 
            'hoods': hoods[['hood', 'city', 'county', 'code']].values}
    """
    for l in locs.items():

        print('collecting {0} geos'.format(l[0]))
        # check file exists
        fn = 'data/geo_data/{0}.csv'.format(l[0])
        if os.path.isfile(fn):
            print('this data set is already collected')
            continue

        rows = []
        for ind, row in enumerate(l[1]):
            try:
                res = poll_google_geo_api(row[:-1], GOOGLE_GEO_API_KEY)
                res = parse_google_geo_response(res)
                res['code'] = row[-1]
                rows.append(res)
            except Exception as err:
                print('error collecting: {0}\terr: {1}'.format(row, err))
        df = pd.DataFrame(rows)
        df.to_csv(fn, index=False)
    
    print('done')
