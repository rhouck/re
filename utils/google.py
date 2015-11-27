## google geo api data
#

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