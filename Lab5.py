#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab05

import streamlit as st
import requests
from urllib.parse import quote

WEATHER_CACHE_SECONDS = 1800 #30 minutes, weather for "today" does not need refetching more often


#### PART A, STEP 3: WEATHER DATA FUNCTION ####
def _clean_location(location):
    return ' '.join(str(location).split()) #Trims the ends and collapses repeated spaces


def _url_safe(location):
    #wttr.in reads '+' as a space (e.g. 'Eiffel+Tower'), anything else unusual is percent-encoded
    return quote(location.replace(' ', '+'), safe='+,')


@st.cache_data(ttl=WEATHER_CACHE_SECONDS, show_spinner=False)
def _fetch_weather_json(location):
    #Only the network call is cached. A failed lookup raises before returning, so errors are never cached
    url = f'https://wttr.in/{_url_safe(location)}?format=j1'

    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e: #Timeout, no connection, etc.
        raise Exception(f'Could not reach wttr.in: {e}')

    if response.status_code != 200:
        raise Exception(f'wttr.in error: status {response.status_code}')

    try:
        return response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f'Could not find a location named {location}')


def _first_value(section, key): #wttr.in wraps most text values as [{'value': '...'}]
    try:
        return section[key][0]['value']
    except (KeyError, IndexError, TypeError):
        return ''


# location can be a city, a zip code, an airport code ('SYR'),
# or a landmark ('Eiffel+Tower')
# note: hard codes units to degrees Fahrenheit
def get_current_weather(location):
    location = _clean_location(location)

    if not location:
        raise Exception('No location was given') #The default location is decided in Part B

    data = _fetch_weather_json(location)

    # j1 has three top-level sections:
    #   current_condition -- one entry, conditions right now
    #   weather           -- three entries, one per day, each with
    #                        min/max, astronomy, and hourly forecasts
    #   nearest_area      -- the location wttr.in actually matched
    current = data['current_condition'][0]

    area = (data.get('nearest_area') or [{}])[0]
    matched_parts = [_first_value(area, key) for key in ('areaName', 'region', 'country')]
    matched_location = ', '.join(part for part in matched_parts if part)

    # two examples; note that some values are nested one level deeper
    return {'location': location,
            'matched_location': matched_location,
            'temperature': float(current['temp_F']),
            'description': current['weatherDesc'][0]['value']
            }


#### MAIN APP ####
st.title(':blue[Lab 5:] :grey[What to Wear] Bot')


#### PART A, STEP 4: TEST BLOCK - COMMENT OUT FOR PART B ####
test_location = st.sidebar.text_input('Test location', placeholder='e.g., Syracuse, NY')

if test_location:
    try:
        with st.spinner('Checking the weather...'):
            weather = get_current_weather(test_location)

        st.subheader(f'Results for: {test_location}')
        st.json(weather)

        if not weather['matched_location']: #Would mean the nearest_area key names differ from what we assumed
            st.warning('No matched location came back. Check the nearest_area section of the full j1 response.')

    except Exception as e:
        st.error(f'Weather lookup failed: {e}')

    raw_url = f'https://wttr.in/{_url_safe(_clean_location(test_location))}?format=j1' #Same URL the function requests
    st.sidebar.link_button('Open full j1 response', raw_url)

else:
    st.info('Enter a location in the sidebar to test get_current_weather.')
#### PART A TEST BLOCK - END ####
