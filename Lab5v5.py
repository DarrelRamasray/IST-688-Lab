#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab05

import streamlit as st
import requests
import json
from urllib.parse import quote
from openai import OpenAI

WEATHER_CACHE_SECONDS = 1800 #30 minutes, weather for "today" does not need refetching more often
DEFAULT_LOCATION = 'Syracuse, NY' #Part B, Step 6b
CHAT_MODEL = 'gpt-5.4-mini' #Same model as Lab 4


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
        return str(section[key][0]['value']).strip()
    except (KeyError, IndexError, TypeError):
        return ''


def _to_number(value): #wttr.in sends numbers as text ('59', '0.0'); None if missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return int(number) if number.is_integer() else number


def _format_hour(raw_time): #Hourly times come as '0', '300', ... '2100'
    try:
        hour = int(raw_time) // 100
    except (TypeError, ValueError):
        return str(raw_time)

    suffix = 'AM' if hour < 12 else 'PM'
    return f'{hour % 12 or 12} {suffix}'


# location can be a city, a zip code, an airport code ('SYR'),
# or a landmark ('Eiffel+Tower')
# note: hard codes units to degrees Fahrenheit
def get_current_weather(location=None):
    location = _clean_location(location or '')

    used_default = not location

    if used_default:
        location = DEFAULT_LOCATION #Part B, Step 6b: Python's half of the default

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

    #Part B, Step 5b: today's range and timeline. weather[0] is today, and its
    #eight hourly slots (every 3 hours, local time) show how the day changes
    today = (data.get('weather') or [{}])[0]
    astronomy = (today.get('astronomy') or [{}])[0]

    hourly_today = []

    for slot in today.get('hourly', []): #Trimmed to what affects clothing and outdoor plans
        hourly_today.append({
            'time': _format_hour(slot.get('time')),
            'temperature': _to_number(slot.get('tempF')),
            'feels_like': _to_number(slot.get('FeelsLikeF')),
            'description': _first_value(slot, 'weatherDesc'),
            'chance_of_rain': _to_number(slot.get('chanceofrain')),
            'chance_of_snow': _to_number(slot.get('chanceofsnow')),
            'wind_speed': _to_number(slot.get('windspeedMiles')),
            'wind_gusts': _to_number(slot.get('WindGustMiles')),
            'uv_index': _to_number(slot.get('uvIndex'))
        })

    # two examples; note that some values are nested one level deeper
    return {'location': location,
            'matched_location': matched_location,
            'used_default_location': used_default, #Lets the model tell the user no city was given
            'units': 'temperatures in F, wind in mph, precipitation in inches, humidity and chances in percent',
            'temperature': float(current['temp_F']),
            'description': current['weatherDesc'][0]['value'].strip(), #wttr.in sometimes adds a trailing space ('Overcast ')
            'feels_like': _to_number(current.get('FeelsLikeF')),
            'humidity': _to_number(current.get('humidity')),
            'wind_speed': _to_number(current.get('windspeedMiles')),
            'precipitation': _to_number(current.get('precipInches')),
            'uv_index': _to_number(current.get('uvIndex')),
            'today': {
                'date': today.get('date', ''),
                'high': _to_number(today.get('maxtempF')),
                'low': _to_number(today.get('mintempF')),
                'sunrise': astronomy.get('sunrise', ''),
                'sunset': astronomy.get('sunset', '')
            },
            'hourly_today': hourly_today
            }


#### PART B, STEP 5b: THE TOOL THE MODEL READS ####
#Same shape as the lecture's get_current_weather example. The model only sees this
#description, never the Python code, so the wording is what guides its choices
WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': ("Get today's weather for a location: current conditions, today's high and low, "
                        "sunrise and sunset, and a three-hour forecast timeline for the day. "
                        f"If the user gives no location, use '{DEFAULT_LOCATION}'."), #Step 6b: the model's half of the default
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': ("City and state or country, e.g. 'Syracuse, NY' or 'Lima, Peru'. "
                                    f"Use '{DEFAULT_LOCATION}' if the user did not give one.")
                }
            }
            #No 'required' list on purpose: if the model leaves location out, Python fills in the default
        }
    }
}

TOOL_SYSTEM_PROMPT = f"""You help people decide what to wear today and which outdoor activities suit the weather.
Use the get_current_weather tool to get the weather you need before giving advice.
Use the City the user gives. If the City is blank but their plans name a place, use that place.
If no location is given anywhere, use '{DEFAULT_LOCATION}'."""


def run_weather_tool(arguments_json): #Reads the model's arguments, which arrive as a JSON string
    try:
        args = json.loads(arguments_json or '{}')
    except json.JSONDecodeError:
        args = {}

    if not isinstance(args, dict):
        args = {}

    return get_current_weather(args.get('location')) #A missing location falls through to the default


def build_user_message(city, plans):
    lines = [f'City: {city.strip() or "(blank)"}']

    if plans.strip():
        lines.append(f'Plans: {plans.strip()}')

    return '\n'.join(lines)


#### PART B, STEP 7: SECOND CALL PIECES ####
ADVICE_PROMPT = f"""Using the weather results above, give today's advice in two parts.
What to wear: specific clothing for the day, and when to add or remove layers based on the hourly timeline.
Outdoor activities: two or three activities that suit the weather, with the best time of day for each.
If the user shared plans, tailor both parts to those plans.
Name the location using matched_location. If the City was blank and no place was named in the plans,
say the advice is for {DEFAULT_LOCATION}, the default.
You do not know the current local time, so cover the whole day rather than assuming the hour.
If a weather lookup returned an error, say so plainly and do not guess the weather.
Keep it concise."""


def assistant_tool_request(reply): #The model's tool request as a plain dict, the same content as the lecture's to_dict()
    return {
        'role': 'assistant',
        'content': reply.content,
        'tool_calls': [
            {'id': call.id,
             'type': 'function',
             'function': {'name': call.function.name, 'arguments': call.function.arguments}}
            for call in reply.tool_calls
        ]
    }


def run_tool_call(call): #Returns (text for the tool message, weather data or None). Always returns text,
                         #because every tool request needs a matching tool message
    if call.function.name != 'get_current_weather':
        return json.dumps({'error': f'Unknown tool: {call.function.name}'}), None

    try:
        weather = run_weather_tool(call.function.arguments)
        return json.dumps(weather), weather
    except Exception as e: #The model is told about the failure instead of the app crashing
        return json.dumps({'error': str(e)}), None


#### PART B: WEATHER SNAPSHOT (shown above the advice) ####
def _peak(slots, key): #Highest value in today's timeline, and the time it happens
    values = [(slot[key], slot['time']) for slot in slots if slot.get(key) is not None]
    return max(values, key=lambda pair: pair[0]) if values else (None, '')


def _fmt(value, suffix=''): #A dash instead of 'None' when wttr.in leaves a value out
    return '—' if value is None else f'{value:.0f}{suffix}'


def show_weather_snapshot(weather, city_blank):
    place = weather['matched_location'] or weather['location']

    #Covers both halves of the default: Python filled it in, or the model sent Syracuse for a blank City
    used_default = weather['used_default_location'] or (city_blank and weather['location'] == DEFAULT_LOCATION)
    note = ' · default location, no city entered' if used_default else ''

    st.markdown(f"**Weather for {place}**")
    st.caption(f"{weather['description']}{note}")

    today = weather['today']
    slots = weather['hourly_today']

    rain, rain_time = _peak(slots, 'chance_of_rain')
    snow, snow_time = _peak(slots, 'chance_of_snow')

    if snow is not None and (rain is None or snow > rain): #Syracuse winters: show snow when it is the bigger risk
        precip_label, precip_value, precip_time = 'Snow chance', snow, snow_time
    else:
        precip_label, precip_value, precip_time = 'Rain chance', rain, rain_time

    uv, uv_time = _peak(slots, 'uv_index')

    cols = st.columns(5)
    cols[0].metric('Now', _fmt(weather['temperature'], '°F'))
    cols[1].metric('Feels like', _fmt(weather['feels_like'], '°F'))
    cols[2].metric('High / Low', f"{_fmt(today['high'], '°')} / {_fmt(today['low'], '°')}")
    cols[3].metric(precip_label, _fmt(precip_value, '%'),
                   help=f"Highest chance in today's forecast, around {precip_time}" if precip_time else None)
    cols[4].metric('UV index', _fmt(uv),
                   help=f'Highest today, around {uv_time}' if uv_time else None)


#### MAIN APP ####
st.title(':blue[Lab 5:] :grey[What to Wear] Bot')

if 'openai_client' not in st.session_state: #Same pattern as Lab 4
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)


#### PART A, STEP 4: TEST BLOCK - COMMENTED OUT FOR PART B ####
# test_location = st.sidebar.text_input('Test location', placeholder='e.g., Syracuse, NY')
#
# if test_location:
#     try:
#         with st.spinner('Checking the weather...'):
#             weather = get_current_weather(test_location)
#
#         st.subheader(f'Results for: {test_location}')
#         st.json(weather)
#
#         if not weather['matched_location']: #Would mean the nearest_area key names differ from what we assumed
#             st.warning('No matched location came back. Check the nearest_area section of the full j1 response.')
#
#     except Exception as e:
#         st.error(f'Weather lookup failed: {e}')
#
#     raw_url = f'https://wttr.in/{_url_safe(_clean_location(test_location))}?format=j1' #Same URL the function requests
#     st.sidebar.link_button('Open full j1 response', raw_url)
#
# else:
#     st.info('Enter a location in the sidebar to test get_current_weather.')
#### PART A TEST BLOCK - END ####


#### PART B, STEP 5a: INPUTS ####
city = st.text_input('City', placeholder='e.g., Syracuse, NY', max_chars=100)

add_plans = st.checkbox('Add plans for today')
plans = ''

if add_plans: #The plans box only appears while the checkbox is ticked
    plans = st.text_area('What are you planning?',
                         placeholder='e.g., Walking tour of campus this afternoon, dinner outside at 7 PM',
                         max_chars=300) #Keeps the prompt short, since every character sent is billed

#### PART B, STEPS 6-7: THE TWO CALLS ####
if st.button('Get my advice', type='primary'):
    client = st.session_state.openai_client

    messages = [
        {'role': 'system', 'content': TOOL_SYSTEM_PROMPT},
        {'role': 'user', 'content': build_user_message(city, plans)}
    ]

    #Step 6: first call. The model decides whether it needs the weather
    try:
        with st.spinner('Reading your request...'):
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=[WEATHER_TOOL],
                tool_choice='auto' #Step 6a
            ) #Not streamed, because the app has to inspect the full reply for tool_calls

    except Exception as e:
        st.error(f'This request has failed: {e}')
        st.stop()

    reply = response.choices[0].message

    if not reply.tool_calls: #The model answered without asking for the weather
        st.write(reply.content)
        st.stop()

    #Step 7: the model's own tool request goes back into the conversation first...
    messages.append(assistant_tool_request(reply))

    #...then one 'tool' message per request, matched by its id (the slide also passes a name; the API only needs the id)
    snapshots = {} #Keyed by place, so a repeated request for the same place shows one snapshot

    with st.spinner('Checking the weather...'):
        for call in reply.tool_calls:
            content, weather = run_tool_call(call)

            messages.append({
                'role': 'tool',
                'tool_call_id': call.id,
                'content': content
            })

            if weather:
                snapshots[weather['matched_location'] or weather['location']] = weather

    for weather in snapshots.values():
        show_weather_snapshot(weather, city_blank=not city.strip())

    if snapshots:
        st.divider()

    messages.append({'role': 'system', 'content': ADVICE_PROMPT}) #Step 7b: ask for clothing and activities

    #Step 7: second call. No tools this time, so the model has to answer with the weather it now has
    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )

        st.write_stream(stream) #Same streaming display as Lab 4

    except Exception as e:
        st.error(f'This request has failed: {e}')
