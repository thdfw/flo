import pandas as pd
import pendulum
import configparser
import os

config = configparser.ConfigParser()
config.read('parameters.conf')
HORIZON_HOURS = config.getint('parameters', 'HORIZON_HOURS')
MIN_TOP_TEMP_F = config.getint('parameters', 'MIN_TOP_TEMP_F')
MAX_TOP_TEMP_F = config.getint('parameters', 'MAX_TOP_TEMP_F')
TEMP_LIFT_F = config.getint('parameters', 'TEMP_LIFT_F')
NUM_LAYERS = config.getint('parameters', 'NUM_LAYERS')
MAX_HP_ELEC_POWER_KW = config.getfloat('parameters', 'MAX_HP_ELEC_POWER_KW')
MIN_HP_ELEC_POWER_KW = config.getfloat('parameters', 'MIN_HP_ELEC_POWER_KW')
STORAGE_VOLUME_GALLONS = config.getfloat('parameters', 'STORAGE_VOLUME_GALLONS')
STORAGE_LOSSES_PERCENT = config.getfloat('parameters', 'STORAGE_LOSSES_PERCENT')
START_TIME = pendulum.parse(config.get('parameters', 'START_TIME')).set(minute=0, second=0)
INITIAL_TOP_TEMP_F = config.getint('parameters', 'INITIAL_TOP_TEMP_F')
INITIAL_THERMOCLINE = config.getint('parameters', 'INITIAL_THERMOCLINE')
ROOM_TEMPERATURE_F = config.getfloat('parameters', 'ROOM_TEMPERATURE_F')
DD_SWT_F = config.getfloat('parameters', 'DD_SWT_F')
DD_POWER_KW = config.getfloat('parameters', 'DD_POWER_KW')
SHOW_PLOT = config.getboolean('parameters', 'SHOW_PLOT')
DISTRIBUTION_PRICES_CSV = config.get('parameters', 'DISTRIBUTION_PRICES_CSV')
LMP_CSV = config.get('parameters', 'LMP_CSV')
OAT_CSV = config.get('parameters', 'OAT_CSV')
YEARLY_HEAT_LOAD_THERMAL_KWH = config.getfloat('parameters', 'YEARLY_HEAT_LOAD_THERMAL_KWH')
ZERO_HEAT_DELTA_F = config.getfloat('parameters', 'ZERO_HEAT_DELTA_F')


NOW_FOR_FILE = round(pendulum.now('UTC').timestamp())


def get_data(time_now):

    start_year = START_TIME.year
    time_list = pd.date_range(start=f'{start_year}-01-01', end=f'{start_year}-12-31 23:00', freq='h', tz='America/New_York')

    if START_TIME.year != START_TIME.add(hours=HORIZON_HOURS).year:
        raise ValueError("Simulations accross multiple years are currently not supported")

    col1_dist = pd.read_csv(DISTRIBUTION_PRICES_CSV, header=None)
    col2_lmp = pd.read_csv(LMP_CSV, header=None)
    col3_oat = pd.read_csv(OAT_CSV, header=None)

    if len(col1_dist)<8760 or len(col2_lmp)<8760 or len(col3_oat)<8760:
        raise ValueError(f"The length of the provided data is shorter than 8760 hours")

    df = pd.DataFrame({
        'time': time_list,
        'dist': col1_dist[0],
        'lmp': col2_lmp[0],
        'oat': col3_oat[0],
    })

    # Convert to cts/kWh
    df['dist'] = df['dist']/10
    df['lmp'] = df['lmp']/10

    heating_degree_hours = [ROOM_TEMPERATURE_F-ZERO_HEAT_DELTA_F-x for x in list(df.oat)]
    heating_degree_hours = [x if x>=0 else 0 for x in heating_degree_hours]
    heating_degree_hours = [heating_degree_hours[i] 
                            if (time_list[i] < pendulum.datetime(start_year, 5, 15, tz='America/New_York') or
                                time_list[i] >= pendulum.datetime(start_year, 9, 15, tz='America/New_York'))
                                else 0
                                for i in range(len(heating_degree_hours))]
    heating_degree_hours = [x/sum(heating_degree_hours) for x in heating_degree_hours]
    heating_load = [x*YEARLY_HEAT_LOAD_THERMAL_KWH for x in heating_degree_hours]

    max_heating_load_elec = max(heating_load)/COP(min(df.oat),required_SWT(max(heating_load)))
    if max_heating_load_elec > MAX_HP_ELEC_POWER_KW:
        error_text = f"""\n\nOn the coldest hour:
    - The heating requirement is {round(max(heating_load),2)} kW 
    - The COP is {round(COP(min(df.oat),required_SWT(max(heating_load))),2)}
    => Need a HP which can reach {round(max_heating_load_elec,2)} kW electrical power
    => The given HP is undersized ({MAX_HP_ELEC_POWER_KW} kW electrical power)
    """
        raise ValueError(error_text)

    df['load'] = heating_load
    df['required_SWT'] = [required_SWT(x) for x in heating_load]

    return df[(df.time >= time_now) & (df.time <= time_now.add(hours=HORIZON_HOURS))]

def COP(oat, lwt, fahrenheit=True):
    if fahrenheit:
        oat = to_celcius(oat)
        lwt = to_celcius(lwt)
    return 2.35607707 + 0.0232784*oat - 0.00671242*lwt

def to_celcius(t):
    return (t-32)*5/9

def to_fahrenheit(t):
    return t*9/5+32

def required_SWT(power):
    return ROOM_TEMPERATURE_F + (DD_SWT_F-ROOM_TEMPERATURE_F)/DD_POWER_KW * power

def check_parameters():
    authorized_temps = list(range(MIN_TOP_TEMP_F, MAX_TOP_TEMP_F+TEMP_LIFT_F, TEMP_LIFT_F))
    if HORIZON_HOURS<=0:
        raise ValueError('Incorrect parameter: HORIZON must be a positive integer')
    if MIN_TOP_TEMP_F>=MAX_TOP_TEMP_F:
        raise ValueError('Incorrect parameter: MIN_TOP_TEMP_F must be smaller than MAX_TOP_TEMP_F')
    if MIN_TOP_TEMP_F<=to_fahrenheit(0):
        raise ValueError('Incorrect parameter: MIN_TOP_TEMP_F is too low, water must stay in liquid state')
    if MAX_TOP_TEMP_F>=to_fahrenheit(100):
        raise ValueError('Incorrect parameter: MAX_TOP_TEMP_F is too high, water must stay in liquid state')
    if TEMP_LIFT_F<=0:
        raise ValueError('Incorrect parameter: TEMP_LIFT_F must be a positive integer')
    if MAX_TOP_TEMP_F not in authorized_temps:
        raise ValueError(f'Incorrect parameters: There should exist an integer x>0 such that MIN_TOP_TEMP_F + x*TEMP_LIFT_F = MAX_TOP_TEMP_F')
    if NUM_LAYERS < 1:
        raise ValueError('Incorrect parameter: NUM_LAYERS must be larger than 1')
    if MAX_HP_ELEC_POWER_KW < MIN_HP_ELEC_POWER_KW or MAX_HP_ELEC_POWER_KW < 0:
        raise ValueError('Incorrect parameter: MAX_HP_ELEC_POWER_KW must be positive and larger than MIN_HP_ELEC_POWER_KW')
    if STORAGE_VOLUME_GALLONS < 0:
        raise ValueError('Incorrect parameter: STORAGE_VOLUME_GALLONS must be non negative')
    if STORAGE_LOSSES_PERCENT < 0 or STORAGE_LOSSES_PERCENT > 100:
        raise ValueError('Incorrect parameter: STORAGE_LOSSES_PERCENT must be between 0 and 100 %')
    if START_TIME < pendulum.datetime(2022,1,1) or START_TIME > pendulum.now(tz='America/New_York'):
        raise ValueError("Incorrect parameter: START_TIME must be between 2022 and today")
    if INITIAL_TOP_TEMP_F not in authorized_temps:
        raise ValueError(f"Incorrect parameter: INITIAL_TOP_TEMP_F is not in the authorized temperatures: {authorized_temps}")
    if INITIAL_THERMOCLINE<1 or INITIAL_THERMOCLINE>NUM_LAYERS :
        raise ValueError(f"Incorrect parameter: INITIAL_THERMOCLINE must be an integer between 1 and NUM_LAYERS ({NUM_LAYERS})")
    if ROOM_TEMPERATURE_F<=to_fahrenheit(0) or ROOM_TEMPERATURE_F>=to_fahrenheit(40):
        raise ValueError('Incorrect parameter: ROOM_TEMPERATURE_F')
    if DD_SWT_F<=ROOM_TEMPERATURE_F or DD_SWT_F>=to_fahrenheit(100):
        raise ValueError('Incorrect parameter: DD_SWT_F')
    if DD_POWER_KW<=0:
        raise ValueError('Incorrect parameter: DD_POWER_KW must be positive')
    # CSV files
    if not os.path.exists(DISTRIBUTION_PRICES_CSV):
        raise ValueError(f'The path to the CSV file containing distribution prices does not exist: {DISTRIBUTION_PRICES_CSV}')
    if not os.path.exists(LMP_CSV):
        raise ValueError(f'The path to the CSV file containing LMPs does not exist: {LMP_CSV}')
    if not os.path.exists(OAT_CSV):
        raise ValueError(f'The path to the CSV file containing OATs does not exist: {OAT_CSV}')
    if DISTRIBUTION_PRICES_CSV[-4:] != '.csv':
        raise ValueError(f'{DISTRIBUTION_PRICES_CSV.split('/')[-1]} is not a CSV file.')
    if LMP_CSV[-4:] != '.csv':
        raise ValueError(f'{LMP_CSV.split('/')[-1]} is not a CSV file.')
    if OAT_CSV[-4:] != '.csv':
        raise ValueError(f'{OAT_CSV.split('/')[-1]} is not a CSV file.')

check_parameters()