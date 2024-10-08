import pandas as pd
import pendulum
import configparser

config = configparser.ConfigParser()
config.read('parameters.conf')
HORIZON_HOURS = config.getint('parameters', 'HORIZON_HOURS')
MIN_TOP_TEMP_F = config.getint('parameters', 'MIN_TOP_TEMP_F')
MAX_TOP_TEMP_F = config.getint('parameters', 'MAX_TOP_TEMP_F')
TEMP_LIFT_F = config.getint('parameters', 'TEMP_LIFT_F')
NUM_LAYERS = config.getint('parameters', 'NUM_LAYERS')
MAX_HP_POWER_KW = config.getfloat('parameters', 'MAX_HP_POWER_KW')
MIN_HP_POWER_KW = config.getfloat('parameters', 'MIN_HP_POWER_KW')
STORAGE_VOLUME_GALLONS = config.getfloat('parameters', 'STORAGE_VOLUME_GALLONS')
LOSSES_PERCENT = config.getfloat('parameters', 'LOSSES_PERCENT')
CONSTANT_COP = config.getfloat('parameters', 'CONSTANT_COP')
TOU_RATES = config.get('parameters', 'TOU_RATES')
START_TIME = pendulum.parse(config.get('parameters', 'START_TIME')).set(minute=0, second=0)
START_TOP_TEMP_F = config.getint('parameters', 'START_TOP_TEMP_F')
START_THERMOCLINE = config.getint('parameters', 'START_THERMOCLINE')
ROOM_TEMPERATURE_F = config.getfloat('parameters', 'ROOM_TEMPERATURE_F')
DD_SWT_F = config.getfloat('parameters', 'DD_SWT_F')
DD_POWER_KW = config.getfloat('parameters', 'DD_POWER_KW')

df = pd.read_csv('yearly_data_2022.csv')

def get_data(time_now, horizon):
    time_now = time_now.timestamp() - 5*3600
    return df[(df.time >= time_now) & (df.time <= time_now + horizon * 3600)]

def COP(oat,lwt):
    return 2.42213946 + 0.02055517*oat -0.00819491*lwt

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
    if MAX_HP_POWER_KW < MIN_HP_POWER_KW or MAX_HP_POWER_KW < 0:
        raise ValueError('Incorrect parameter: MAX_HP_POWER_KW must be positive and larger than MIN_HP_POWER_KW')
    if STORAGE_VOLUME_GALLONS < 0:
        raise ValueError('Incorrect parameter: STORAGE_VOLUME_GALLONS must be non negative')
    if LOSSES_PERCENT < 0 or LOSSES_PERCENT > 100:
        raise ValueError('Incorrect parameter: LOSSES_PERCENT must be between 0 and 100 %')
    if CONSTANT_COP < 0:
        raise ValueError('Incorrect parameter: CONSTANT_COP must be non negative')
    if TOU_RATES not in ['new', 'old']:
        raise ValueError('Incorrect parameter: TOU_RATES')
    if START_TIME < pendulum.datetime(2022,1,1) or START_TIME > pendulum.now(tz='America/New_York'):
        raise ValueError("Incorrect parameter: START_TIME must be between 2022 and today")
    if START_TOP_TEMP_F not in authorized_temps:
        raise ValueError(f"Incorrect parameter: START_TOP_TEMP_F is not in the authorized temperatures: {authorized_temps}")
    if START_THERMOCLINE<1 or START_THERMOCLINE>NUM_LAYERS :
        raise ValueError(f"Incorrect parameter: START_THERMOCLINE must be an integer between 1 and NUM_LAYERS ({NUM_LAYERS})")
    if ROOM_TEMPERATURE_F<=to_fahrenheit(0) or ROOM_TEMPERATURE_F>=to_fahrenheit(40):
        raise ValueError('Incorrect parameter: ROOM_TEMPERATURE_F')
    if DD_SWT_F<=ROOM_TEMPERATURE_F or DD_SWT_F>=to_fahrenheit(100):
        raise ValueError('Incorrect parameter: DD_SWT_F')
    if DD_POWER_KW<=0:
        raise ValueError('Incorrect parameter: DD_POWER_KW must be positive')

check_parameters()