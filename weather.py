import herbie
import pendulum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MILLINOCKET = {'latitude': [45.65329193300256], 'longitude': [-68.71116099379636]}

def get_temp(time, coordinates=MILLINOCKET):
    """
    Finds the outside air temperature at a given point in space and time
    """

    time = time.in_tz('UTC')
    now = pendulum.now('UTC').replace(minute=0, second=0, microsecond=0)

    if time < now:
        H = herbie.Herbie(
            time.strftime("%Y-%m-%d %H:%M"),
            model='hrrr', 
            product='sfc',
            fxx=0,
        )
    else:
        H = herbie.Herbie(
            (now.add(hours=1)).strftime("%Y-%m-%d %H:%M"),
            model='hrrr', 
            product='sfc',
            fxx=int((time - now).total_seconds()/3600)+1,
        )

    ds = H.xarray("TMP:2 m")
    x = herbie.accessors.HerbieAccessor(ds)
    temperature = x.pick_points(pd.DataFrame(coordinates), method='nearest')['t2m'].values[0] 
    temperature = round((temperature - 273.15), 2) # For Fahrenheit: * 9/5 + 32

    return temperature


if __name__ == '__main__':

    today = now = pendulum.now('America/New_York')
    today_midnight = today.replace(hour=0, minute=0, second=0, microsecond=0)

    wf_herbie = []
    for hour in range(24):
        try:
            wf_herbie.append(get_temp(today_midnight.add(hours=hour)))
        except Exception as e:
            print(f"Could not find weather at hour {hour}: {e}")
            wf_herbie.append(np.nan)

    plt.figure(figsize=(10, 5))
    plt.plot(wf_herbie[:today.hour+1],label='Past data', color='tab:blue')
    plt.plot([np.nan]*(today.hour) + wf_herbie[today.hour:], label='Forecast', linestyle='dashed', color='tab:blue')
    plt.scatter(range(24), wf_herbie)
    plt.axvspan(today.hour, today.hour+1, color='orange', alpha=0.1, label='Current hour')
    plt.xticks(list(range(24)))
    plt.xlabel('Time in Maine [hours]')
    plt.ylabel('Outside air temperature [F]')
    plt.title(f'Weather in Maine on {today.date()}')
    plt.legend()
    plt.show()