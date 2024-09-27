import pandas as pd
import pendulum

df = pd.read_excel('gridworks_yearly_data.xlsx', header=3)
df = df.drop(df.columns[[2,3,6]], axis=1)

renamed_columns = {
    'FLO start US/Eastern ': 'date_time',
    'Hour Starting': 'hour',
    'Total Delivered Energy Cost ($/MWh)': 'elec_prices',
    'Outside Temp F': 'oat',
    'House Power Required AvgKw': 'load'
}
df.rename(columns=renamed_columns, inplace=True)

df['oat'] = df['oat'].apply(lambda x: round(5/9 * (x-32),2))

df['time'] = df['date_time'].dt.round('h')
df['time'] = pd.to_datetime(df['time'], utc=True)
df['time'] = df['time'].dt.tz_convert('America/New_York')
df.drop(columns=['date_time','hour'], inplace=True)

def get_data(time_now, horizon):
    cropped_df = df[df.time >= time_now]
    cropped_df = cropped_df[cropped_df.time <= time_now.add(hours=horizon)]
    return cropped_df

if __name__ == '__main__':
    time_now = pendulum.datetime(2022, 12, 1, 14, 0, 0, tz='America/New_York')
    print(get_data(time_now, 2))
