import pandas as pd
import pendulum
import matplotlib.pyplot as plt

df = pd.read_excel('data/gridworks_yearly_data.xlsx', header=3)
df = df.drop(df.columns[[2,3,6]], axis=1)

renamed_columns = {
    'FLO start US/Eastern ': 'date_time',
    'Hour Starting': 'hour',
    'Total Delivered Energy Cost ($/MWh)': 'elec_prices',
    'Outside Temp F': 'oat',
    'House Power Required AvgKw': 'load'
}
df.rename(columns=renamed_columns, inplace=True)

df['elec_prices'] = df['elec_prices']/10

weekday_prices = [0.07026]*7 + [0.50477]*5 + [0.07508]*4 + [0.50477]*4 + [0.07026]*4
weekend_prices = [0.07026]*7 + [0.07508]*13 + [0.07026]*4
def get_price_jan24(row):
    if row['day_type'] == 'weekday':
        return weekday_prices[row['hour']]*100
    elif row['day_type'] == 'weekend':
        return weekend_prices[row['hour']]*100

weekday_prices_jul24 = [0.07099]*7 + [0.15432]*5 + [0.13450]*4 + [0.15432]*4 + [0.07099]*4
weekend_prices_jul24 = [0.07099]*7 + [0.13450]*13 + [0.07099]*4
def get_price_jul24(row):
    if row['day_type'] == 'weekday':
        return weekday_prices_jul24[row['hour']]*100
    elif row['day_type'] == 'weekend':
        return weekend_prices_jul24[row['hour']]*100
    
df['oat'] = df['oat'].apply(lambda x: round(5/9 * (x-32),2))
df['time'] = df['date_time'].dt.round('h')
df['time'] = pd.to_datetime(df['time'], utc=True)
df['time'] = df['time'].dt.tz_convert('America/New_York')
df.drop(columns=['date_time','hour'], inplace=True)
df['hour'] = df['time'].dt.hour
df['dayofweek'] = df['time'].dt.dayofweek
df['day_type'] = df['time'].dt.dayofweek.apply(lambda x: 'weekend' if x >= 5 else 'weekday')
df['jan24_prices'] = df.apply(get_price_jan24, axis=1)
df['jul24_prices'] = df.apply(get_price_jul24, axis=1)

def get_data(time_now, horizon):
    cropped_df = df[df.time >= time_now]
    cropped_df = cropped_df[cropped_df.time <= time_now.add(hours=horizon)]
    return cropped_df

if __name__ == '__main__':
    plt.step(list(range(len(weekday_prices))), weekday_prices, where='post', alpha=0.5, label='weekday')
    plt.step(list(range(len(weekend_prices))), weekend_prices, where='post', alpha=0.5, label='weekend')
    plt.ylim([0,0.6])
    plt.legend()
    plt.show()
    plt.step(list(range(len(weekday_prices))), weekday_prices_jul24, where='post', alpha=0.5, label='weekday')
    plt.step(list(range(len(weekend_prices))), weekend_prices_jul24, where='post', alpha=0.5, label='weekend')
    plt.legend()
    plt.ylim([0,0.6])
    plt.show()
    plt.step(list(range(len(weekday_prices))), weekday_prices, where='post', alpha=0.5, label='Before')
    plt.step(list(range(len(weekday_prices))), weekday_prices_jul24, where='post', alpha=0.5, label='After')
    plt.show()
    time_now = pendulum.datetime(2022, 12, 4, 17, 0, 0, tz='America/New_York')
    print(get_data(time_now, 10))
