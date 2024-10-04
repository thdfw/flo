import pandas as pd

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

def required_SWT(oat):
    return 160-2*(oat+15) if oat<5 else 120
    
# import matplotlib.pyplot as plt
# oats = list(range(-15,20,1))
# oats = [to_fahrenheit(x) for x in oats]
# plt.plot(oats, [required_SWT(to_ceclius(x)) for x in oats])
# plt.xlabel('Outside air temperature [F]')
# plt.ylabel('Minimum SWT [F]')
# plt.show()