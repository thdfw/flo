import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Group all experimental data in a large dataframe
big_df = pd.DataFrame()
for file in os.listdir('data/'):
    if '.csv' in file:
        df = pd.read_csv(f'data/{file}')
        df.drop(columns=['time', 'Unnamed: 0'], inplace=True)
        big_df = pd.concat([big_df, df], ignore_index=True) 

# Model for the COP
def model(X,a,b,c):
    x1, x2 = X
    approx = a + b*x1 + c*x2
    return approx

# Fit the model
x_o = np.array(list(big_df.oat))
x_l = np.array(list(big_df.lwt))
X_list = np.vstack((x_o, x_l))
y = np.array(list(big_df.COP))
popt, pcov = curve_fit(model, X_list, y)
big_df['COP_approx'] = model(X_list, *popt)

def COP(oat,lwt):
    # return 2
    return model([oat,lwt], *popt)

def to_fahrenheit(t):
    return t*9/5 + 32

def to_celcius(t):
    return (t-32)*5/9

# Plot the approximation
if __name__ == '__main__':
    oat_range = np.linspace(-10, 10, 100)
    lwt_range = np.linspace(60, 80, 100)
    oat_grid, lwt_grid = np.meshgrid(oat_range, lwt_range)
    cop_values = COP(oat_grid, lwt_grid)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(oat_grid, lwt_grid, cop_values, cmap='viridis')
    ax.view_init(elev=16, azim=-145)
    ax.set_xlabel('OAT [C]')
    ax.set_xticks([-10,-5,0,5,10])
    ax.set_ylabel('LWT [C]')
    ax.set_yticks([60,65,70,75,80])
    ax.set_zlabel('COP')
    plt.tight_layout()
    plt.show()