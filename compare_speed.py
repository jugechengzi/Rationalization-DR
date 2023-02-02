import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

def read_data(file):
    df=pd.read_csv(file)
    return df

aspect=1

# fig, ax = plt.subplots()


for value_type in ['_1_1','_01_1','_1_01']:
    plt.xlabel('epochs', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    for dis in ['pred','gen']:
        file = './time_data/compare_speed/a4_p=2_{}{}.csv'.format(dis,value_type)
        a = read_data(file)
        x=a.iloc[:,1].values
        y=a.iloc[:,2].values
        plt.plot(x[:200],y[:200])
    plt.legend(['pred','gen'], fontsize=15,loc='upper right')
    plt.axvline(x=300, ls="-", c="gray", linewidth=0.3)
    plt.axvline(x=600, ls="-", c="gray", linewidth=0.3)
    plt.axvline(x=200, ls="-", c="gray", linewidth=0.3)
    plt.show()

