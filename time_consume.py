import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

def read_data(file):
    df=pd.read_csv(file)
    return df

aspect=2

# fig, ax = plt.subplots()


for value_type in ['f1','acc','cls','dev']:
    plt.xlabel('epochs', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    for dis in ['rnp','dis']:
        file = './time_data/as{}_{}_{}.csv'.format(aspect, dis,value_type)
        a = read_data(file)
        x=a.iloc[:,1].values
        y=a.iloc[:,2].values
        plt.plot(x,y)
        if value_type =='f1':
            plt.ylabel('F1', fontsize=15)
        elif value_type =='acc':
            plt.ylabel('Acc', fontsize=15)
            plt.ylim(0.5,0.95)
        elif value_type == 'cls':
            plt.ylabel('Prediction Loss', fontsize=15)
        elif value_type == 'dev':
            plt.ylabel('Dev Acc', fontsize=15)
            plt.ylim(0.5,0.95)
    if value_type == 'cls':
        plt.legend(['RNP', 'RS'], fontsize=15,loc='lower left')
    else:
        plt.legend(['RNP','RS'],fontsize=15,loc='lower right')
    print('yes')
    plt.show()