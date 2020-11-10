"""
Contains a set of utility function to process data
"""

from __future__ import print_function

import csv
import datetime
import numpy as np
import pandas as pd


def normalize(x):
    return (x - 1) * 100

def load_observations(window_size,market_feature,feature_num,product_list):
    product_list = product_list
    ts_d = pd.read_csv('Data/'+'D_'+'AUDUSD'+'.csv')
    ts_d_len = len(ts_d)
    data = np.zeros((ts_d_len-window_size+1,feature_num, len(product_list),window_size ), dtype=float)
    
    for k in range(len(product_list)):
        product = product_list[k]
        #print(product)
        ts_d = pd.read_csv('Data/'+'D_'+product+'.csv')
        ts_d = ts_d.dropna(axis=0,how='any')
        for j in range(len(market_feature)):
            ts_d_temp = ts_d[market_feature[j]].values
            for i in range(len(ts_d)-window_size+1):
                temp = np.zeros((window_size))
                for t in range(i, i+window_size):

                #temp = np.zeros((para_num))
                    temp[t-i] = ts_d_temp[t]
                #print(temp)
                data[i][j][k] = temp
    data = data[::-1].copy()
    observations = data

    print("Shape for observations -- T: ", observations.shape)
    return observations,ts_d_len


