import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import datetime

eps = 1e-8

# Put these into config
# date_format = '%Y-%m-%d'
# start_date = '2014-03-21'
# end_date = '2020-10-14'

# A class that is responsible for data processing
class Dataset_Custom(object):
    def __init__(self, product_list, market_feature, feature_num, steps, window_length, mode, train_ratio, val_ratio, start_index=0, start_date=None):
        
        self.product_list = product_list
        self.market_feature = market_feature
        self.feature_num = feature_num
        # Max step
        self.steps = steps + 1
        self.start_index = start_index
        self.start_date = start_date
        self.window_length = window_length
        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.load_observations()
    
    def load_observations(self):
        # 使用列表推导式读取所有产品的数据
        data_list = [pd.read_csv(f'Data/D_{product}.csv', engine='pyarrow')[self.market_feature].dropna() for product in self.product_list]

        # 获取数据的shape并初始化一个空的数组来存储合并后的数据
        data_shape = data_list[0].shape
        combined_data = np.zeros((data_shape[0], data_shape[1], len(self.product_list)))

        # 将读取到的数据添加到合并后的数组中
        for idx, data in enumerate(data_list):
            combined_data[:, :, idx] = data.values

        combined_data = combined_data[::-1].copy()
        observations = combined_data

        if self.mode == "Train":
            self._data = observations[0:int(self.train_ratio * observations.shape[0])]
            print("Shape for Train observations -- T: ", self._data.shape)
        elif self.mode == "Val":
            self._data = observations[int(self.train_ratio * observations.shape[0]):int((self.train_ratio+self.val_ratio) * observations.shape[0])]
            print("Shape for Val observations -- T: ", self._data.shape)
        elif self.mode == "Test":
            self._data = observations[int((self.train_ratio+self.val_ratio) * observations.shape[0]):]
            print("Shape for Test observations -- T: ", self._data.shape)
        self._data = np.squeeze(self._data)
        self._data = self._data.transpose(2, 0, 1)
    

    def _step(self):

        self.index += 1

        obs = self.data[:, self.index:self.index + self.window_length,:].copy()

        done = self.index >= self.steps

        return obs, done


    def reset(self):
        self.index = 0
        # When training, do not fix the start date
        if self.start_date is None:
            # Ramdom sample date when training
            self.idx = np.random.randint(low=self.window_length, high=self._data.shape[1]-self.steps)
        else:
            # During Validation, Test, Backtest, Fix date
            # Start from beginning
            self.idx = self.window_length
        self.data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1,:]
        obs = self.data[:, self.index:self.index + self.window_length, :].copy()
        return obs







