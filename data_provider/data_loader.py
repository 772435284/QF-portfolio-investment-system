import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import datetime
from utils.qpl import get_nqpr

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
        self.get_nqpr()
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

    def get_nqpr(self):
        # 初始化一个空的DataFrame，用于存储合并后的股票nqpr数据
        combined_nqpr = pd.DataFrame()

        for symbol in self.product_list:
            # 读取每个股票的OHLC数据
            file_path = f'Data/D_{symbol}.csv'
            if os.path.exists(file_path):
                stock_data = pd.read_csv(file_path)
                
                # Combine the 'Year', 'Month', and 'Day' columns to form a single 'Date' column and convert it to datetime format
                stock_data['Date'] = pd.to_datetime(stock_data[['Year', 'Month', 'Day']])

                # Drop the 'Year', 'Month', and 'Day' columns
                stock_data = stock_data.drop(columns=['Year', 'Month', 'Day'])

                # Sort the DataFrame based on the 'Date' column in ascending order
                stock_data = stock_data.sort_values(by='Date')
                
                # 提取收盘价数据
                close_prices = stock_data['Close'].tolist()
                
                # 使用get_nqpr函数计算nqpr
                nqpr = get_nqpr(close_prices)

                # 将nqpr转换为DataFrame，并重命名行索引，以便在最后的DataFrame中识别不同股票的nqpr
                nqpr_df = pd.DataFrame(nqpr[:-1]).T.rename(index={0: symbol})

                # 将nqpr数据添加到合并后的DataFrame中
                combined_nqpr = pd.concat([combined_nqpr, nqpr_df])
            else:
                print(f"File {file_path} does not exist.")

        # 创建一个名为Fiat_Currency的Series，数值全部为1
        # 创建一个名为Fiat_Currency的DataFrame，数值全部为1
        fiat_currency = pd.DataFrame([[1.0] * 20], columns=combined_nqpr.columns, index=['Fiat_Currency'])

        # 将Fiat_Currency这一行添加到combined_nqpr的最前面
        combined_nqpr = pd.concat([fiat_currency, combined_nqpr])

        combined_nqpr = combined_nqpr.to_numpy()
        
        return combined_nqpr
        
        

        

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







