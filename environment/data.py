'''
@Author: Yitao Qiu
'''
import numpy as np
import pandas as pd
import datetime


eps = 1e-8
date_format = '%Y-%m-%d'
start_date = '2014-03-21'
end_date = '2020-10-14'
start_datetime = datetime.datetime.strptime(start_date, date_format)
end_datetime = datetime.datetime.strptime(end_date, date_format)

def date_to_index(date_string,start_datetime):
    # Transfer the date to index 0, 1, 2 ,3...
    return (datetime.datetime.strptime(date_string, date_format) - start_datetime).days

def index_to_date(index):
    # Transfer index back to date
    return (start_datetime + datetime.timedelta(index)).strftime(date_format)


# A class that is responsible for data processing
class DataProcessor(object):

    def __init__(self, product_list, market_feature, feature_num, steps, window_length, mode, start_index=0, start_date=None):

        import copy
        self.train_ratio = 0.8
        self.steps = steps + 1
        self.window_length = window_length
        self.window_size = 1
        self.start_index = start_index
        self.start_date = start_date
        self.feature_num = feature_num
        self.market_feature = market_feature
        self.mode = mode
        self._data= []
         
        self.product_list = product_list
        self.load_observations()

    # Load data from the .csv files
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
        elif self.mode == "Test":
            self._data = observations[int(self.train_ratio * observations.shape[0]):]
            print("Shape for Test observations -- T: ", self._data.shape)
        self._data = np.squeeze(self._data)
        self._data = self._data.transpose(2, 0, 1)
        

    def _step(self):

        self.step += 1

        obs = self.data[:, self.step:self.step + self.window_length, :].copy()

        next_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()

        done = self.step >= self.steps

        return obs, done, next_obs

    def reset(self):
        self.step = 0

        
        if self.start_date is None:
            # randomly sample date
            self.idx = np.random.randint(
                low=self.window_length, high=self._data.shape[1] - self.steps)
        else:
            self.idx = date_to_index(self.start_date) - self.start_index
            assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'

        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :8]
        self.data = data
        return self.data[:, self.step:self.step + self.window_length, :].copy(), \
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()