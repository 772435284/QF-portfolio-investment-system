import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

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
    
    def load_observations(self):
        pass

    def _step(self):

        self.index += 1

        obs = self.data[:, self.index:self.index + self.window_length,:].copy()

        next_obs = self.data[:, self.index + self.window_length:self.index + self.window_length + 1, :].copy()

        done = self.index >= self.steps

        return obs, done, next_obs


    def reset(self):
        self.index = 0





