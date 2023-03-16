"""
Inspired by https://github.com/wassname/rl-portfolio-management/blob/master/src/environments/portfolio.py
and https://github.com/vermouth1992/drl-portfolio-management/blob/master/src/environment/portfolio.py, which are 
based on [Jiang 2017](https://arxiv.org/abs/1706.10059) 
https://github.com/ZhengyaoJiang/PGPortfolio
"""

'''
@Author: Yitao Qiu
'''

import numpy as np
import pandas as pd
import matplotlib as plt
import datetime
import gym
import gym.spaces
#from environment.data import DataProcessor, date_to_index, index_to_date
from data_provider.data_factory import data_provider
from api_types import GlobalConfig, AgentProps
from environment.portfolio import Portfolio
from environment.data import DataProcessor, date_to_index, index_to_date


eps = 1e-8

def sharpe(returns, freq=252, rfr=0):
    # The function that is used to caculate sharpe ratio
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)

def max_drawdown(return_list):
    # The function that is used to calculate the max drawndom
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i]) 
    return (return_list[j] - return_list[i]) / (return_list[j])

# A class for portfolio enviroment
class envs(gym.Env):
    def __init__(self,
                 config:GlobalConfig):

        self.window_length = config.window_size
        self.start_index = 0
        self.mode = config.mode
        self.dataprocessor = data_provider(config)
        if self.mode == "Train":
            trading_cost = 0.0000
        elif self.mode == "Test":
            trading_cost = 0.0025
        self.portfolio = Portfolio(steps=config.max_step,trading_cost=trading_cost, mode=config.mode)
        
        
    def step(self, action):

        # Normalize the action
        weights = np.clip(action, 0, 1)
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)

        observation, done1, = self.dataprocessor._step()

        # Connect 1, no risk asset to the portfolio
        c_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((c_observation, observation), axis=0)


        # Obtain the price vector
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]

        reset = 0
        y1 = observation[:, 0, 3] / observation[:, 0, 0]

        reward, info, done2 = self.portfolio._step(weights, y1, reset)
        info['date'] = index_to_date(self.start_index + self.dataprocessor.idx + self.dataprocessor.index)
        self.infos.append(info)

        return observation, reward, done1 or done2, info

    def reset(self):
        self.infos = []
        self.portfolio.reset()
        observation = self.dataprocessor.reset()
        c_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((c_observation, observation), axis=0)
        
        info = {}
        return observation, info

    def render(self):
        df_info = pd.DataFrame(self.infos)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.portfolio_value)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        print("Max drawdown", mdd)
        print("Sharpe ratio",sharpe_ratio)
        print("Final portfolio value", df_info["portfolio_value"][-1])