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
from api_types import GlobalConfig, AgentProps
from data_provider.data_factory import data_provider
from environment.data import DataProcessor, date_to_index, index_to_date
from environment.portfolio import Portfolio
from utils.evals import sharpe, max_drawdown

eps = 1e-8

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
        open_price_vector = observation[:, 0, 0]
        high_price_vector = observation[:, 0, 1]
        low_price_vector = observation[:, 0 , 2]
        close_price_vector = observation[:, 0, 3]

        reset = 0
        y1 = close_price_vector / open_price_vector

        # observation shape (10*3*6)
        # NQPR SHAPE (10*20)
        # TODO: calculate the QPL in real time
         



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