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
        self.combined_nqpr = self.dataprocessor.get_nqpr()
        self.qpl_level = config.qpl_level
        self.action_size = config.qpl_level + 1
        
    def step(self, action,action_policy):

        # Normalize the action
        action = np.clip(action, 0, 1)
        weights = action
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)

        observation, done1 = self.dataprocessor._step()

        # Connect 1, no risk asset to the portfolio
        c_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((c_observation, observation), axis=0)

        
        open_price_vector = observation[:, -1, 0]
        high_price_vector = observation[:, -1, 1]
        low_price_vector = observation[:, -1 , 2]
        close_price_vector = observation[:, -1, 3]

        # Price relative vector
        pr = observation[:, 0, 3] / observation[:, -1, 3]
        
        # Use qpl levels
        combine_qpl = []
        for i in range(self.qpl_level):
            QPL_vector = open_price_vector * self.combined_nqpr[:,i]
            combine_qpl.append(QPL_vector)
        combine_qpl = np.array(combine_qpl)

        reset = 0
        y1 = np.zeros((10,), dtype=float)
        
        y1[0] = close_price_vector[0]/open_price_vector[0]
        for i in range(1, len(open_price_vector)):
            for j in range(len(combine_qpl)):
                # Select action 0
                # Buy and hold
                if action_policy == 0:
                    y1[i] = pr[i]
                # Select action 1: choose QPL+1
                # for the + QPL in range
                if combine_qpl[j][i] < high_price_vector[i] and combine_qpl[j][i]>low_price_vector[i] and combine_qpl[j][i]!=0 and action_policy==j+1:
                    y1[i] = combine_qpl[j][i]/close_price_vector[i]
                    reset = 1
                # for the +QPL not in range
                if combine_qpl[j][i] > high_price_vector[i] and action_policy==j+1:
                    y1[i] = pr[i]
            
           
        open_price_vector = observation[:, -1, 0]
        high_price_vector = observation[:, -1, 1]
        low_price_vector = observation[:, -1 , 2]
        close_price_vector = observation[:, -1, 3]
        
        
        policy_reward = np.zeros((10,1), dtype=float)
        for i in range(len(open_price_vector)):
            for j in range(len(combine_qpl)):
                if action_policy == 0:
                    policy_reward[i] = close_price_vector[i]-open_price_vector[i]
                # Select action 1: choose QPL+1
                # for the + QPL in range
                if combine_qpl[j][i] < high_price_vector[i] and combine_qpl[j][i]>low_price_vector[i] and combine_qpl[j][i]!=0 and action_policy==j+1:
                    policy_reward[i] = combine_qpl[j][i]-open_price_vector[i]
                # for the +QPL not in range
                if combine_qpl[j][i] > high_price_vector[i] and action_policy==j+1:
                    policy_reward[i] = close_price_vector[i]-open_price_vector[i]
            
        
        policy_reward = np.dot(weights, policy_reward)
        policy_reward = np.sum(policy_reward)
        
        reward, info, done2 = self.portfolio._step(weights, y1, reset)
        info['date'] = index_to_date(self.start_index + self.dataprocessor.idx + self.dataprocessor.index)
        self.infos.append(info)

        return observation, reward, policy_reward, done1 or done2, info

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