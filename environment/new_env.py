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
        self.combined_nqpr = self.dataprocessor.get_nqpr()
        self.qpl_level = config.qpl_level
        
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

        
        open_price_vector = observation[:, 0, 0]
        high_price_vector = observation[:, 0, 1]
        low_price_vector = observation[:, 0 , 2]
        close_price_vector = observation[:, -1, 3]
        # Price relative vector
        pr = observation[:, 0, 3] / observation[:, -1, 3]
        
        
        #print(self.combined_nqpr[:,self.qpl_level-1].shape)
        new_qpl = open_price_vector * self.combined_nqpr[:,self.qpl_level-1]
        
        QPL1_vector = observation[:, 0, 4]
        
        

        
        reset = 0
        y1 = np.zeros((10,), dtype=float)
        
        y1[0] = close_price_vector[0]/open_price_vector[0]
        for i in range(1, len(open_price_vector)):
            # Select action 0
            # Buy and hold
            if action_policy == 0:
                y1[i] = pr[i]
            # Select action 1: choose QPL+1
            # for the + QPL in range
            if QPL1_vector[i] < high_price_vector[i] and QPL1_vector[i]>low_price_vector[i] and QPL1_vector[i]!=0 and action_policy==1:
                y1[i] = QPL1_vector[i]/close_price_vector[i]
                reset = 1
            # for the +QPL not in range
            if QPL1_vector[i] > high_price_vector[i] and action_policy==1:
                y1[i] = pr[i]
            
           
        open_price_vector = observation[:, -1, 0]
        high_price_vector = observation[:, -1, 1]
        low_price_vector = observation[:, -1 , 2]
        close_price_vector = observation[:, -1, 3]
        
        QPL1_vector = observation[:, -1, 4]
        QPLn1_vector = observation[:, -1, 5]
        
        c1 = close_price_vector / open_price_vector
        
        policy_reward = np.zeros((10,1), dtype=float)
        for i in range(len(open_price_vector)):
            
            if action_policy == 0:
                policy_reward[i] = close_price_vector[i]-open_price_vector[i]
            
            # Select action 1: choose QPL+1
            # for the + QPL in range
            if QPL1_vector[i] < high_price_vector[i] and QPL1_vector[i]>low_price_vector[i] and QPL1_vector[i]!=0 and action_policy==1:
                policy_reward[i] = QPL1_vector[i]-open_price_vector[i]
            # for the +QPL not in range
            if QPL1_vector[i] > high_price_vector[i] and action_policy==1:
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