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
from utils.evals import sharpe, max_drawdown,annualized_sharpe_ratio,annualized_return,annual_volatility

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

        observation, done1, groud_truth = self.dataprocessor._step()

        # Connect 1, no risk asset to the portfolio
        c_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((c_observation, observation), axis=0)
        
        c_groud_truth = np.ones((1, 1, groud_truth.shape[2]))
        groud_truth = np.concatenate((c_groud_truth, groud_truth), axis=0)
        
        open_price_vector = groud_truth[:, -1, 0]
        high_price_vector = groud_truth[:, -1, 1]
        low_price_vector = groud_truth[:, -1 , 2]
        close_price_vector = groud_truth[:, -1, 3]
        # Price relative vector
        
        combine_qpl = np.array([open_price_vector * self.combined_nqpr[:, i] for i in range(self.qpl_level)])
        
        reset = 0
        y1 = np.zeros((10,), dtype=float)
        y1[0] = close_price_vector[0] / open_price_vector[0]

        for i in range(1, len(open_price_vector)):
            if action_policy == 0:
                y1[i] = close_price_vector[i] / open_price_vector[i]
            else:
                for j, qpl in enumerate(combine_qpl):
                    in_range = qpl[i] < high_price_vector[i] and qpl[i] > low_price_vector[i] and qpl[i] != 0
                    out_of_range = qpl[i] > high_price_vector[i]

                    if action_policy == j + 1:
                        if in_range:
                            y1[i] = qpl[i] / open_price_vector[i]
                        elif out_of_range:
                            y1[i] = close_price_vector[i] / open_price_vector[i]
            
        
        policy_reward = np.zeros((10, 1), dtype=float)

        for i in range(len(open_price_vector)):
            if action_policy == 0:
                policy_reward[i] = close_price_vector[i] - open_price_vector[i]
            else:
                for j, qpl in enumerate(combine_qpl):
                    in_range = qpl[i] < high_price_vector[i] and qpl[i] > low_price_vector[i] and qpl[i] != 0
                    out_of_range = qpl[i] > high_price_vector[i]

                    if action_policy == j + 1:
                        if in_range:
                            policy_reward[i] = qpl[i] - open_price_vector[i]
                        elif out_of_range:
                            policy_reward[i] = close_price_vector[i] - open_price_vector[i]
            
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
        a_return = annualized_return(df_info.rate_of_return)
        a_volatility = annual_volatility(df_info.rate_of_return)
        culmulative_return = ((df_info.portfolio_value[-1] - df_info.portfolio_value[0]) / df_info.portfolio_value[0])*100
        print("Max drawdown", mdd)
        print("Sharpe ratio",sharpe_ratio)
        print("annualized return",a_return)
        print("annualized volatility",a_volatility)
        print("culmulative return",culmulative_return,"%")
        print("Final portfolio value", df_info["portfolio_value"][-1])

        return sharpe_ratio , mdd, df_info["portfolio_value"][-1],culmulative_return, a_return, a_volatility