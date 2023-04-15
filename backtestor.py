import os
from os.path import join as path_join
import numpy as np
from typing import cast
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from api_types import GlobalConfig, AgentProps
import math
import yaml
from torch.distributions import Categorical
from utils.utils import normalize,load_observations
from tools.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
import models.DDPG as DDPG
import models.A2C as A2C
import models.QFPIS as QFPIS
from models.QFPIS import Policy
from environment.env import envs
from typing import Callable, List, cast, OrderedDict
from observation.obs_creator import obs_creator

MODEL_DICT = {
    "A2C": A2C,
    "DDPG": DDPG,
    "QFPIS": QFPIS
    # "OtherModel": OtherModelClass
}


class backtestor(object):
    config: GlobalConfig

        
    def __init__(self, env, actor_noise: Callable, device: str, config: GlobalConfig):
        self.env = env
        self.actor_noise = actor_noise
        self.device = device
        self.config = config
        self.agent_index = cast(int, config.use_agents)
        self.product_list = AgentProps(config.agent_list[self.agent_index]).product_list
        self.product_num = len(self.product_list)
        self.window_size = config.window_size
        self.market_feature = config.market_feature
        self.feature_num = len(self.market_feature)
        self.steps = config.max_step
        self.mode = config.mode
        self.action_size = config.qpl_level + 1
        self.num_features = len(config.factor)
    
    

    def load_actor(self, model_type, isbaseline=True):
        model_class = MODEL_DICT.get(model_type)
        self.actor = model_class.Actor(product_num=self.product_num,win_size=self.window_size,num_features=self.num_features).to(self.device)
        if isbaseline:
            self.actor.load_state_dict(torch.load(path_join(self.config.baseline_dir, AgentProps(self.config.agent_list[self.agent_index]).name )))
        else:
            self.actor.load_state_dict(torch.load(path_join(self.config.ddpg_model_dir, AgentProps(self.config.agent_list[self.agent_index]).name )))
        

    def load_policy(self,action_size):
        self.policy = Policy(product_num = self.product_num, win_size = self.window_size,num_features=self.num_features, action_size = action_size).to(self.device)
        self.policy.load_state_dict(torch.load(path_join(self.config.pga_model_dir, AgentProps(self.config.agent_list[self.agent_index]).name)))
        
    def backtest_A2C(self):
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        observation, info = self.env.reset()
        observation = creator.create_obs(observation)
        done = False
        ep_reward = 0
        wealth = self.config.wealth
        # Collect culmulative return
        CR = []
        while not done:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)
            action_probs = self.actor(observation)
            with torch.no_grad():
                action_probs = action_probs.cpu().numpy().squeeze()
            observation, reward, done, info = self.env.step(action_probs)
            ep_reward += reward
            r = info['log_return']
            wealth=wealth*math.exp(r)
            CR.append(wealth)
            observation =  creator.create_obs(observation)
        return CR

    def backtest_DDPG(self):
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        observation, info = self.env.reset()
        observation = creator.create_obs(observation)
        done = False
        ep_reward = 0
        wealth = self.config.wealth
        # Collect culmulative return
        CR = []
        while not done:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.actor(observation).squeeze(0).cpu().detach().numpy()
            observation, reward, done, info = self.env.step(action)
            ep_reward += reward
            r = info['log_return']
            wealth=wealth*math.exp(r)
            CR.append(wealth)
            observation =  creator.create_obs(observation)
        return CR

    def backtest_QFPIS(self):
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        eps = 1e-8
        actions = []
        weights = []
        observation, info = self.env.reset()
        observation = creator.create_obs(observation)
        done = False
        ep_reward = 0
        wealth = self.config.wealth
        # Collect culmulative return
        CR = []
        while not done:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.actor(observation).squeeze(0).cpu().detach().numpy()
            # Here is the code for the policy gradient
            actions_prob = self.policy(observation)
            m = Categorical(actions_prob)
            # Selection action by sampling the action prob
            action_policy = m.sample()
            actions.append(action_policy.cpu().numpy())
            w1 = np.clip(action, 0, 1)  # np.array([cash_bias] + list(action))  # [w0, w1...]
            w1 /= (w1.sum() + eps)
            weights.append(w1)
            observation, reward,policy_reward, done, info = self.env.step(action,action_policy)
            r = info['log_return']
            wealth=wealth*math.exp(r)
            CR.append(wealth)
            ep_reward += reward
            observation = creator.create_obs(observation)
        return actions, weights, CR
    
    def backtest(self, model_type):
        backtest_func = getattr(self, f'backtest_{model_type}', None)
        if backtest_func is not None:
            res = backtest_func()
        return res





