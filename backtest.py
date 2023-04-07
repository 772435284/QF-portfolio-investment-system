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
from models.DDPG import Actor
from models.QFPIS import Policy
from utils.utils import obs_normalizer
from environment.env import envs
from typing import Callable, List, cast, OrderedDict


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
        self.feature_num = len(market_feature)
        steps = config.max_step
        mode = config.mode


    def load_actor(self):
        actor = Actor(product_num=self.product_num,win_size=window_size).to(device)
        actor.load_state_dict(torch.load(path_join(config.ddpg_model_dir, AgentProps(config.agent_list[agent_index]).name +'_'+ str(config.episode-1))))
        return actor

    def load_policy():
        policy = Policy(product_num = product_num, win_size = window_size, action_size = 3).to(device)
        policy.load_state_dict(torch.load(path_join(config.pga_model_dir, AgentProps(config.agent_list[agent_index]).name +'_'+ str(config.episode-1))))
        return policy

    def backtest_ddpg(env, model):
        observation, info = env.reset()
        observation = obs_normalizer(observation)
        observation = observation.transpose(2, 0, 1)
        done = False
        ep_reward = 0
        wealth=10000

    def backtest_qf():
        pass







