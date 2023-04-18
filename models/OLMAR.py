import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
from torch.autograd import Variable
import pandas as pd
import numpy as np
from tools.ddpg.replay_buffer import ReplayBuffer
from api_types import GlobalConfig, AgentProps
from data_provider.data_factory import data_provider
from typing import Callable, List, cast, OrderedDict
from os.path import join as path_join
from torch.distributions import Categorical
from utils.utils import hidden_init
from tensorboardX import SummaryWriter
from observation.obs_creator import obs_creator


class OLMAR:
    config: GlobalConfig
    actor_noise: Callable
    summary_path: str = path_join('train_results', 'ddpg')
    current_agent: AgentProps
    price_history: pd.DataFrame
    trading_dates: List[str]
    window_size: int
    use_cuda: bool

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.product_list=AgentProps(config.agent_list[config.use_agents]).product_list
        self.dataprocessor = data_provider(config)
        self.dataprocessor.data_for_selection()
        self.data = self.dataprocessor._data
    
    def train(self, window=5, epsilon=10):
        prices = self.data
        n_days, _, n_assets = prices.shape
        close_prices = prices[:, 3, :]

        # 初始化权重矩阵
        weights = np.zeros((n_days, n_assets))

        # 初始化资产权重
        weights[0, :] = 1 / n_assets

        for t in range(1, n_days):
            # 计算移动平均
            moving_average = np.mean(close_prices[max(0, t - window):t, :], axis=0)

            # 计算 x_t 和 x_tilde
            x_t = close_prices[t - 1, :] / moving_average
            x_tilde = x_t / np.sum(x_t)

            # 更新权重
            weights[t, :] = weights[t - 1, :] * (1 + epsilon * (x_tilde - x_t))
            weights[t, :] /= np.sum(weights[t, :])

        return weights

    

