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


class CWMR:
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
    
    def train(self, window_size=30):
        ohlcv_data = self.data
        n_days, _, n_assets = ohlcv_data.shape
        weight_history = np.zeros((n_days, n_assets))
        weights = np.ones(n_assets) / n_assets

        for t in range(window_size + 1, n_days):
            historical_returns = ohlcv_data[t - window_size:t, 3, :] / ohlcv_data[t - window_size - 1:t - 1, 3, :] - 1
            confidence_weights = 1 / np.std(historical_returns, axis=0)
            confidence_weights = confidence_weights / np.sum(confidence_weights)

            # 更新权重
            weights = confidence_weights
            weight_history[t] = weights

        # 为权重数组的前 window_size + 1 天填充均匀权重
        #weight_history[:window_size + 1] = np.ones(n_assets) / n_assets
        return weight_history
        
