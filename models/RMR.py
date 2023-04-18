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


class RMR:
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
    
    def compute_weights(self, prices, window_size=30):
        price_relatives = np.divide(prices[1:], prices[:-1])
        medians = np.median(price_relatives[-window_size:], axis=0)
        deviations = medians / price_relatives[-1]
        weights = 1 / deviations
        weights /= np.sum(weights)
        return weights

    def train(self, window_size=30):
        n_days, _, n_assets = self.data.shape
        close_prices = self.data[:, 3, :]
        weights_history = np.zeros((n_days - 1, n_assets))

        for t in range(window_size, n_days - 1):
            weights = self.compute_weights(close_prices[:t], window_size)
            weights_history[t] = weights

        return weights_history
        
