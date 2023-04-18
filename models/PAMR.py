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


class PAMR:
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
    
    def train(self, epsilon=0.5):
        # 初始化
        reshaped_data = self.data
        num_days, _, num_assets = reshaped_data.shape
        portfolio_weights = np.zeros((num_days, num_assets))
        portfolio_weights[0] = np.ones(num_assets) / num_assets

        # 训练
        for t in range(1, num_days):
            # 计算价格回报
            daily_returns = reshaped_data[t, 3] / reshaped_data[t - 1, 3] - 1

            # 更新投资组合权重
            loss = np.dot(portfolio_weights[t - 1], daily_returns)
            denom = np.linalg.norm(daily_returns)**2
            if denom != 0:
                tau = max(0, min(epsilon, loss / denom))
            else:
                tau = 0
            new_weights = portfolio_weights[t - 1] - tau * daily_returns

            # 裁剪权重以确保它们在有效范围内
            new_weights = np.clip(new_weights, 0, 1)
            portfolio_weights[t] = new_weights / np.sum(new_weights)

        return portfolio_weights
