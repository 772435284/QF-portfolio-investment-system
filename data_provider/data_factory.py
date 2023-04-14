import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from api_types import GlobalConfig, AgentProps
from data_provider.data_loader  import Dataset_Custom

# Select 
data_dict = {
    'custom': Dataset_Custom,
}

def data_provider(config: GlobalConfig):
    Data = data_dict[config.dataset]

    if config.mode == 'Train':
        start_date = None
    elif config.mode == 'Val':
        start_date = config.train_start_date
    elif config.mode == 'Test':
        start_date = config.train_start_date
    elif config.mode == 'Backtest':
        pass

    data_loader = Data(product_list=AgentProps(config.agent_list[config.use_agents]).product_list,
                       market_feature=config.market_feature,
                       feature_num=len(config.market_feature),
                       steps=config.max_step,
                       window_length=config.window_size,
                       mode=config.mode,
                       train_ratio=config.train_ratio,
                       val_ratio= config.val_ratio,
                       factor = config.factor,
                       norm_method = config.norm_method,
                       norm_type = config.norm_type,
                       data_dir = config.data_dir,
                       start_date= start_date
                       )
    return data_loader

