import argparse
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
from environment.env import envs
from typing import Callable, List, cast, OrderedDict
from backtestor import backtestor
from models.OLMAR import OLMAR

# Add more models here
MODEL_DICT = {
    "OLMAR": OLMAR
    # "OtherModel": OtherModelClass
}

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="OLMAR", help='baseline_model_name')
args = parser.parse_args()

# Load configuration
# Generate config file name based on selected model
config_file_name = f'config/{args.model}.yml'

# Load configuration
with open(config_file_name, 'r', encoding='utf-8') as f:
    config = GlobalConfig(yaml.safe_load(f))
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Create model based on command line argument
model_class = MODEL_DICT.get(args.model)
if model_class is None:
    raise ValueError(f"Unsupported model: {args.model}")

model = model_class(config=config)
weights = model.train()
remaining_fund = 1.0 - np.sum(weights, axis=1)

# 在第一列插入剩余资金比例的列
weights = np.insert(weights, 0, remaining_fund, axis=1)
env = envs(config)
backtestor = backtestor(env, OrnsteinUhlenbeckActionNoise, device, config)

CR = backtestor.backtest_selection(weights)
print(CR)