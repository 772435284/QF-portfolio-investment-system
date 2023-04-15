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
from environment.env import envs
from typing import Callable, List, cast, OrderedDict
from backtestor import backtestor

with open('config/ddpg.yml', 'r', encoding='utf-8') as f:
    config = GlobalConfig(yaml.safe_load(f))
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
env = envs(config)
backtestor = backtestor(env, OrnsteinUhlenbeckActionNoise, device, config)

backtestor.load_actor()

CR = backtestor.backtest_DDPG(backtestor.actor)
print(CR)