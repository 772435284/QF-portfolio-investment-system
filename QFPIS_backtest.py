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
from utils.utils import normalize,load_observations
from tools.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.QF_env import envs
from typing import Callable, List, cast, OrderedDict
from backtestor import backtestor

with open('config/QFPIS.yml', 'r', encoding='utf-8') as f:
    config = GlobalConfig(yaml.safe_load(f))
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
env = envs(config)
backtestor = backtestor(env, OrnsteinUhlenbeckActionNoise, device, config)

backtestor.load_actor("QFPIS",isbaseline=False)
backtestor.load_policy(action_size=config.qpl_level+1)

results = [backtestor.backtest("QFPIS") for _ in range(config.backtest_number)]
CRs, SRs, MDDs, FPVs = map(np.array, zip(*results))
print("CR: ", CRs.mean(axis=0))
print("SR: ", SRs.mean())
print("MDD: ", MDDs.mean())
print("FPV: ", FPVs.mean())

    