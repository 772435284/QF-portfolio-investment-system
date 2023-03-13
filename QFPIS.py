from typing import cast, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import json
import yaml
from api_types import GlobalConfig, AgentProps
from torch.autograd import Variable
from torch.distributions import Categorical
from utils.qf_data import normalize,load_observations
from environment.QF_env_1 import envs
from tools.ddpg.replay_buffer import ReplayBuffer
from tools.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from tensorboardX import SummaryWriter
from models.QFPIS import DDPG

with open('stable_config.yml', 'r', encoding='utf-8') as f:
		config = GlobalConfig(yaml.safe_load(f))
assert type(config.use_agents) == int, 'You must specify one agent for training!'
agent_index = cast(int, config.use_agents)
product_list = AgentProps(config.agent_list[agent_index]).product_list
product_num = len(product_list)
window_size = config.window_size
market_feature = config.market_feature
feature_num = len(market_feature)
steps = config.max_step
mode = config.mode

action_dim = [product_num+1]
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

env = envs(product_list,market_feature,feature_num,steps,window_size,mode)
model = DDPG(env=env,window_size=window_size,actor_noise=actor_noise,config=config)
model.train()

