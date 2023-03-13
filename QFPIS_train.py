from typing import cast
import numpy as np
import torch
import random
import yaml
from api_types import GlobalConfig, AgentProps
from environment.QF_env_1 import envs
from tools.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from models.QFPIS import QFPIS

# Load configuration
with open('config/stable_config.yml', 'r', encoding='utf-8') as f:
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

# Set random seed 
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# Start training
env = envs(product_list,market_feature,feature_num,steps,window_size,mode)
model = QFPIS(env=env,window_size=window_size,actor_noise=actor_noise,config=config)
model.train()
