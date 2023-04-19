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
from environment.selection_env import envs
from typing import Callable, List, cast, OrderedDict
from backtestor import backtestor
from models.OLMAR import OLMAR
from models.PAMR import PAMR
from models.CWMR import CWMR
from models.RMR import RMR

# Add more models here
MODEL_DICT = {
    "OLMAR": OLMAR,
    "PAMR": PAMR,
    "CWMR": CWMR,
    "RMR": RMR,
    # "OtherModel": OtherModelClass
}

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="CWMR", help='baseline_model_name')
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


results = [backtestor.backtest_selection(weights) for _ in range(config.backtest_number)]
CRs, SRs, MDDs, FPVs,CRRs, ARs, AVs = map(np.array, zip(*results))
print("\n")
print(f"Print final metric for {config.backtest_number} times backtest:" )
print("CR: ", CRs.mean(axis=0))
print("SR: ", SRs.mean())
print("MDD: ", MDDs.mean())
print("FPV: ", FPVs.mean())
print("CRR: ", CRRs.mean())
print("AR: ", ARs.mean())
print("AV: ", AVs.mean())

# 将结果存储为一个字典
results_dict = {
            'CR': CRs.mean(axis=0),
                'SR': SRs.mean(),
                'MDD': MDDs.mean(),
                'FPV': FPVs.mean(),
                'CRR': CRRs.mean(),
                'AR': ARs.mean(),
                'AV': AVs.mean()
                }

# 创建一个 DataFrame 对象
results_df = pd.DataFrame(results_dict.items(), columns=['metric', 'value'])
agent_index = cast(int, config.use_agents)
current_agent = AgentProps(config.agent_list[agent_index])
# 将 DataFrame 写入 csv 文件
results_df.to_csv(f'backtest_result/{current_agent.name}_{config.data_dir}_results.csv', index=False)