import os
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
from utils.utils import normalize,load_observations
from tools.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.QF_env import envs
from typing import Callable, List, cast, OrderedDict
from backtestor import backtestor


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="QFPIS", help='baseline_model_name')
args = parser.parse_args()

config_file_name = f'config/{args.model}.yml'

with open(config_file_name, 'r', encoding='utf-8') as f:
    config = GlobalConfig(yaml.safe_load(f))
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
env = envs(config)
backtestor = backtestor(env, OrnsteinUhlenbeckActionNoise, device, config)

backtestor.load_actor(args.model,isbaseline=False)
backtestor.load_policy(action_size=config.qpl_level+1)

results = [backtestor.backtest(args.model) for _ in range(config.backtest_number)]
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
                'AR': ARs.mean(),
                'CRR': CRRs.mean(),
                'SR': SRs.mean(),
                'AV': AVs.mean(),
                'MDD': MDDs.mean(),
                'FPV': FPVs.mean(),
                }

# 创建一个 DataFrame 对象
results_df = pd.DataFrame(results_dict.items(), columns=['metric', 'value'])
agent_index = cast(int, config.use_agents)
current_agent = AgentProps(config.agent_list[agent_index])
# 将 DataFrame 写入 csv 文件
results_df.to_csv(f'backtest_result/{current_agent.name}_QPL_{config.qpl_level}_{config.data_dir}_results.csv', index=False)



    