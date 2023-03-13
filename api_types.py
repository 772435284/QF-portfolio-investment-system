from typing import List, Dict, cast, Callable, Union
import pandas as pd
from utils.utils import DictLikeObject
from datetime import datetime
from torch import nn
import numpy as np

# 通过API TYPES 将dict数据转换为标准化的类
# 并且能够识别出相应的数据类型 在VS code提供高亮


class AgentProps(DictLikeObject):
	name: str
	product_list: List[str]


class GlobalConfig(DictLikeObject):
	# 通用参数
	mode: str
	use_cuda: bool
	ddpg_model_dir: str
	pg_model_dir: str
	pga_model_dir: str
	agent_list: List[AgentProps]
	use_agents: Union[List[int], int]
	market_feature:List[str]

	# TODO: DELETE ME
	model_name: None
	product_list: None

	# 模型训练参数
	gm_train_strategy_id: str
	window_size: int
	episode: int
	max_step: int
	buffer_size: int
	batch_size: int
	tau: float
	gamma: float
	actor_learning_rate: float
	critic_learning_rate: float
	policy_learning_rate: float
	train_end_date: str
	train_intermediate_dir: str

	# 回测核心参数
	gm_strategy_id: str
	gm_token: str
	backtest_start_date: str
	backtest_end_date: str
	backtest_commission_ratio: float
	backtest_slippage_ratio: float
	agent_cash_available: float
	qpl_profit_level: int
	qpl_loss_level: int
	use_qpl: bool
	use_pga: bool
	nqpr_window_size: int
	random_seed: int
	use_dropout: Union[str, bool]
	dropout_rate: float
	