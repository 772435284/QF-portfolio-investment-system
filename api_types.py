from typing import List, Dict, cast, Callable, Union
from utils.utils import DictLikeObject
from datetime import datetime

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
	factor: List[str]
	market_feature:List[str]

	# TODO: DELETE ME
	model_name: str
	product_list: None

	# 模型训练参数
	gm_train_strategy_id: str
	use_qpl: bool
	qpl_level: int
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
	train_start_date: str
	train_end_date: str
	train_ratio: float
	val_ratio: float
	date_format: str
	train_intermediate_dir: str
	is_summary: bool
	summary_path: str
	verbose: int
	dataset: str
	norm_method: str
	norm_type: str

	# 模拟回测参数
	wealth: float

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

	