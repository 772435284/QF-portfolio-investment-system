# QF-portfolio-investment-system
Quantum Finance-based 
Hybrid Deep Reinforcement Learning 
Portfolio Investment System

## Gudience
+ DDPG.ipynb: Run this notebook to train and test DDPG-based portfolio investment system
+ QFPIS-1.ipynb: Run this notebook to train and test Quantum finance portfolio investment system using one Quantum price level(QPL)
+ QFPIS-2.ipynb: Run this notebook to train and test Quantum finance portfolio investment system using two QPL
+ backtest.ipynb: Run this notebook to do back test
+ ./data: Contain the training and testing forex data, which are obtained form *MetaTrader4*
+ ./environment: Contain the reinforcement learning enviroment:1)QF_env: enviroment for DDPG 2）QF_env_1: environment for QFPIS-1 3) environment for QFPIS-2
+ ./model: Contain the trained models
+ ./config/config.json: configure the training settings:
```
{
  "episode": 100,
  "max step": 1000,
  "buffer size": 100000,
  "batch size": 64,
  "tau": 0.001,
  "gamma": 0.99,
  "actor learning rate": 0.0001,
  "critic learning rate": 0.001,
  "policy learning rate": 0.0001
}
```
## Requirement
+ Python 3.7
+ Jupyter notebook
+ Pytorch
+ numpy
+ pandas
+ matplotlib
+ detailed requirement will be upadted ...

## Contact
+ Yitao Qiu m730026088@mail.uic.edu.cn
+ Rongkai Liu m730026073@mail.uic.edu.cn
