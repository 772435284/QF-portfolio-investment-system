import numpy as np

eps = 1e-8

def sharpe(returns, freq=252, rfr=0):
    # The function that is used to caculate sharpe ratio
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)

def annualized_sharpe_ratio(returns):
    # 计算无风险利率，这里假设为0
    rf = 0.0
    
    # 计算每日收益率的平均值和标准差
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    # 计算年化收益率和年化波动率
    annual_return = (1 + mean_return) ** 252 - 1
    annual_volatility = std_return * np.sqrt(252)
    
    # 计算年化夏普比率
    sharpe_ratio = (annual_return - rf) / annual_volatility
    
    return sharpe_ratio

def max_drawdown(return_list):
    # The function that is used to calculate the max drawndom
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i]) 
    return (return_list[j] - return_list[i]) / (return_list[j])

def annualized_return(daily_returns):
    total_return = np.prod(1 + daily_returns) - 1
    years = len(daily_returns) / 252   # 假设一年有252个交易日
    annualized_return = (1 + total_return) ** (1 / years) - 1
    return annualized_return * 100

def annual_volatility(daily_returns):
    annual_vol = np.std(daily_returns) * np.sqrt(252)
    return annual_vol * 100