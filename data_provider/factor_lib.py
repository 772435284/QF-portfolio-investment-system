import numpy as np
from  data_provider.normalizer import normalizer

class factor_lib(object):
    def __init__(self ,factor, norm_method,norm_type):
        self.factor = factor
        self.norm_type = norm_type
        self.norm = normalizer(norm_method)


    def cal_price_return(self, observation):
        pr = observation[:, 3, :] / observation[:, 0, :]
        pr = pr.reshape(pr.shape[0], 1, pr.shape[1])
        return pr

    def ema(self, arr, window):
        alpha = 2 / (window + 1)
        result = np.zeros_like(arr)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    def cal_MACD(self,observation):
        close_data = observation[:, 3, :]  # 提取收盘价数据

        ema12 = self.ema(close_data, 12)
        ema26 = self.ema(close_data, 26)

        dif = ema12 - ema26
        dea = self.ema(dif, 9)

        macd_values = 2 * (dif - dea)

        # 调整输出格式为(1637, 1, 9)
        macd_values = np.expand_dims(macd_values, axis=1)

        return macd_values

    def cal_RSI(self,observation, window=14):
        close_data = observation[:, 3, :]  # 提取收盘价数据

        delta = np.diff(close_data, axis=0)  # 计算价格变化
        gains = np.where(delta > 0, delta, 0)  # 正向价格变化
        losses = np.where(delta < 0, -delta, 0)  # 负向价格变化（取正值）

        # 计算平均收益和损失的EMA
        avg_gains = self.ema(gains, window)
        avg_losses = self.ema(losses, window)

        # 避免除以零的情况
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gains / avg_losses  # 计算相对强度（RS）
            rsi_values = 100 - (100 / (1 + rs))  # 计算RSI

        # 处理初始值和边界情况
        rsi_values = np.insert(rsi_values, 0, np.nan, axis=0)
        rsi_values[np.isinf(rsi_values)] = 100
        rsi_values[np.isnan(rsi_values)] = 0

        # 调整输出格式为(1637, 1, 9)
        rsi_values = np.expand_dims(rsi_values, axis=1)

        return rsi_values

    def create_factor(self, observation):
        factors = [observation]
        for f in self.factor:
            cal_func = getattr(self, f'cal_{f}', None)
            if cal_func is not None:
                factor = cal_func(observation)
                factors.append(factor)
        factors = np.concatenate(factors, axis=1)
        # Do global normalization
        if self.norm_type == "global":
            for i in range(factors.shape[1]):
                for j in range(factors.shape[2]):
                    factors[:, i, j] = self.norm.normalize(factors[:, i, j].reshape(-1, 1)).flatten()
        return factors