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


    def cal_KDJ(self, observation, n=9, m1=3, m2=3):
        stock_data = observation
        days, attributes, num_products = stock_data.shape
        kdj_values = np.zeros((days, 1, num_products))

        for product in range(num_products):
            highest_high = np.zeros(days)
            lowest_low = np.zeros(days)
            rsv = np.zeros(days)
            k = np.zeros(days)
            d = np.zeros(days)
            j = np.zeros(days)

            for i in range(days):
                start_idx = max(0, i - n + 1)
                high_values = stock_data[start_idx:i+1, 1, product]
                low_values = stock_data[start_idx:i+1, 2, product]
                close_value = stock_data[i, 3, product]
                
                highest_high[i] = np.max(high_values)
                lowest_low[i] = np.min(low_values)

                if highest_high[i] == lowest_low[i]:
                    rsv[i] = 0
                else:
                    rsv[i] = (close_value - lowest_low[i]) / (highest_high[i] - lowest_low[i]) * 100
                
                if i == 0:
                    k[i] = rsv[i]
                    d[i] = k[i]
                else:
                    k[i] = (1 - 1/m1) * k[i-1] + rsv[i] / m1
                    d[i] = (1 - 1/m2) * d[i-1] + k[i] / m2

                j[i] = 3 * k[i] - 2 * d[i]
                kdj_values[i, 0, product] = j[i]

        return kdj_values
    
    def cal_WILLR(self, observation, n=14):
        stock_data = observation
        days, attributes, num_products = stock_data.shape
        willr_values = np.zeros((days, 1, num_products))

        for product in range(num_products):
            highest_high = np.zeros(days)
            lowest_low = np.zeros(days)
            willr = np.zeros(days)

            for i in range(days):
                start_idx = max(0, i - n + 1)
                high_values = stock_data[start_idx:i+1, 1, product]
                low_values = stock_data[start_idx:i+1, 2, product]
                close_value = stock_data[i, 3, product]

                highest_high[i] = np.max(high_values)
                lowest_low[i] = np.min(low_values)

                if highest_high[i] == lowest_low[i]:
                    willr[i] = 0
                else:
                    willr[i] = (highest_high[i] - close_value) / (highest_high[i] - lowest_low[i]) * -100

                willr_values[i, 0, product] = willr[i]

        return willr_values

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