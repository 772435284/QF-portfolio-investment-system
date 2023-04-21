'''
@Author: Yitao Qiu
'''
import numpy as np

eps = 1e-8

# A class that is to perform the calculation of the portfolio
class Portfolio(object):

    def __init__(self, steps, trading_cost, mode,num_assets,time_cost=0.0):
        #self.asset_names = asset_names
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.mode = mode
        self.num_assets = num_assets

    def backtest_mu(self,dw1,w1):
        # Update the function for mu according to the original paper
        mu0 = 1
        mu1 = 1 - 2*self.cost + self.cost ** 2
        while abs(mu1-mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - self.cost * dw1[0] -(2 * self.cost - self.cost ** 2) *np.sum(np.maximum(dw1[1:] - mu1*w1[1:], 0))) / (1 - self.cost * w1[0]) # (eq14)
        return mu1
        

    def _step(self, w1, y1, reset):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        w0 = self.w0
        p0 = self.p0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        # if self.mode == "Test" and reset == 1:
        #     mu1 = self.cost * (np.abs(w1[1:])).sum()
        # else:
        if self.mode == "Train":
            mu1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum()
        else:
            mu1 = self.backtest_mu(dw1,w1)
            mu1 = 1-(mu1)
            
        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return
        reward = r1 / self.steps * 1000.  # (22) average logarithmic accumulated return
        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "rate_of_return": rho1,
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done


    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * self.num_assets)
        self.p0 = 1.0