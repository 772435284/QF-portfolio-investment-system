'''
@Author: Yitao Qiu
'''
import numpy as np
import pandas as pd
import matplotlib as plt
import datetime
import gym
import gym.spaces

eps = 1e-8

# A class that is to perform the calculation of the portfolio
class Portfolio(object):

    def __init__(self, steps, trading_cost, mode):

        self.steps = steps
        self.cost = trading_cost
        self.mode = mode

    def _step(self, w1, y1, reset):

        assert w1.shape == y1.shape, 'w1 and y1 must have same number of products'
        assert y1[0] == 1.0, 'y1[0] should be 1'

        w0 = self.w0
        p0 = self.p0
        y0 = self.y0
        dw1 = (y0 * w0) / (np.dot(y0, w0) + eps) 
        
        if self.mode == "Test" and reset == 1:
            mu1 = self.cost * (np.abs(w1[1:])).sum()
        else:
            mu1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum()
        
        p1 = p0 * (1 - mu1) * np.dot(y1, w1)   

        rho1 = p1 / p0 - 1  
        r1 = np.log((p1 + eps) / (p0 + eps))  
        reward = r1 / self.steps * 1000.  
        
        self.w0 = w1
        self.p0 = p1
        self.y0 = y1

        # Run out of money, done
        done = p1 == 0

        info = {
            "portfolio_value": p1,
            "rate_of_return": rho1,
            "log_return": r1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.w0 = np.array([1.0] + [0.0] * 9)
        self.infos = []
        self.p0 = 1.0
        self.y0 = np.zeros((10,), dtype=float)
        self.y0[0] = 1