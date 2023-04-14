import numpy as np
from  data_provider.normalizer import normalizer

class obs_creator(object):
    def __init__(self, norm_method,norm_type):
        self.norm_type = norm_type
        self.norm_method = norm_method
        self.norm = normalizer(norm_method)

    

    def create_obs(self,observation):
        # Only use the factors after OHLC
        obs = observation[:,:,4:]
        if self.norm_type == 'rolling':
            for i in range(observation.shape[0]):
                for j in range(observation.shape[2]):
                    observation[i, :, j] = self.norm.normalize(observation[i, :, j].reshape(-1, 1)).flatten()
        obs = obs.transpose(2, 0, 1)
        return obs
        
        
