import numpy as np

class normalizer(object):
    def __init__(self ,norm_method):
        self.norm_method = norm_method
        
    # Normalization method
    def min_max_normalization(self, data):
        min_values = np.min(data, axis=0, keepdims=True)
        max_values = np.max(data, axis=0, keepdims=True)

        normalized_data = (data - min_values) / (max_values - min_values)

        return normalized_data

    def z_score_normalization(self, data):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_data = (data - mean) / std

        normalized_data[np.isnan(normalized_data)] = 0

        return normalized_data

    def log_normalization(self, data, C=1):
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_data = np.log(data + C)

        normalized_data[np.isinf(normalized_data)] = 0
        normalized_data[np.isnan(normalized_data)] = 0

        return normalized_data

    def normalize(self, data):
        norm_func = getattr(self, f'{self.norm_method}_normalization', None)
        if norm_func is not None:
            normalized_data = norm_func(data)
        else:
            normalized_data = data

        return normalized_data
    