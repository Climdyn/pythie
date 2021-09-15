
import pickle
import numpy as np
from abc import ABC, abstractmethod


class PostProcessor(ABC):

    def __init__(self):

        self.parameters_list = list()

    @abstractmethod
    def train(self, observations, predictors, **kwargs):
        pass

    @abstractmethod
    def __call__(self, predictors):
        pass

    def load_from_file(self, filename, **kwargs):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f, **kwargs)
        f.close()

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

    def save_to_file(self, filename, **kwargs):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, **kwargs)
        f.close()

    @classmethod
    def _sanitize_data(cls, observations, predictors):
        # NaNify obs and predictors accordingly
        pred = predictors.copy()
        obs = observations.copy()

        # sanitize predictors first
        nan_positions = np.where(np.isnan(obs.data))

        for p in range(pred.number_of_predictors):
            for m in range(pred.number_of_members):
                idx = np.full(len(nan_positions[2]), m)
                idxp = np.full(len(nan_positions[0]), p)
                w = list(nan_positions)
                w[2] = idx
                w[0] = idxp
                pred.data[tuple(w)] = np.nan

        # then sanitize the observations
        p = np.sum(pred.data, axis=0)[np.newaxis, ...]
        p = np.sum(p, axis=2)[:, :, np.newaxis, ...]
        nan_positions = np.where(np.isnan(p))
        obs.data[nan_positions] = np.nan

        return obs, pred
