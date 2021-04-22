
import pickle
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
