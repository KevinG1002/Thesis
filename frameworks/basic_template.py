from abc import ABC, abstractmethod


class BasicLearning(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def test(self):
        ...
