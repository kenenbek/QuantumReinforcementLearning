from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, next_state, reward):
        pass