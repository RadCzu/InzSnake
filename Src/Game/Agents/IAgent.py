from abc import ABC, abstractmethod

from Src.Game.Game import Game


class IAgent(ABC):

    def __init__(self):
        self.snake = None

    def set_snake(self, snake):
        self.snake = snake

    @abstractmethod
    def get_cookies(self):
        pass

    @abstractmethod
    def make_decision(self, game: Game):
        pass

    @abstractmethod
    def validate(self, game: Game):
        pass

    @abstractmethod
    def on_game_over(self, game: Game):
        pass

    @abstractmethod
    def on_init(self, game: Game):
        pass