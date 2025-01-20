from abc import ABC, abstractmethod


class IMovePickingStrategy(ABC):

    @abstractmethod
    def pick_move(self, q_values):
        pass
