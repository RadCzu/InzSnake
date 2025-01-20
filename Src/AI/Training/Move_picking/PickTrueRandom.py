from abc import ABC, abstractmethod
from random import randint

import numpy as np

from src.Agent.Move_picking.MovePickingStrategy import IMovePickingStrategy


class PickTrueRandom(IMovePickingStrategy):

    def pick_move(self, q_values):
        best_move_index = randint(0, len(q_values) - 1)
        return best_move_index
