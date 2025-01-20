from abc import ABC, abstractmethod

import numpy as np

from src.Agent.Move_picking.MovePickingStrategy import IMovePickingStrategy


class PickBest(IMovePickingStrategy):

    def pick_move(self, q_values):
        best_move_index = np.argmax(q_values).item()
        return best_move_index
