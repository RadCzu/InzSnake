from abc import ABC, abstractmethod

import numpy as np

from Src.Game.Agents.Move_picking.MovePickingStrategy import IMovePickingStrategy
from Src.Game.Agents.Move_picking.PickBest import PickBest
from Src.Game.Agents.Move_picking.PickRandom import PickWeightedRandom, PickScaledRandom


class PickDynamic(IMovePickingStrategy):

    def pick_move(self, q_values):
        worst_move = q_values[np.argmin(q_values)]
        temp_q_values = np.copy(q_values)

        if worst_move < 0:
            temp_q_values += abs(worst_move)
        else:
            temp_q_values -= abs(worst_move)

        temp_temp = temp_q_values.copy()
        sorted_indices = np.argsort(temp_temp.squeeze())[::-1]
        second_best_move = temp_temp[sorted_indices[1]]
        third_best_move = temp_temp[sorted_indices[2]]

        best_move = temp_q_values[np.argmax(temp_q_values)]

        if best_move > 1.5 * second_best_move or best_move > 3.0 * third_best_move:
            picker = PickBest()
            return picker.pick_move(q_values)
        else:
            picker = PickScaledRandom()
            return picker.pick_move(q_values)

