from abc import ABC, abstractmethod
from random import random

import numpy as np

from Src.Game.Agents.Move_picking.MovePickingStrategy import IMovePickingStrategy


class PickWeightedRandom(IMovePickingStrategy):

    def pick_move(self, q_values):
        min_value = np.min(q_values)
        max_value = np.max(q_values)

        temp_q_values = np.copy(q_values)

        if min_value < 0:
            # if there are any positive values, select them over the negative ones
            # if max_value > 0:
            #     for i in range(0, len(temp_q_values)):
            #         if temp_q_values[i] < 0:
            #             temp_q_values[i] = 0
            # else:
            temp_q_values += abs(min_value)

        ans_sum = np.sum(temp_q_values)

        if ans_sum == 0:
            return np.argmax(temp_q_values)

        chances = temp_q_values / ans_sum
        choice = random()
        best_move_index = 0
        cumulated_chance = 0.0

        for i in range(len(chances)):
            cumulated_chance += chances[i]
            if choice < cumulated_chance:
                best_move_index = i
                break

        return best_move_index

class PickScaledRandom(IMovePickingStrategy):

    def pick_move(self, q_values):
        min_value = np.min(q_values)
        temp_q_values = np.copy(q_values)

        if min_value > 0:
            temp_q_values -= abs(min_value)
        else:
            temp_q_values += abs(min_value)

        ans_sum = np.sum(temp_q_values)

        if ans_sum == 0:
            return np.argmax(temp_q_values)

        chances = temp_q_values / ans_sum
        choice = random()
        best_move_index = 0
        cumulated_chance = 0.0

        for i in range(len(chances)):
            cumulated_chance += chances[i]
            if choice < cumulated_chance:
                best_move_index = i
                break

        return best_move_index