import math

from typing import List

import torch

from Src.AI.NeuralNetworks.SnakeBrain import SnakeBrain
from Src.Game.Agents.AIAgent import AIAgent
from Src.AI.Training.LearningExperience import LearningExperience
from Src.Game.Agents.Move_picking.PickBest import PickBest
from Src.AI.Training.Reward import Reward
from Src.Game.Agents.TrainingAIAgent import TrainingAIAgent
from Src.Game.Interfaces.Observer import Observer


class TrainingAIAgentNoFood(TrainingAIAgent):

    def __init__(self, brain_network, experience_manager=None, reward_decay=1., move_memory=20,
                 move_picking_strategy=None):
        super().__init__(brain_network, experience_manager=experience_manager, reward_decay=reward_decay, move_memory=move_memory,
                         move_picking_strategy=move_picking_strategy)

    def make_decision(self, game, get_map_size=None):
        self.has_eaten = False
        self.apply_reward_decay()
        possible_moves = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        map_64_tensor, fragment_tensor, death_tensor, closest_food_tensor, history_tensor = self.get_game_data(game)

        Q_values = []
        done = False

        for i in range(len(possible_moves)):
            move_dir = possible_moves[i]
            move_tensor = torch.tensor(move_dir, dtype=torch.float32).unsqueeze(0)
            q_value = self.brain_network.run(
                map_summary=map_64_tensor,
                local_map=fragment_tensor,
                distance_from_walls=death_tensor,
                food_info=closest_food_tensor,
                move_history=history_tensor,
                move_direction=move_tensor,
            )
            Q_values.append(q_value)
            # evaluate for each possible move

        # Get the index of the move with the maximum Q-value
        best_move_index = self.move_picking_strategy.pick_move(Q_values)
        action = best_move_index

        # Select the best move direction
        best_move = possible_moves[best_move_index]
        self.add_to_history(best_move)

        game.input(action, self.snake)

        move_tensor = torch.tensor(possible_moves[action], dtype=torch.float32).unsqueeze(0)
        state = (map_64_tensor,
                 fragment_tensor,
                 death_tensor,
                 closest_food_tensor,
                 history_tensor,
                 move_tensor,
                 )

        self.state_hold = state
        self.done_hold = done

        return Q_values

    def is_done(self):
        previous = self.done_hold
        self.done_hold = self.snake.dead
        return previous
