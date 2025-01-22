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
        super().__init__(brain_network, experience_manager=None, reward_decay=1., move_memory=20,
                         move_picking_strategy=None)

    def make_decision(self, game, get_map_size=None):
        self.apply_reward_decay()
        possible_moves = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        snake_x, snake_y = self.snake.head.get_tile().get_position()

        closest_food_objects = game.get_n_closest_food_items(4, snake_x, snake_y)
        food_coordinates = []
        for food_object in closest_food_objects:
            (food_x, food_y) = food_object.get_tile().get_position()
            food_coordinates.append((food_x, food_y))

        food_distances_1 = []
        for food_coordinate in food_coordinates:
            food_x, food_y = food_coordinate
            food_distance = math.sqrt((food_x - snake_x) ** 2 + (food_y - snake_y) ** 2)
            food_distances_1.append(food_distance)

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

        snake_x, snake_y = self.snake.head.get_tile().get_position()
        food_distances_2 = []
        for food_coordinate in food_coordinates:
            food_x, food_y = food_coordinate
            food_distance = math.sqrt((food_x - snake_x) ** 2 + (food_y - snake_y) ** 2)
            food_distances_2.append(food_distance)

        for i in range(len(food_distances_2)):
            food_distance_1 = food_distances_1[i]
            food_distance_2 = food_distances_2[i]
            proximity_reward = ((food_distance_1 - food_distance_2) / food_distance_1) * 2
            self.reward(proximity_reward)

        move_tensor = torch.tensor(possible_moves[action], dtype=torch.float32).unsqueeze(0)
        state = (map_64_tensor,
                 fragment_tensor,
                 death_tensor,
                 closest_food_tensor,
                 history_tensor,
                 move_tensor,
                 )

        self.state_hold = state
        self.action_hold = possible_moves[action]
        self.done_hold = done

        if game.over:
            self.snake.dead = True
            map_64_tensor, fragment_tensor, death_tensor, closest_food_tensor, history_tensor = self.get_game_data(game)
            move_dir = [0, 0, 0, 0]
            move_tensor = torch.tensor(move_dir, dtype=torch.float32).unsqueeze(0)
            state = (map_64_tensor,
                     fragment_tensor,
                     death_tensor,
                     closest_food_tensor,
                     history_tensor,
                     move_tensor,
                     )
            self.add_brain_experience(state, move_dir, self.cookies, True)
            self.cookies = 0
        return Q_values
