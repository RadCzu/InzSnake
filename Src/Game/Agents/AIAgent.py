import math

from typing import List

import torch

from Src.AI.NeuralNetworks.SnakeBrain import SnakeBrain
from Src.Game.Agents.IAgent import IAgent
from Src.Game.Agents.Move_picking.PickBest import PickBest


class AIAgent(IAgent):
    def __init__(self, brain_network, move_memory=20, move_picking_strategy=None):
        super().__init__()
        self.brain_network: SnakeBrain = brain_network

        if move_picking_strategy is None:
            self.move_picking_strategy = PickBest()
        else:
            self.move_picking_strategy = move_picking_strategy

        self.move_memory = move_memory
        self.has_eaten = False
        self.move_history = []
        self.clear_history()
        self.previous_move = None

    def on_init(self, game):
        self.snake.death_observer.subscribe(self.clear_history)
        self.snake.score_observer.subscribe(self.on_eat)

    def on_eat(self):
        self.has_eaten = True

    def make_decision(self, game, get_map_size=None):
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

        return Q_values

    def add_to_history(self, move: List):
        previous_move = self.move_history[0]
        previous_move_vector = (previous_move[0], previous_move[1], previous_move[2], previous_move[3])
        previous_move_real = previous_move[4]

        are_move_dirs_the_same = (
                previous_move_vector[0] == move[0] and previous_move_vector[1] == move[1] and previous_move_vector[2] == move[2] and previous_move_vector[3] == move[3]
        )

        if are_move_dirs_the_same and previous_move_real == 1:
            updated_move = (
                previous_move[0],
                previous_move[1],
                previous_move[2],
                previous_move[3],
                previous_move[4] + 1,
                previous_move[5],
            )
            self.move_history[0] = updated_move
        else:
            self.move_history.pop(len(self.move_history) - 1)
            self.move_history.insert(0, (move[0], move[1], move[2], move[3], 1, 1))

    def on_game_over(self, game):
        return

    def get_game_data(self, game):
        # resized map
        resized = game.map.get_resized_map(64, 64)
        # shape: [1, 3, 64, 64]
        map_64_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # map size
        map_size_x, map_size_y = game.map.get_map_size()
        map_size_tensor = torch.tensor([map_size_x, map_size_y], dtype=torch.float32).unsqueeze(0)

        # snake position
        snake_x, snake_y = self.snake.head.get_position()

        # 15 x 15 fragment of the map
        fragment = game.map.get_map_fragment(snake_x - 7, snake_y - 7, snake_x + 7, snake_y + 7)

        snake_x_float = snake_x / map_size_x
        snake_y_float = snake_y / map_size_y
        snake_position_tensor = torch.tensor([snake_x_float, snake_y_float], dtype=torch.float32).unsqueeze(0)

        # 15x15 area around head
        normalized_fragment = game.map.get_normalized_fragment_3D(fragment)
        # shape: [1, 3, 15, 15]
        fragment_tensor = torch.tensor(normalized_fragment, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # distance from death
        left_death, right_death, up_death, down_death = self.snake.get_distance_from_deadly()
        death_tensor = torch.tensor([left_death / map_size_x, right_death / map_size_x, up_death / map_size_y, down_death / map_size_y], dtype=torch.float32).unsqueeze(0)

        # 5 closest food items
        # Get the closest food items (assuming game.get_n_closest_food_items returns a list of Food objects)
        _tracked_food_count = 5
        closest_food_objects = game.get_n_closest_food_items(_tracked_food_count, snake_x, snake_y)

        closest_food_positions = []
        for food in closest_food_objects:
            food_x, food_y = food.tile.get_position()
            closest_food_positions.append(((food_x - snake_x) / map_size_x, (food_y - snake_y) / map_size_y, 1))

        while len(closest_food_positions) < _tracked_food_count:
            closest_food_positions.append((0, 0, 0))

        closest_food_tensor = torch.tensor(closest_food_positions, dtype=torch.float32).unsqueeze(0)

        # move history
        history_tensor = torch.tensor(self.move_history, dtype=torch.float32).unsqueeze(0)
        history_tensor = history_tensor.view(history_tensor.size(0), -1)

        return map_64_tensor, fragment_tensor, death_tensor, closest_food_tensor, history_tensor

    def clear_history(self):
        self.move_history = []
        for i in range(self.move_memory):
            self.move_history.append((0, 0, 0, 0, 0, 0))
