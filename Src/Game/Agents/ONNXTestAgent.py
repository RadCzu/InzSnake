import torch
import numpy as np
from typing import List
from Src.AI.NeuralNetworks.SnakeBrain import SnakeBrain
from Src.Game.Agents.AIAgent import AIAgent
from Src.Game.Agents.IAgent import IAgent
from Src.Game.Agents.Move_picking.PickBest import PickBest
from Src.AI.NeuralNetworks.ONNXBrainAdapter import ONNXBrainAdapter

class ONNXTestAgent(AIAgent):
    def __init__(self, brain_network, onnx_model_path, move_memory=20, move_picking_strategy=None):
        super().__init__(brain_network, move_memory, move_picking_strategy)
        self.onnx_brain = ONNXBrainAdapter(onnx_model_path)  # ONNX model adapter

    def make_decision(self, game, get_map_size=None):
        possible_moves = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        # Gather game data (same data for both models)
        map_64_tensor, fragment_tensor, death_tensor, closest_food_tensor, history_tensor = self.get_game_data(game)

        Q_values_regular = []
        Q_values_onnx = []

        for i in range(len(possible_moves)):
            move_dir = possible_moves[i]
            move_tensor = torch.tensor(move_dir, dtype=torch.float32).unsqueeze(0)

            # Get Q-values from the regular brain network
            q_value_regular = self.brain_network.run(
                map_summary=map_64_tensor,
                local_map=fragment_tensor,
                distance_from_walls=death_tensor,
                food_info=closest_food_tensor,
                move_history=history_tensor,
                move_direction=move_tensor,
            )
            Q_values_regular.append(q_value_regular)

            # Get Q-values from the ONNX brain network (test comparison)
            text = "a"
            q_value_onnx = self.onnx_brain.run(
                map_summary=map_64_tensor.numpy(),  # Convert to numpy array for ONNX
                local_map=fragment_tensor.numpy(),
                distance_from_walls=death_tensor.numpy(),
                food_info=closest_food_tensor.numpy(),
                move_history=history_tensor.numpy(),
                move_direction=move_tensor.numpy()
            )
            Q_values_onnx.append(q_value_onnx)

            # Print comparison of Q-values
            print(f"Move {i}:")
            print(f"  Regular Model Q-value: {q_value_regular}")
            print(f"  ONNX Model Q-value: {q_value_onnx}")
            print("-" * 40)

        # Compare Q-values between the regular model and ONNX model
        match = all(np.isclose(Q_values_regular, Q_values_onnx))
        if not match:
            print("Warning: Q-values do not match between the regular model and ONNX model!")
        else:
            print("Success: Q-values match between the regular model and ONNX model.")

        # Get the index of the move with the maximum Q-value (based on regular model)
        best_move_index = self.move_picking_strategy.pick_move(Q_values_regular)
        action = best_move_index

        # Select the best move direction
        best_move = possible_moves[best_move_index]
        self.add_to_history(best_move)

        game.input(action, self.snake)

        snake_x, snake_y = self.snake.head.get_tile().get_position()

        return Q_values_regular  # Return Q-values for further inspection
