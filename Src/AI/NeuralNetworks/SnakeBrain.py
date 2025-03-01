import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeBrain(nn.Module):
    def __init__(self):
        super(SnakeBrain, self).__init__()

        # Input: direction (2D vector)
        self.snake_direction = nn.Linear(4, 8)  # Reduce 2D direction vector to 8 features
        self.direction_2_local_embed = nn.Linear(4, 225)  # Embed direction into 15x15 grid (15x15 = 225)
        self.direction_2_map_embed = nn.Linear(4, 4096)  # Embed direction into 64x64 grid (64x64 = 4096)

        # Input: Map Summary (64x64x3)
        self.map_conv2d_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)  # Output: 64x64x32
        self.map_conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x64x64
        self.map_pool = nn.MaxPool2d(2, 2)  # Downsample to 32x32

        # Input: 15x15 Localized Map (15x15x3)
        self.local_conv2d_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)  # Output: 15x15x32
        self.local_conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 15x15x64
        self.local_pool = nn.MaxPool2d(2, 2)  # Downsample to 7x7

        # Input: Distance from Deadly Walls (4 values)
        self.wall_distance = nn.Linear(4 + 4, 16)  # Output: 16-dimensional vector

        # Input: 5 Nearest Food Items + Flag (6 values)
        self.closest_food = nn.Linear(5 * 3 + 4, 16)  # Output: 16-dimensional vector

        # Input: Last 20 Moves (20 moves, 3 values each)
        self.move_history_1 = nn.Linear(6 * 20 + 4, 256)
        self.move_history_2 = nn.Linear(256, 64)

        # Fully connected layers for combining all features
        self.combined_1 = nn.Linear(65536 + 3136 + 16 + 16 + 64 + 8, 256)  # 68776

        self.combined_2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)  # Output: 1 Q-value per action (single value)

    def forward(self, map_summary, local_map, distance_from_walls, food_info, move_history, move_direction):

        # Use Leaky ReLU with a negative slope for all activations
        direction_embedding = F.relu(self.direction_2_local_embed(move_direction))
        direction_embedding = direction_embedding.view(-1, 1, 15, 15)

        map_embedding = F.relu(self.direction_2_map_embed(move_direction))
        map_embedding = map_embedding.view(-1, 1, 64, 64)

        # Process Map Summary (64x64x3)
        x_map = torch.cat([map_summary, map_embedding], dim=1)
        x_map = F.leaky_relu(self.map_conv2d_1(x_map), negative_slope=0.01)
        x_map = F.leaky_relu(self.map_conv2d_2(x_map), negative_slope=0.01)
        x_map = self.map_pool(x_map)

        # Process Local Map (15x15x3)
        x_local_map = torch.cat([local_map, direction_embedding], dim=1)
        x_local_map = F.leaky_relu(self.local_conv2d_1(x_local_map), negative_slope=0.01)
        x_local_map = F.leaky_relu(self.local_conv2d_2(x_local_map), negative_slope=0.01)
        x_local_map = self.local_pool(x_local_map)

        # Process Distance from Walls (4 values)
        x_distance = torch.cat([distance_from_walls, move_direction], dim=1)
        x_distance = F.relu(self.wall_distance(x_distance))

        # Process Nearest Food Items (3x5 array)
        x_food = food_info.view(food_info.size(0), -1)
        x_food = torch.cat([x_food, move_direction], dim=1)
        expected_input_dim = self.closest_food.in_features  # Typically 19
        if x_food.shape[1] != expected_input_dim:
            print(f"Shape mismatch detected! x_food.shape: {x_food.shape}, expected: (N, {expected_input_dim})")
            print(f"food_info.shape: {food_info.shape}, move_direction.shape: {move_direction.shape}")

        x_food = F.leaky_relu(self.closest_food(x_food), negative_slope=0.01)

        # Process Move History (20 moves × 4 values each)
        x_moves = torch.cat([move_history, move_direction], dim=1)
        x_moves = F.leaky_relu(self.move_history_1(x_moves), negative_slope=0.01)
        x_moves = F.relu(self.move_history_2(x_moves))

        # Process Move Direction (2D vector)
        x_direction = F.relu(self.snake_direction(move_direction))

        # Flatten everything for the fully connected layers
        x_map = x_map.view(x_map.size(0), -1)
        x_local_map = x_local_map.view(x_local_map.size(0), -1)

        # Concatenate all features
        x = torch.cat([x_map, x_local_map, x_distance, x_food, x_moves, x_direction], dim=1)

        # Fully connected layers for decision-making
        x = F.relu(self.combined_1(x))
        x = F.relu(self.combined_2(x))
        q_value = self.output(x)  # Output: Single Q-value

        return q_value  # This Q-value can be used for selecting the best action

    def run(self, map_summary, local_map, distance_from_walls, food_info, move_history, move_direction, device='cpu'):

        # Put the model on the specified device and set to evaluation mode
        self.to(device)
        self.eval()

        # Perform forward pass and get the Q-values
        with torch.no_grad():
            q_value = self.forward(
                map_summary=map_summary,
                local_map=local_map,
                distance_from_walls=distance_from_walls,
                food_info=food_info,
                move_history=move_history,
                move_direction=move_direction
            )

        # Convert the output to numpy and return it
        return q_value.squeeze(0).detach().cpu().numpy()

    def run_batch(self, states, device='cpu'):
        """
        Forward pass for a batch of states.
        Each element in `states` is a dictionary containing tensors for map_summary,
        local_map, distance_from_walls, food_info, move_history, and move_direction.
        """
        self.to(device)
        self.train() # train mode for pytorch enabled

        map_summaries, local_maps, distances, food_infos, move_histories, move_directions = zip(*states)

        map_summary = torch.stack(map_summaries).squeeze(1).to(device)
        local_map = torch.stack(local_maps).squeeze(1).to(device)
        distance_from_walls = torch.stack(distances).squeeze(1).to(device)
        food_info = torch.stack(food_infos).squeeze(1).to(device)
        move_history = torch.stack(move_histories).squeeze(1).to(device)
        move_direction = torch.stack(move_directions).squeeze(1).to(device)

        q_values = self.forward(
            map_summary=map_summary,
            local_map=local_map,
            distance_from_walls=distance_from_walls,
            food_info=food_info,
            move_history=move_history,
            move_direction=move_direction
        )
        return q_values

    def save(self, path, filename):
        torch.save(self.state_dict(), path + "\\" + filename + ".pth")
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path, device='cpu'):
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded from {path}")
        return model

