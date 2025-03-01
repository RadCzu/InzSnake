import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

class SnakeTrainer:
    def __init__(self, network, experience_manager, batch_size=64, device='cpu', gamma=0.95, learning_rate=0.0001,
                 learning_damp_frequency=500):
        self.experience_manager = experience_manager
        self.network = network
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor for future rewards
        self.gamma_min = 0.1
        self.brain_optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.device = device
        self.learning_rate = learning_rate
        self.episode = 0
        self.learning_damp_frequency = learning_damp_frequency
        self.avg_loss = []

    def train(self):
        # Train the brain network
        self.train_brain()
        self.episode += 1

    def train_brain(self):
        if len(self.experience_manager.memory) < self.batch_size:
            return  # Not enough experience to sample a batch

        experience_batch = self.experience_manager.sample_batch(self.batch_size)

        # Unpack the batch
        states, rewards, next_states, dones = experience_batch

        # Process tensors for training
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Current Q-values
        q_values = self.network.run_batch(states, device=self.device).squeeze(1)  # Ensure it's 1D

        # for i in range(len(rewards)):
        #     if dones[i] == True and rewards[i] == 0.0:
        #         local_map = states[i][1].squeeze(0)
        #         local_map_next = next_states[i][1].squeeze(0)
        #
        #         # Convert tensors to numpy for visualization
        #         local_map_np = local_map.cpu().numpy()
        #         local_map_next_np = local_map_next.cpu().numpy()
        #
        #         local_map_np = np.transpose(local_map_np, (1, 2, 0))  # Change (3, 15, 15) -> (15, 15, 3)
        #         local_map_next_np = np.transpose(local_map_next_np, (1, 2, 0))  # Change (3, 15, 15) -> (15, 15, 3)
        #
        #         # Display local_map
        #         print(f"Debug: Visualizing local_map for state {i}")
        #         plt.imshow(local_map_np)
        #         plt.title("Local Map")
        #         plt.figtext(0.25, 0.01, f"Q:${q_values[i]}")
        #         plt.axis('off')
        #         plt.show()
        #
        #         # Display local_map
        #         print(f"Debug: Visualizing local_map for next state {i}")
        #         plt.imshow(local_map_next_np)
        #         plt.title(f"Local Map Next")
        #         plt.axis('off')
        #         plt.show()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.network.run_batch(next_states, device=self.device).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute loss
        loss = self.criterion(q_values, target_q_values)
        self.avg_loss.append(loss.item())
        if len(self.avg_loss) > 500:
            self.avg_loss.pop(0)
        print(f"Average Loss: {np.average(self.avg_loss)}")

        # Backpropagate and update the network
        self.brain_optimizer.zero_grad()
        loss.backward()
        self.brain_optimizer.step()

        if self.learning_damp_frequency != 0 and self.episode % self.learning_damp_frequency == 0:
            self.apply_learning_damp()

    def apply_learning_damp(self):
        self.learning_rate = self.learning_rate * 0.9
        for param_group in self.brain_optimizer.param_groups:
            param_group['lr'] = self.learning_rate
