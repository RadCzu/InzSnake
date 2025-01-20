import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from Src.Game.Agents.TrainingAIAgent import TrainingAIAgent


class SnakeTrainer:
    def __init__(self, network, experience_manager, batch_size=64, device='cpu', gamma=0.95, learning_rate=0.0001, learning_damp_frequency=500):
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
    def train(self):
        # Train the brain network
        self.train_brain()
        self.episode += 1

    def train_brain(self):
        if len(self.experience_manager.memory) < self.batch_size:
            return  # Not enough experience to sample a batch

        experience_batch = self.experience_manager.sample_batch(self.batch_size)

        # Unpack the batch
        states, action, rewards, next_states, dones = experience_batch

        # Process tensors for training
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Current Q-values
        q_values = self.network.run_batch(states, device=self.device).squeeze(1)  # Ensure it's 1D

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.network.run_batch(next_states, device=self.device).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagate and update the network
        self.brain_optimizer.zero_grad()
        loss.backward()
        self.brain_optimizer.step()

        if self.learning_damp_frequency != 0 and self.episode % self.learning_damp_frequency == 0:
            self.learning_rate = self.learning_rate * 0.9
            if self.gamma > self.gamma_min:
                self.gamma = self.gamma * 0.9
            for param_group in self.brain_optimizer.param_groups:
                param_group['lr'] = self.learning_rate
