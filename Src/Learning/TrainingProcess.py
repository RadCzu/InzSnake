import csv
import os
import random
import numpy as np
import torch

from Src.AI.NeuralNetworks.SnakeBrain import SnakeBrain
from Src.Game.Agents.AIAgent import AIAgent
from Src.Game.Agents.Move_picking.PickBest import PickBest
from Src.Game.Agents.Move_picking.PickRandom import PickWeightedRandom, PickScaledRandom
from Src.Game.Agents.TrainingAIAgent import TrainingAIAgent
from Src.AI.Training.LearningExperience import LearningExperience
from Src.Game.Agents.Move_picking.PickDynamic import PickDynamic
from Src.Game.Agents.TrainingAIAgentNoFood import TrainingAIAgentNoFood
from Src.Game.Main import Main
from Src.AI.Training.SnakeTrainer import SnakeTrainer


class AITrainingProcess:
    def __init__(self, save_path, file_name, snapshot_name, score_file, experience_buffer_size=5000):
        self.save_path = save_path
        self.file_name = file_name
        self.snapshot_name = snapshot_name
        self.score_file = score_file

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        print(f"Using device: {self.device}")

        self._initialize_score_file()
        self.brain = self._load_brain()
        self.experience_manager = LearningExperience(experience_buffer_size)
        self.trainer = SnakeTrainer(network=self.brain,
                                    experience_manager=self.experience_manager,
                                    batch_size=16,
                                    device=self.device,
                                    gamma=0.9,
                                    learning_damp_frequency=0)

    def _initialize_score_file(self):
        if not os.path.exists(self.score_file):
            with open(self.score_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Episode', 'Average Score'])

    def _load_brain(self):
        try:
            brain = SnakeBrain().load_model(path=f"{self.save_path}\\{self.snapshot_name}.pth", device=self.device)
            print("Brain loaded successfully.")
        except Exception as e:
            print(f"Error loading brain: {e}. Initializing a new brain.")
            brain = SnakeBrain()
        return brain

    def _print_network(self, neural_network):
        for name, param in neural_network.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}")
                print(f"Weights: {param.data}")
                print(f"Gradients: {param.grad}")

    def train(self, num_episodes=5000, game_length=100, avg_reset=100):
        averages = []
        total_score = 0
        global_score = []
        high_score = 0

        for episode in range(num_episodes):
            # Generate random map parameters and type
            rand_X = random.randint(7, 32)
            rand_Y = random.randint(7, 32)
            map_type = random.choice(["cross", "grid", "box"])

            if map_type == "cross":
                plus_size = random.randint(1, int(min(rand_X, rand_Y) * 0.4))
                map_params = (rand_X, rand_Y, plus_size)
            else:
                map_params = (rand_X, rand_Y)

            # Define agents with initial positions
            agents = [
                (TrainingAIAgentNoFood(self.brain, self.experience_manager, 0.2, move_picking_strategy=PickDynamic()),
                 rand_X // 2,
                 rand_Y // 4),
                (TrainingAIAgentNoFood(self.brain, self.experience_manager, 0.2, move_picking_strategy=PickDynamic()),
                 rand_X // 2,
                 (3 * rand_Y) // 4)
            ]

            # Initialize the game with multiple agents
            game = Main(map_params, map_type, agents, game_length, 0.1)
            game.run()

            self.trainer.train()

            if episode != 0 and episode % avg_reset == 0:
                averages.append(total_score / avg_reset)
                self.brain.save(self.save_path, self.snapshot_name)
                total_score = 0
                print(f"High score: {high_score} =====================================")

                with open(self.score_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode + 1, np.average(global_score)])

            # Aggregate scores for all agents
            episode_score = (sum(agent[0].snake.score for agent in agents) / len(agents))
            total_score += episode_score
            global_score.append(episode_score)

            if len(global_score) > 500:
                global_score.pop(0)

            print(f"Episode {episode + 1}/{num_episodes} completed")
            print(f"Global Score: {np.average(global_score):.2f}")

            high_score = max(high_score, episode_score)

            if episode == num_episodes - 1:
                self.brain.save(self.save_path, self.file_name)

        self.brain.save(self.save_path, self.file_name)
        print("Training complete. Averages:", averages)

    def example_game(self, game_length=100):
        rand_X = random.randint(7, 12)
        rand_Y = random.randint(7, 12)
        map_type = random.choice(["cross", "grid", "box", "box", "box", "box"])

        if map_type == "cross":
            plus_size = random.randint(1, int(min(rand_X, rand_Y) * 0.4))
            map_params = (rand_X, rand_Y, plus_size)
        else:
            map_params = (rand_X, rand_Y)

        # Define agents with initial positions
        agents = [
            (
                AIAgent(self.brain, 20, move_picking_strategy=PickBest()), (rand_X+2) // 2,
                (rand_Y+2) // 4),
            (
                AIAgent(self.brain, 20, move_picking_strategy=PickBest()), (rand_X+2) // 2,
                (3 * (rand_Y+2)) // 4)
        ]

        # Initialize the game with multiple agents
        game = Main(map_params, map_type, agents, game_length, 0.1)
        game.run_with_print()


    def train_single(self, num_episodes=5000, game_length=100, avg_reset=100):
        averages = []
        total_score = 0
        global_score = []
        high_score = 0

        for episode in range(num_episodes):
            # Generate random map parameters and type
            rand_X = random.randint(7, 32)
            rand_Y = random.randint(7, 32)
            map_type = random.choice(["cross", "grid", "box"])

            if map_type == "cross":
                plus_size = random.randint(1, int(min(rand_X, rand_Y) * 0.4))
                map_params = (rand_X, rand_Y, plus_size)
            else:
                map_params = (rand_X, rand_Y)

            # Define agents with initial positions
            agents = [
                (TrainingAIAgentNoFood(self.brain, self.experience_manager, 0.2, move_picking_strategy=PickDynamic()),
                 rand_X // 2,
                 rand_Y // 2),
            ]

            # Initialize the game with multiple agents
            game = Main(map_params, map_type, agents, game_length, 0.1)
            game.run()

            self.trainer.train()

            if episode != 0 and episode % avg_reset == 0:
                averages.append(total_score / avg_reset)
                self.brain.save(self.save_path, self.snapshot_name)
                total_score = 0
                print(f"High score: {high_score} =====================================")

                with open(self.score_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode + 1, np.average(global_score)])

            # Aggregate scores for all agents
            episode_score = (sum(agent[0].snake.score for agent in agents) / len(agents))
            total_score += episode_score
            global_score.append(episode_score)

            if len(global_score) > 500:
                global_score.pop(0)

            print(f"Episode {episode + 1}/{num_episodes} completed")
            print(f"Global Score: {np.average(global_score):.2f}")

            high_score = max(high_score, episode_score)

            if episode == num_episodes - 1:
                self.brain.save(self.save_path, self.file_name)

        self.brain.save(self.save_path, self.file_name)
        print("Training complete. Averages:", averages)

    def example_game_single(self, game_length=100):
        rand_X = random.randint(7, 12)
        rand_Y = random.randint(7, 12)
        map_type = random.choice(["cross", "grid", "box", "box", "box", "box"])

        if map_type == "cross":
            plus_size = random.randint(1, int(min(rand_X, rand_Y) * 0.4))
            map_params = (rand_X, rand_Y, plus_size)
        else:
            map_params = (rand_X, rand_Y)

        # Define agents with initial positions
        agents = [
            (
                AIAgent(self.brain, 20, move_picking_strategy=PickBest()), (rand_X + 2) // 2,
                (rand_Y + 2) // 2),
        ]

        # Initialize the game with multiple agents
        game = Main(map_params, map_type, agents, game_length, 0.1)
        game.run_with_print()
