import math

from typing import List

import torch

from Src.Game.Agents.AIAgent import AIAgent
from Src.AI.Training.LearningExperience import LearningExperience
from Src.AI.Training.Reward import Reward
from Src.Game.Interfaces.Observer import Observer


class TrainingAIAgent(AIAgent):

    def __init__(self, brain_network, experience_manager=None, reward_decay=1., move_memory=20, move_picking_strategy=None):
        super().__init__(brain_network, move_memory, move_picking_strategy)
        self.experience_manager: LearningExperience | None = experience_manager
        self.reward_decay = reward_decay
        self.cookies = 0
        self.has_eaten = False
        self.previous_reward = 0
        self.state_hold = None
        self.done_hold = False

    def on_init(self, game):
        death_reward = Reward(-10, self)
        self.snake.death_observer.subscribe(death_reward.apply_reward)
        self.snake.death_observer.subscribe(self.clear_history)
        move_reward = Reward(-1, self)
        self.snake.move_observer.subscribe(move_reward.apply_reward)
        score_reward = Reward(10, self)
        self.snake.score_observer.subscribe(score_reward.apply_reward)
        self.snake.score_observer.subscribe(self.on_eat)

    def on_eat(self):
        self.has_eaten = True

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

        # if self.snake.dead:

        return Q_values

    def is_done(self):
        previous = self.done_hold
        self.done_hold = self.snake.dead or self.has_eaten
        return previous

    def add_last_state(self, game):
        state = self.state_hold
        move_dir = [0, 0, 0, 0]
        move_tensor = torch.tensor(move_dir, dtype=torch.float32).unsqueeze(0)
        state = (state[0], state[1], state[2], state[3], state[4], move_tensor)
        self.add_brain_experience(state, self.cookies, True)
        self.cookies = 0
        if self.experience_manager is not None:
            self.experience_manager.previous_state = None

    def on_game_over(self, game):
        return

    def reward(self, cookies):
        self.cookies += cookies

    def apply_reward_decay(self):
        self.cookies = self.cookies * self.reward_decay

    def add_brain_experience(self, state, reward, done):
        if self.experience_manager is not None:
            self.experience_manager.add_deep_experience(reward, state, done)
            if done:
                self.cookies = 0

    def get_cookies(self):
        return self.cookies
