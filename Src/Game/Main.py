import math

from Src.Game.Agents.IAgent import IAgent
from Src.Game.Agents.TrainingAIAgent import TrainingAIAgent
from Src.Game.Game import Game
from Src.Game.Interfaces.Observer import Observer
from Src.Game.Map.MapFactory import MapFactory


class Main:
    def __init__(self, map_params, map_type, agents, game_length=0, food_density=0.1):
        self.game_width = map_params[0]
        self.game_height = map_params[1]
        self.map_type = map_type
        self.map_params = map_params
        self.food_density = food_density
        self.agents = []
        self.agent_positions = []

        for agent in agents:
            the_agent: IAgent = agent[0]
            self.agents.append(the_agent)
            self.agent_positions.append((agent[1], agent[2]))
        self.game_length = game_length
        self.turn = 0
        self.game = None

    def set_up(self):
        self.turn = 0
        if self.map_type == "cross":
            the_map = MapFactory().build_cross_map(self.map_params[0], self.map_params[1], self.map_params[2])
        elif self.map_type == "grid":
            the_map = MapFactory().build_grid_map(self.map_params[0], self.map_params[1])
        else:
            the_map = MapFactory().build_box_map(self.map_params[0], self.map_params[1])

        game = Game(the_map, self.agent_positions, self.food_density)
        for i in range(len(game.snakes)):
            self.agents[i].set_snake(game.snakes[i])
        game.begin()
        return game

    def run(self):
        self.game = self.set_up()
        for agent in self.agents:
            agent.on_init(self.game)
        living = len(self.agents)

        while self.game.over is False and living > 0:
            living = len(self.agents)
            for agent in self.agents:
                agent.snake.moved = False
            if (self.game_length != 0 and self.turn >= self.game_length) or living == 0:
                self.game.game_over()
            self.turn += 1

            for agent in self.agents:
                if agent.snake.dead:
                    living = living - 1
                    continue
                else:
                    agent.make_decision(self.game)

                    # training actions before a move is made
                    if isinstance(agent, TrainingAIAgent) and agent.snake.dead == False:
                        snake_x, snake_y = agent.snake.head.get_tile().get_position()
                        closest_food_objects = self.game.get_n_closest_food_items(4, snake_x, snake_y)
                        food_coordinates = []
                        for food_object in closest_food_objects:
                            (food_x, food_y) = food_object.get_tile().get_position()
                            food_coordinates.append((food_x, food_y))
                        _food_positions_1 = self._get_food_positions(agent, food_coordinates)

                    self.game.update(agent.snake)
                    if agent.snake.dead:
                        here = "here"

                    # training actions after a move is made
                    if isinstance(agent, TrainingAIAgent):
                        if agent.snake.dead is False:
                            food_positions_2 = self._get_food_positions(agent, food_coordinates)

                            for i in range(len(food_positions_2)):
                                food_distance_1 = _food_positions_1[i]
                                food_distance_2 = food_positions_2[i]
                                if food_distance_1 == 0:
                                    agent.reward(food_distance_2 * -1)
                                else:
                                    proximity_reward = ((food_distance_1 - food_distance_2) / food_distance_1)
                                    agent.reward(proximity_reward)

                        agent.add_brain_experience(agent.state_hold, agent.previous_reward, agent.is_done())
                        agent.previous_reward = agent.cookies

                        if agent.snake.dead:
                            agent.add_last_state(self.game)


            self.game.snakeEliminator.notify()
            self.game.snakeEliminator = Observer()

        for agent in self.agents:
            agent.on_game_over(self.game)

    def _get_food_positions(self, agent: TrainingAIAgent, food_coordinates):
        food_positions = []
        if agent.snake.head.get_tile() is None:
            snake_x, snake_y = agent.snake.last_tile.get_position()
        else:
            snake_x, snake_y = agent.snake.head.get_tile().get_position()
        for food_coordinate in food_coordinates:
            food_x, food_y = food_coordinate
            food_distance = math.sqrt((food_x - snake_x) ** 2 + (food_y - snake_y) ** 2)
            food_positions.append(food_distance)
        return food_positions

    def run_with_print(self):
        self.game = self.set_up()
        for agent in self.agents:
            agent.on_init(self.game)
        living = len(self.agents)
        while self.game.over is False and living > 0:
            self.game.map.print_map()
            living = len(self.agents)
            for agent in self.agents:
                agent.snake.moved = False
            if self.game_length != 0 and self.turn >= self.game_length:
                self.game.game_over()
            self.turn += 1

            for i in range(len(self.agents)):
                if self.agents[i].snake.dead:
                    living = living - 1
                    continue
                else:
                    q_vals = self.agents[i].make_decision(self.game)
                    self.game.update(self.agents[i].snake)
                    print(f"Agent {i}:")
                    print("Q Values: ")
                    print(q_vals)

            self.game.snakeEliminator.notify()
            self.game.snakeEliminator = Observer()

        for agent in self.agents:
            agent.on_game_over(self.game)
        self.game.map.print_map()
