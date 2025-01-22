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

            longest_len = 0
            longest_agent = None
            for agent in self.agents:
                if agent.snake.length > longest_len:
                    longest_len = agent.snake.length
                if agent.snake.head.tile is None and agent.snake.dead == False:
                    here = "here"
                    self.game.map.print_map()
                if agent.snake.dead:
                    living = living - 1
                    continue
                else:
                    agent.make_decision(self.game)
                    self.game.update(agent.snake)
                    if isinstance(agent, TrainingAIAgent):
                        agent.add_brain_experience(agent.state_hold, agent.action_hold, agent.previous_reward, agent.done_hold)
                        agent.previous_reward = agent.cookies

            if isinstance(longest_agent, TrainingAIAgent):

                longest_agent.on_winning.notify()

            self.game.snakeEliminator.notify()
            self.game.snakeEliminator = Observer()

        for agent in self.agents:
            agent.on_game_over(self.game)

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

            longest_len = 0
            longest_agent = None

            for i in range(len(self.agents)):
                if self.agents[i].snake.length > longest_len:
                    agent = self.agents[i]
                    longest_len = agent.snake.length
                    longest_agent = agent
                if self.agents[i].snake.dead:
                    living = living - 1
                    continue
                else:
                    q_vals = self.agents[i].make_decision(self.game)
                    self.game.update(self.agents[i].snake)
                    print(f"Agent {i}:")
                    print("Q Values: ")
                    print(q_vals)

            if isinstance(longest_agent, TrainingAIAgent):
                longest_agent.on_winning.notify()

            self.game.snakeEliminator.notify()
            self.game.snakeEliminator = Observer()

        for agent in self.agents:
            agent.on_game_over(self.game)
        self.game.map.print_map()
