from Src.Game.Agents.IAgent import IAgent
from Src.Game.Game import Game
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
            the_agent.on_init(self.game)
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
        game.begin()
        for snake in game.snakes:
            snake.death_observer.subscribe(self.game.game_over)
        return game

    def run(self):
        self.game = self.set_up()
        while self.game.over is False:
            for agent in self.agents:
                agent.snake.moved = False
            if self.game_length != 0 and self.turn >= self.game_length:
                self.game.game_over()
            self.turn += 1
            for agent in self.agents:
                agent.make_decision(self.game)

        for agent in self.agents:
            agent.on_game_over(self.game)

    def run_with_print(self):
        self.game = self.set_up()

        while self.game.over is False:
            for agent in self.agents:
                agent.snake.moved = False
            self.game.print_state()
            for i in range(len(self.agents)):
                q_vals = self.agents[i].make_decision(self.game)
                print(f"Agent {i}:")
                print("Q Values: ")
                print(q_vals)
                print(self.agents[i].get_cookies())

        for agent in self.agents:
            agent.on_game_over(self.game)
