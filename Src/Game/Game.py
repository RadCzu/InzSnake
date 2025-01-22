import copy
import random
from typing import List

import numpy as np

from Src.Game.Interfaces.Observer import Observer
from Src.Game.Map.Map import Map
from Src.Game.Map.Tiles.Tileables.Empty import Empty
from Src.Game.Map.Tiles.Tileables.Food import Food
from Src.Game.Map.Tiles.Tileables.Snake.Snake import Snake


class Game():
    def __init__(self, map, snake_positions, food_density=0.1):
        self.map: Map = map
        self.snakes = []
        self.snakeEliminator = Observer()
        for snake_position in snake_positions:
            snake = Snake(x=snake_position[0], y=snake_position[1], map=self.map)
            self.snakes.append(snake)
            snake.food_observer.subscribe(self.spawn_random_food)
            snake.death_observer.subscribe(lambda: self.snakeEliminator.subscribe(lambda: self.kill_snake(snake)))
        self.food_list = []
        self.active = False
        self.over = False
        self.food_density = food_density

    def spawn_random_food(self, amount=1):
        for i in range(0, amount):
            empty_tile = self.get_random_empty_tile()
            if empty_tile:
                food = Food(empty_tile)
                self.food_list.append(food)
                food.on_eaten.subscribe(lambda f=food: self.remove_food_from_list(f))

    def get_random_empty_tile(self):
        empty_tiles = []
        for row in self.map.get_map_state():
            for tile in row:
                if isinstance(tile.get_content(), Empty):
                    empty_tiles.append(tile)
        if not empty_tiles:
            return None
        return random.choice(empty_tiles)

    def remove_food_from_list(self, food_item):
        self.food_list.remove(food_item)

    def get_n_closest_food_items(self, n, x, y):
        """
        Get the n closest food items to the given position (x, y).

        Args:
            n (int): The number of closest food items to retrieve.
            x (float): The x-coordinate of the reference position.
            y (float): The y-coordinate of the reference position.

        Returns:
            List[Food]: A list of the n closest food items.
        """
        food_distances = []

        for food in self.food_list:
            food_position = food.get_tile().get_position()
            distance = np.linalg.norm(np.array(food_position) - np.array([x, y]))
            food_distances.append((distance, food))

        food_distances.sort(key=lambda item: item[0])

        closest_food_items = [item[1] for item in food_distances[:n]]

        return np.array(closest_food_items)

    def begin(self):
        foods = int(self.map.get_playable_area() * self.food_density)
        self.spawn_random_food(foods)
        self.start()

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def kill_snake(self, snake):
        snake.die()

    def game_over(self):
        self.over = True
        self.stop()

    def left(self, snake):
        if self.active:
            snake.head.set_direction([-1, 0])

    def right(self, snake):
        if self.active:
            snake.head.set_direction([1, 0])

    def up(self, snake):
        if self.active:
            snake.head.set_direction([0, 1])

    def down(self, snake):
        if self.active:
            snake.head.set_direction([0, -1])

    def update(self, snake):
        if self.active:
            snake.move()

    def input(self, number, snake):
        if number == 0:
            self.left(snake)
        elif number == 1:
            self.right(snake)
        elif number == 2:
            self.up(snake)
        elif number == 3:
            self.down(snake)
        else:
            print("wrong input, stopping snake")
            self.stop()

    def print_state(self):
        self.map.print_map()



