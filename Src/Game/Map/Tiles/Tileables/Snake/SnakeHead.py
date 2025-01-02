from Src.Game.Map.Tiles.Tileables.Snake.SnakePart import SnakePart
from Src.Game.Map.Tiles.Tileables.TileColors import TileColors
from Src.Game.Map.Tiles.Tileables.TileNames import TileNames


class SnakeHead(SnakePart):
    def __init__(self, tile, snake):
        super().__init__(TileNames.SNAKE_HEAD, tile)
        self.direction = [-1, 0]
        self.previous_direction = [0, 0]
        self.snake = snake

    def get_direction(self):
        return self.direction

    def set_direction(self, direction):
        self.direction = direction

    def interact(self, snake):
        snake.die(self)
        if not self.is_part_of_snake(snake):
            self.tile = snake.head.tile
            snake.head.tile.set_content(self)
            if self.snake.moved:
                self.snake.die()

    def to_string(self):
        return "S"

    def to_int(self):
        return 1

    def to_numbers(self):
        return TileColors.SNAKE_HEAD.value
