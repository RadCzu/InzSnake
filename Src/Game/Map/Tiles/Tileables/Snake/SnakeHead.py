from Src.Game.Map.Tiles.Tileables.Snake.Snake import Snake
from Src.Game.Map.Tiles.Tileables.Snake.SnakePart import SnakePart
from Src.Game.Map.Tiles.Tileables.TileColors import TileColors
from Src.Game.Map.Tiles.Tileables.TileNames import TileNames


class SnakeHead(SnakePart):
    def __init__(self, tile, snake):
        super().__init__(TileNames.SNAKE_HEAD, tile, snake)
        self.direction = [-1, 0]
        self.previous_direction = [0, 0]

    def get_direction(self):
        return self.direction

    def set_direction(self, direction):
        self.direction = direction

    def interact(self, snake: Snake):
        if not self.snake.moved:
            if self.next() is not None:
                snake.death_observer.notify()
                snake.head.tile.set_content(self)
                snake.head.tile = None
            else:
                dir_dif = (snake.head.direction[0] - self.snake.head.direction[0], snake.head.direction[1] - self.snake.head.direction[1])
                if dir_dif == (0, 0):
                    snake.death_observer.notify()
                    self.snake.death_observer.notify()
                    self.snake.head.tile = None
                    snake.head.tile = None
                return
        else:
            self.snake.death_observer.notify()
            self.snake.head.tile = None
            snake.death_observer.notify()
            snake.head.tile = None

    def to_string(self):
        return "S"

    def to_int(self):
        return 1

    def to_numbers(self):
        return TileColors.SNAKE_HEAD.value
