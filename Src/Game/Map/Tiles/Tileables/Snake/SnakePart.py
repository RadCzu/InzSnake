from abc import ABC

from Src.Game.Map.Tiles.Tileables.ITileable import ITileable
from Src.Game.Map.Tiles.Tileables.TileColors import TileColors


class SnakePart(ITileable, ABC):
    def __init__(self, name, tile, snake):
        self.name = name
        self.next = None
        self.tile = tile
        if self.tile is not None:
            tile.set_content(self)
        self.snake = snake

    def get_name(self):
        return self.name

    def get_content(self):
        return self.next

    def get_tile(self):
        return self.tile

    def get_next(self):
        return self.next

    def set_next(self, next):
        self.next = next

    def get_position(self):
        return self.tile.get_position()

    def get_last(self):
        if not self.next:
            return self
        else:
            return self.get_next().get_last()

    def get_second_to_last(self):
        if not self.next:
            print("second to last could not be found")
            return self
        elif not self.next.next:
            return self
        else:
            return self.get_next().get_second_to_last()

    def interact(self, snake):
        if not self.snake.moved and self.next is not None:
            snake.death_observer.notify()
        if not self.is_part_of_snake(snake):
            self.tile = snake.head.tile
            snake.head.tile.set_content(self)
            snake.head.tile = None
        else:
            snake.death_observer.notify()
            snake.head.tile = None
        snake.dead = True

    def is_part_of_snake(self, snake):
        """
        Determine if this SnakePart is part of the given snake.

        :param snake: The snake object whose parts we want to check.
        :return: True if this part belongs to the snake, False otherwise.
        """
        return snake == self.snake

    def to_string(self):
        pass

    def to_int(self):
        pass

    def to_numbers(self):
        return TileColors.SNAKE_BODY.value

    def is_deadly(self):
        return True
