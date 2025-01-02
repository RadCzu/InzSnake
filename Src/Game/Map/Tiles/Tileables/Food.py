from Src.Game.Interfaces.Observer import Observer
from Src.Game.Map.Tiles.Tileables.ITileable import ITileable
from Src.Game.Map.Tiles.Tileables.TileColors import TileColors
from Src.Game.Map.Tiles.Tileables.TileNames import TileNames


class Food(ITileable):
    def __init__(self, tile):
        self.name = TileNames.FOOD
        tile.set_content(self)
        self.tile = tile
        self.on_eaten = Observer()

    def get_name(self):
        return self.name

    def get_content(self):
        return None

    def get_tile(self):
        return self.tile

    def interact(self, snake):
        snake.grow_on_tile(snake.last_tile)
        snake.score = snake.score + 1
        snake.score_observer.notify()
        self.on_eaten.notify()

    def to_string(self):
        return "F"

    def to_int(self):
        return 3

    def to_numbers(self):
        return TileColors.FOOD.value

    def is_deadly(self):
        return False
