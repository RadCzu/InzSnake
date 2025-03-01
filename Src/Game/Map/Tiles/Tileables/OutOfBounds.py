from Src.Game.Map.Tiles.Tileables.ITileable import ITileable
from Src.Game.Map.Tiles.Tileables.TileColors import TileColors
from Src.Game.Map.Tiles.Tileables.TileNames import TileNames


class OutOfBounds(ITileable):
    def __init__(self, tile):
        self.name = TileNames.OUT_OF_BOUNDS
        tile.set_content(self)
        self.tile = tile

    def get_name(self):
        return self.name

    def get_content(self):
        return None

    def get_tile(self):
        return self.tile

    def interact(self, snake):
        snake.death_observer.notify()
        OutOfBounds(snake.head.tile)
        snake.head.tile = None
        return

    def to_string(self):
        return "O"

    def to_int(self):
        return 5

    def to_numbers(self):
        return TileColors.OUT_OF_BOUNDS.value

    def is_deadly(self):
        return True

