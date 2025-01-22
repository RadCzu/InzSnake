from Src.Game.Map.Tiles.Tileables.ITileable import ITileable
from Src.Game.Map.Tiles.Tileables.TileColors import TileColors
from Src.Game.Map.Tiles.Tileables.TileNames import TileNames


class Wall(ITileable):
    def __init__(self, tile):
        self.name = TileNames.WALL
        if tile is not None:
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
        Wall(snake.head.tile)
        snake.head.tile = None
        snake.dead = True
        return

    def to_string(self):
        return "W"

    def to_int(self):
        return 5

    def to_numbers(self):
        return TileColors.WALL.value

    def is_deadly(self):
        return True
