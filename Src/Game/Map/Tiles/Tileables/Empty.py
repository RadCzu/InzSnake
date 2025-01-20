
from Src.Game.Map.Tiles.Tileables.ITileable import ITileable
from Src.Game.Map.Tiles.Tileables.TileColors import TileColors
from Src.Game.Map.Tiles.Tileables.TileNames import TileNames


class Empty(ITileable):
    def __init__(self, tile):
        self.name = TileNames.EMPTY
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
        return

    def to_string(self):
        return "□"

    def to_int(self):
        return 4

    def to_numbers(self):
        return TileColors.EMPTY.value

    def is_deadly(self):
        return False

