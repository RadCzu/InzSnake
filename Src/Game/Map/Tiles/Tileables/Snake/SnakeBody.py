from Src.Game.Map.Tiles.Tileables.Snake.SnakePart import SnakePart
from Src.Game.Map.Tiles.Tileables.TileNames import TileNames


class SnakeBody(SnakePart):
    def __init__(self, tile, snake):
        super().__init__(TileNames.SNAKE_BODY, tile, snake)

    def to_string(self):
        return "N"

    def to_int(self):
        return 2
