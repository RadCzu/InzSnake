from enum import Enum

class TileColors(Enum):
    EMPTY = [1.0, 1.0, 1.0]
    SNAKE_HEAD = [0.1, 1.0, 0.1]
    SNAKE_BODY = [0.3, 0.8, 0.3]
    FOOD = [1.0, 0.0, 0.0]
    WALL = [0.0, 0.0, 0.0]
    OUT_OF_BOUNDS = [0.0, 0.0, 0.0]

