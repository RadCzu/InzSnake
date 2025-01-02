from Src.Game.Map.Map import Map
from Src.Game.Map.Tiles.Tile import Tile
from Src.Game.Map.Tiles.Tileables.Empty import Empty
from Src.Game.Map.Tiles.Tileables.Wall import Wall


class MapFactory:
    @staticmethod
    def build_box_map(width, height):
        map_tiles = [[None for _ in range(height + 2)] for _ in range(width + 2)]

        for x in range(width + 2):
            for y in range(height + 2):
                tile = Tile(x, y, None)
                if x == 0 or y == 0 or x == width + 1 or y == height + 1:
                    wall = Wall(tile)
                    tile.set_content(wall)
                else:
                    empty = Empty(tile)
                    tile.set_content(empty)
                map_tiles[x][y] = tile

        return Map(map_tiles)

    @staticmethod
    def build_cross_map(width, height, plus_size):
        map_tiles = [[None for _ in range(height + 2)] for _ in range(width + 2)]

        for x in range(width + 2):
            for y in range(height + 2):
                tile = Tile(x, y, None)
                if x == 0 or y == 0 or x == width + 1 or y == height + 1:
                    wall = Wall(tile)
                    tile.set_content(wall)
                elif (x < plus_size + 1 or x > width - plus_size) and (y < plus_size + 1 or y > height - plus_size):
                    wall = Wall(tile)
                    tile.set_content(wall)
                else:
                    empty = Empty(tile)
                    tile.set_content(empty)
                map_tiles[x][y] = tile

        return Map(map_tiles)

    @staticmethod
    def build_grid_map(width, height):
        map_tiles = [[None for _ in range(height + 2)] for _ in range(width + 2)]

        for x in range(width + 2):
            for y in range(height + 2):
                tile = Tile(x, y, None)
                if x == 0 or y == 0 or x == width + 1 or y == height + 1:
                    wall = Wall(tile)
                    tile.set_content(wall)
                elif (x % 2 == 1) and (y % 2 == 1):
                    wall = Wall(tile)
                    tile.set_content(wall)
                else:
                    empty = Empty(tile)
                    tile.set_content(empty)
                map_tiles[x][y] = tile

        return Map(map_tiles)