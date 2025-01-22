import copy
import numpy as np
import cv2

from Src.Game.Map.Tiles.Tile import Tile
from Src.Game.Map.Tiles.Tileables.Empty import Empty
from Src.Game.Map.Tiles.Tileables.OutOfBounds import OutOfBounds
from typing import List
from Src.Game.Map.Tiles.Tileables.TileNames import TileNames


class Map:
    def __init__(self, state: List[List[Tile]]):
        self.state: List[List[Tile]] = state

    def get_map_state(self):
        return self.state

    def get_tile(self, x, y):
        if x < 0 or y < 0:
            empty = Tile(x, y, None)
            out_of_bounds = OutOfBounds(empty)
            Empty.content = out_of_bounds
            return empty
        try:
            return self.state[x][y]
        except IndexError:
            empty = Tile(x, y, None)
            out_of_bounds = OutOfBounds(empty)
            Empty.content = out_of_bounds
            return empty

    def set_tile(self, x, y, tile_content):
        try:
            self.state[x][y].set_content(tile_content)
            return True
        except IndexError:
            print("Cell out of bounds")
            return False

    def swap_tiles(self, tile_1, tile_2):
        temp_tile_1 = tile_1.get_content()
        tile_1.set_content(tile_2.get_content())
        tile_2.set_content(temp_tile_1)

    def swap_tiles_by_coordinates(self, x1, y1, x2, y2):
        tile_1 = self.get_tile(x1, y1)
        tile_2 = self.get_tile(x2, y2)
        tile_1_content = tile_1.get_content()
        tile_2_content = tile_2.get_content()
        tile_1.set_content(tile_2_content)
        tile_2.set_content(tile_1_content)

    def move_tile_by_coordinates(self, x1, y1, x2, y2):
        tile_1 = self.get_tile(x1, y1)
        tile_2 = self.get_tile(x2, y2)
        tile_2.set_content(tile_1.get_content())
        tile_1.set_content(Empty(tile_1))

    def print_map(self):
        num_rows = len(self.state)
        num_cols = len(self.state[0]) if num_rows > 0 else 0

        # Print each row
        for col in reversed(range(num_cols)):
            print(" ".join(self.state[row][col].content.to_string() for row in range(num_rows)))
        print()

    def get_map_size(self):
        return len(self.state), len(self.state[0])

    def get_resized_map(self, toX, toY):
        # Get the normalized 3D map
        normalized_map = self.get_normalized_map_3D()

        # Resize using bilinear interpolation
        resized_map = cv2.resize(normalized_map, (toX, toY), interpolation=cv2.INTER_LINEAR)

        return resized_map

    def print_map_fragment(self, fragment):
        num_rows = len(fragment)
        num_cols = len(fragment[0]) if num_rows > 0 else 0

        if hasattr(fragment[0][0], 'content'):
            for row in range(num_rows):
                print(" ".join(fragment[row][col].content.to_string() for col in range(num_cols)))
        else:
            for row in range(num_rows):
                print(" ".join(str(fragment[row][col]) for col in range(num_cols)))
        print()

    def get_map_fragment(self, start_x, start_y, end_x, end_y):
        requested_width = end_x + 1 - start_x
        requested_height = end_y + 1 - start_y

        temp_map = [[None for _ in range(requested_width)] for _ in range(requested_height)]

        if hasattr(self.state[0][0], 'content'):
            for col in range(start_x, end_x + 1):
                for row in range(start_y, end_y + 1):
                    try:
                        temp_map[row - start_y][col - start_x] = self.get_tile(col, row)
                    except IndexError:
                        tile = self.get_tile(col, row)
                        temp_map[row - start_y][col - start_x] = tile
        answer = np.array(temp_map)
        return answer

    def get_normalized_map_3D(self):
        num_rows = len(self.state)
        num_cols = len(self.state[0]) if num_rows > 0 else 0

        norm_map = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        for row in range(num_rows):
            for col in range(num_cols):
                norm_map[row][col] = self.state[row][col].content.to_numbers()

        data = np.array(norm_map)
        return data

    def get_normalized_fragment_3D(self, fragment):
        num_rows = len(fragment)
        num_cols = len(fragment[0]) if num_rows > 0 else 0

        norm_map = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        for row in range(num_rows):
            for col in range(num_cols):
                norm_map[row][col] = fragment[row][col].content.to_numbers()

        data = np.array(norm_map)
        return data

    def get_playable_area(self):
        num_rows = len(self.state)
        num_cols = len(self.state[0]) if num_rows > 0 else 0

        empty_tiles = 0

        for row in range(num_rows):
            for col in range(num_cols):
                if self.state[row][col].get_content().get_name() == TileNames.EMPTY:
                    empty_tiles += 1

        return empty_tiles