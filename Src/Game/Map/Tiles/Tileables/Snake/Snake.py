from Src.Game.Interfaces.Observer import Observer
from Src.Game.Map.Tiles.Tile import Tile
from Src.Game.Map.Tiles.Tileables.Empty import Empty
from Src.Game.Map.Tiles.Tileables.Snake.SnakeBody import SnakeBody
from Src.Game.Map.Tiles.Tileables.Snake.SnakeHead import SnakeHead


class Snake():
    def __init__(self, x, y, map):
        self.map = map
        self.head = SnakeHead(tile=self.map.get_tile(x, y), snake=self)
        self.last_tile = None
        self.food_observer = Observer()
        self.score_observer = Observer()
        self.move_observer = Observer()
        self.death_observer = Observer()
        self.length = 1
        self.score = 0
        self.moved = False
        self.dead = False

    def move(self):
        self.move_observer.notify()
        snake_direction = self.head.get_direction()
        head_tile = self.head.get_tile()
        next_tile = self.map.get_tile(head_tile.x + snake_direction[0], head_tile.y + snake_direction[1])
        if next_tile is None:
            return self
        content = next_tile.get_content()
        last_part = self.head.get_last()
        self.last_tile = last_part.tile
        self.head.previous_direction = self.head.get_direction()
        self.map.move_tile_by_coordinates(head_tile.x, head_tile.y, head_tile.x + snake_direction[0],
                                          head_tile.y + snake_direction[1])

        if last_part != self.head:
            self.map.move_tile_by_coordinates(x1=last_part.tile.x, y1=last_part.tile.y, x2=head_tile.x, y2=head_tile.y)

            second_to_last = self.head.get_second_to_last()

            if second_to_last == self.head:
                content.interact(self)
                return self

            last_part.set_next(self.head.next)
            self.head.set_next(last_part)
            second_to_last.set_next(None)

        content.interact(self)
        self.moved = True

    def grow_on_tile(self, tile):
        last_part = self.head.get_last()
        last_part.set_next(SnakeBody(tile))
        self.food_observer.notify()
        self.length += 1

    def get_distance_from_deadly(self):
        return self.get_deadly_left(), self.get_deadly_right(), self.get_deadly_up(), self.get_deadly_down()

    def get_deadly_right(self):
        my_x, my_y = self.head.get_position()
        distance = 0
        while True:
            tile: Tile = self.map.get_tile(my_x, my_y + distance + 1)
            if not tile.content.is_deadly():
                distance = distance + 1
            else:
                return distance

    def get_deadly_left(self):
        my_x, my_y = self.head.get_position()
        distance = 0
        while True:
            tile: Tile = self.map.get_tile(my_x, my_y - distance - 1)
            if not tile.content.is_deadly():
                distance = distance + 1
            else:
                return distance

    def get_deadly_up(self):
        my_x, my_y = self.head.get_position()
        distance = 0
        while True:
            tile: Tile = self.map.get_tile(my_x - distance - 1, my_y)
            if not tile.content.is_deadly():
                distance = distance + 1
            else:
                return distance

    def get_deadly_down(self):
        my_x, my_y = self.head.get_position()
        distance = 0
        while True:
            tile: Tile = self.map.get_tile(my_x + distance + 1, my_y)
            if not tile.content.is_deadly():
                distance = distance + 1
            else:
                return distance

    def die(self):
        checked = self.head
        while checked.next is not None:
            Empty(checked.next.tile)
            checked = checked.next

