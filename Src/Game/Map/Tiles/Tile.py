from Src.Game.Map.Tiles.Tileables.ITileable import ITileable


class Tile:
    def __init__(self, x, y, content):
        self.x = x
        self.y = y
        self.content : ITileable | None = content

    def set_content(self, content):
        self.content = content
        content.tile = self

    def get_content(self):
        return self.content

    def get_position(self):
        return self.x, self.y
