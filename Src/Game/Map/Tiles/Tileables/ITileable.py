from abc import ABC, abstractmethod


class ITileable(ABC):
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_content(self):
        pass

    @abstractmethod
    def get_tile(self):
        pass

    @abstractmethod
    def interact(self, snake: Snake):
        pass

    @abstractmethod
    def to_string(self):
        pass

    @abstractmethod
    def to_int(self):
        pass

    @abstractmethod
    def is_deadly(self):
        pass

    @abstractmethod
    def to_numbers(self):
        pass
