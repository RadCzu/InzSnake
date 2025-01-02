

class Observer():

    def __init__(self):
        self.listeners = []

    def subscribe(self, callback):
        self.listeners.append(callback)

    def unsubscribe(self, callback):
        self.listeners.remove(callback)

    def notify(self):
        for callback in self.listeners:
            callback()
