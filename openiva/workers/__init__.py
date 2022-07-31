from threading import Thread, Event

class StoppableThread(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.setDaemon(True)


from .data import ThreadDATA
from .data import ThreadImgsLocal,ThreadVideoLocal

from .compute import ThreadCompute