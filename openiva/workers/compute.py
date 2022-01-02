from . import StoppableThread

from queue import Queue

from openiva.models import BaseNet, ModelGroup
class ThreadCompute(StoppableThread):
    def __init__(self,q_in: Queue,q_out: Queue,models:ModelGroup=None) -> None:
        super().__init__()
        if models is None:
            raise ValueError
