from . import StoppableThread

from queue import Queue

from openiva.models import BaseNet, ModelConfig


class ThreadCompute(StoppableThread):
    def __init__(self, q_in: Queue, q_out: Queue, model_config: ModelConfig = None) -> None:
        if model_config is None:
            raise ValueError

    def _load_model(self):
        pass
