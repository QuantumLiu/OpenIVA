from . import StoppableThread

from queue import Queue

from openiva.models import BaseNet, ModelConfig


class ThreadCompute(StoppableThread):
    def __init__(self, q_in: Queue, q_out: Queue, model_config: ModelConfig = None, engine_config: dict = None) -> None:
        if model_config is None:
            raise ValueError

        self.q_in = q_in
        self.q_out = q_out
        self.model_config = model_config

    def _load_model(self):
        pass
