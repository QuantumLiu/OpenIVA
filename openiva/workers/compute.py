from . import StoppableThread

from queue import Queue

from openiva.models import BaseNet, ModelConfig


class ThreadCompute(StoppableThread):
    def __init__(self, q_in: Queue, q_out: Queue, model_config: ModelConfig = None) -> None:
        super().__init__()
        if model_config is None:
            raise ValueError

        self.q_in = q_in
        self.q_out = q_out
        self.model_config = model_config
        self.model_name = self.model_config.model_name
        self.weights_path = self.model_config.weights_path
        self.model = None

    def run(self):
        print("Loading {} model from path {}".format(
            self.model_name, self.weights_path))
        self.model = self.model_config.model_class(
            self.model_config.weights_path, **self.model_config.kwargs)

        while True:
            q_dict_in = self.q_in.get()

            data_infer = q_dict_in[self.model_name]['data_preproc']

            infer_result = self.model.infer(data_infer)

            q_dict_in[self.model_name]['infer_result'] = infer_result

            self.q_out.put(q_dict_in)
