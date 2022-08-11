from .base import BaseNetNew


class ModelConfig(object):

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def __init__(self, model_name: str, model_class: BaseNetNew, weights_path: str, func_preproc: callable = None, func_postproc: callable = None, preproc_kwargs: dict = None, postproc_kwargs: dict = None, **kwargs) -> None:
        super().__init__()
        self._model_name = model_name
        self.model_class = model_class
        self._weights_path = weights_path

        self._func_preproc = self.model_class.func_pre_process(
        ) if func_preproc is None else func_preproc
        self._func_postproc = self.model_class.func_post_process(
        ) if func_postproc is None else func_postproc

        # self._is_proc_batch = is_proc_batch

        self._preproc_kwargs = (
            {} if preproc_kwargs is None else preproc_kwargs)

        self._postproc_kwargs = (
            {} if postproc_kwargs is None else postproc_kwargs)

        self.kwargs = kwargs

        # if not isinstance(keys_preproc, tuple):
        #     if isinstance(keys_preproc, list):
        #         self._keys_preproc = tuple(keys_preproc)
        #     elif isinstance(keys_preproc, str):
        #         self._keys_preproc = (keys_preproc,)
        #     elif keys_preproc is None:
        #         self._keys_preproc = tuple()
        # else:
        #     self._keys_preproc = keys_preproc

    def get_dict(self):
        return {"model_name": self.model_name,
                "model_class": self.model_class,
                "weights_path": self.weights_path,
                "func_preproc": self.func_preproc,
                "func_postproc": self.func_postproc,
                "is_proc_batch": self.is_proc_batch,
                "preproc_kwargs": self.preproc_kwargs,
                "postproc_kwargs": self.postproc_kwargs,
                "kwargs": "kwargs"
                }

    @property
    def model_name(self):
        return self._model_name

    @property
    def weights_path(self):
        return self._weights_path

    @property
    def func_preproc(self):
        return self._func_preproc

    @property
    def func_postproc(self):
        return self._func_postproc

    # @property
    # def is_proc_batch(self):
    #     return self._is_proc_batch

    @property
    def preproc_kwargs(self):
        return self._preproc_kwargs

    @property
    def postproc_kwargs(self):
        return self._postproc_kwargs


class ModelGroup(object):
    def __init__(self, models: dict = None) -> None:
        super().__init__()
        self.models = {}

        self.append(models)

    def append(self, models: dict = None):
        if isinstance(models, dict):
            self.models.update(models)
        elif isinstance(models, list):
            self.models.update(
                {"model_{}".format(n+1): m for n, m in enumerate(models)})
        elif isinstance(models, BaseNet):
            self.models.update({"model_1": models})
