from .base import BaseNet


class ModelDataConfig(object):

    @classmethod
    def from_dict(cls,config_dict):
        return cls(**config_dict)

    def __init__(self,model_name:str,func_preproc:callable,keys_preproc:tuple=None,preproc_kwargs:dict=None,is_proc_batch=False) -> None:
        super().__init__()
        self._model_name=model_name
        self._func_preproc=func_preproc
        self._preproc_kwargs=({} if preproc_kwargs is None else preproc_kwargs)
        self._is_proc_batch=is_proc_batch


        if not isinstance(keys_preproc,tuple):
            if isinstance(keys_preproc,list):
                self._keys_preproc=tuple(keys_preproc)
            elif isinstance(keys_preproc,str):
                self._keys_preproc=(keys_preproc,)
            elif keys_preproc is None:
                self._keys_preproc=tuple()
        else:
            self._keys_preproc=keys_preproc
        

    def get_dict(self):
        return {"model_name":self.model_name,
                "func_preproc":self.func_preproc,
                "keys_preproc":self.keys_preproc,
                "is_proc_batch":self.is_proc_batch}

    @property
    def model_name(self):
        return self._model_name



    @property
    def func_preproc(self):
        return self._func_preproc

    @property
    def keys_preproc(self):
        return self._keys_preproc

    @property
    def preproc_kwargs(self):
        return self._preproc_kwargs

    @property
    def is_proc_batch(self):
        return self._is_proc_batch

class ModelGroup(object):
    def __init__(self,models:dict=None) -> None:
        super().__init__()
        self.models={}

        self.append(models)

    def append(self,models:dict=None):
        if isinstance(models,dict):
            self.models.update(models)
        elif isinstance(models,list):
            self.models.update({"model_{}".format(n+1):m for n,m in enumerate(models)})
        elif isinstance(models,BaseNet):
            self.models.update({"model_1":models})

