class ModelDataConfig(object):

    @classmethod
    def from_dict(cls,config_dict):
        return cls(**config_dict)

    def __init__(self,model_name:str,key_data:str,func_pre_proc:callable,keys_prepro:tuple=None,prepro_kwargs:dict=None,is_proc_batch=False) -> None:
        super().__init__()
        self._model_name=model_name
        self._key_data=key_data
        self._func_pre_proc=func_pre_proc
        self._prepro_kwargs=prepro_kwargs
        self._is_proc_batch=is_proc_batch


        if not isinstance(keys_prepro,tuple):
            if isinstance(keys_prepro,list):
                self._keys_prepro=tuple(keys_prepro)
            elif isinstance(keys_prepro,str):
                self._keys_prepro=(keys_prepro,)
            elif keys_prepro is None:
                self._keys_prepro=tuple()
        else:
            self._keys_prepro=keys_prepro
        

    def get_dict(self):
        return {"model_name":self.model_name,
                "key_data":self.key_data,
                "func_pre_proc":self.func_pre_proc,
                "keys_prepro":self.keys_prepro,
                "is_proc_batch":self.is_proc_batch}

    @property
    def model_name(self):
        return self._model_name

    @property
    def key_data(self):
        return self._key_data


    @property
    def func_pre_proc(self):
        return self._func_pre_proc

    @property
    def keys_prepro(self):
        return self._keys_prepro

    @property
    def prepro_kwargs(self):
        return self._prepro_kwargs

    @property
    def is_proc_batch(self):
        return self._is_proc_batch

