from .import StoppableThread
from queue import Queue, Empty

import traceback

import time
import uuid

from openiva.commons.videocoding import decode_video_batch_local
from openiva.commons.generators import read_images_local


class ThreadProc(StoppableThread):
    def __init__(self, q_task: Queue, q_compute: Queue,
                 model_configs: tuple,
                 key_data: list = None,
                 key_batch_data: str = "batch_images"):
        '''
            Basic class for data loading and processing threads. 
            Waiting for the tasks in a loop from input `Queue`, read data via a `generator`, and process them by multiple pre-processing functions of models,
        finally put processed batch datas in output `Queue`.
            Functional programming, arguments of data generator and processing are defined and passed by `kwargs`.
        You can just write your own data generator for loading different types of data, and pass it as an argument.
        args:
            @param q_task: Queue, 
                the Thread loop and try to get task(dictionary) from it.
            @param q_compute: Queue, 
                the queue connected to the computing Thread, put result datas(dictionary) in it.
            @param model_configs: tuple, ModelDataConfig objects
                functional programming interface, configure pre-processing functions for each model and parameters keys,
                         for example:
                             {'model_name': 'yolo',
                            'key_data': ('batch_images',),
                            'func_preproc': <function __main__.<func_yolo>(x)>,
                            'keys_preproc': ('width','height'),
                            'is_proc_batch': True}

                        In which `func_yolo` and `fun_resnet` are two functions
            @param data_gen_func: function, 
                to start a data generator
            @param batch_size: int, 
                argument of `data_gen_func`
            @param data_gen_keys: list, 
                a list of string, parameters keys of `data_gen_func`
            @param data_gen_kwargs: dict, 
                arguments of `data_gen_func`
        '''
        super().__init__()

        self.q_task, self.q_compute = q_task, q_compute
        self.model_configs = model_configs

        self.key_data = ["batch_images", "batch_frames",
                         "batch_indecies", "batch_src_size", "flag_start", "flag_end"]

        self.key_batch_data = key_batch_data

        if isinstance(key_data, (list, tuple)):
            for k in key_data:
                self.key_data.append(k)

    def _apply_proc(self, task_id, data_dict_batch):
        q_dict_out = {'task_id': task_id}
        batch_data = data_dict_batch  # [self.key_batch_data]

        for model_config in self.model_configs:
            preproc_kwargs = model_config.preproc_kwargs

            q_dict_out[model_config.model_name] = {}
            if True:  # model_config.is_proc_batch:
                q_dict_out[model_config.model_name]["data_preproc"] = model_config.func_preproc(
                    batch_data, **preproc_kwargs)
            # else:
            #     q_dict_out[model_config.model_name] = [model_config.func_preproc(
            #         frame, **preproc_kwargs) for frame in batch_data]

        for k in data_dict_batch.keys():
            if k not in q_dict_out:
                q_dict_out[k] = data_dict_batch.get(k, None)

        self.q_compute.put(q_dict_out)

    def run(self):

        while True:
            try:
                try:
                    data_dict_batch = self.q_task.get(timeout=1.)
                    task_id = data_dict_batch["task_id"]
                except Empty:
                    continue

                self._apply_proc(task_id, data_dict_batch)

            except KeyboardInterrupt:
                return

            except:
                traceback.print_exc()
                continue


class ThreadDATA(ThreadProc):
    def __init__(self, q_task: Queue, q_compute: Queue,
                 model_configs: tuple,
                 data_gen_func, batch_size: int,
                 data_gen_keys: list, data_gen_kwargs: dict,
                 key_data: list = None,
                 key_batch_data: str = "batch_images"):
        '''
            Basic class for data loading and processing threads. 
            Waiting for the tasks in a loop from input `Queue`, read data via a `generator`, and process them by multiple pre-processing functions of models,
        finally put processed batch datas in output `Queue`.
            Functional programming, arguments of data generator and processing are defined and passed by `kwargs`.
        You can just write your own data generator for loading different types of data, and pass it as an argument.
        args:
            @param q_task: Queue, 
                the Thread loop and try to get task(dictionary) from it.
            @param q_compute: Queue, 
                the queue connected to the computing Thread, put result datas(dictionary) in it.
            @param model_configs: tuple, ModelDataConfig objects
                functional programming interface, configure pre-processing functions for each model and parameters keys,
                         for example:
                             {'model_name': 'yolo',
                            'key_data': ('batch_images',),
                            'func_preproc': <function __main__.<func_yolo>(x)>,
                            'keys_preproc': ('width','height'),
                            'is_proc_batch': True}

                        In which `func_yolo` and `fun_resnet` are two functions
            @param data_gen_func: function, 
                to start a data generator
            @param batch_size: int, 
                argument of `data_gen_func`
            @param data_gen_keys: list, 
                a list of string, parameters keys of `data_gen_func`
            @param data_gen_kwargs: dict, 
                arguments of `data_gen_func`
        '''

        super().__init__(q_task, q_compute, model_configs,
                         key_data=key_data, key_batch_data=key_batch_data)

        self.batch_size = batch_size
        self.data_gen_keys = data_gen_keys
        self.data_gen_kwargs = data_gen_kwargs

        self.key_batch_data = key_batch_data

        self._data_gen_func = data_gen_func

    def run(self):
        if not callable(self._data_gen_func):
            raise NotImplementedError(
                "Please define the data generator function self._data_gen_func")

        while True:
            try:
                try:
                    q_dict_task = self.q_task.get(timeout=1.)
                    task_id = q_dict_task["task_id"]
                except Empty:
                    continue

                data_gen_kwargs = (self.data_gen_kwargs).copy()
                for k in self.data_gen_keys:
                    if k in q_dict_task:
                        data_gen_kwargs[k] = q_dict_task[k]

                data_gen_kwargs["batch_size"] = self.batch_size

                gen = self._data_gen_func(**data_gen_kwargs)
                for data_dict_batch in gen:
                    self._apply_proc(task_id, data_dict_batch)

            except KeyboardInterrupt:
                return


class ThreadVideoLocal(ThreadDATA):
    def __init__(self, q_task: Queue, q_compute: Queue, model_configs: tuple, batch_size: int = 8, skip: int = 1):
        data_gen_keys = ["video_path", "skip"]
        data_gen_kwargs = {"batch_size": batch_size, "skip": skip}
        super().__init__(q_task, q_compute, model_configs, decode_video_batch_local,
                         batch_size, data_gen_keys, data_gen_kwargs, key_batch_data="batch_frames")


class ThreadImgsLocal(ThreadDATA):
    def __init__(self, q_task: Queue, q_compute: Queue, model_configs: tuple, batch_size: int = 8, shuffle: bool = False):
        data_gen_keys = ["pathes_imgs", "shuffle"]
        data_gen_kwargs = {"batch_size": batch_size, "shuffle": shuffle}
        super().__init__(q_task, q_compute, model_configs, read_images_local,
                         batch_size, data_gen_keys, data_gen_kwargs, key_batch_data="batch_images")
