from threading import Thread, Event
from queue import Queue,Empty

import traceback

import time
import uuid

from openiva.commons.videocoding import decode_video_batch_local
from openiva.commons.generators import read_images_local

class ThreadDATA(Thread):
    def __init__(self,q_task:Queue,q_compute:Queue,\
                model_configs:dict,\
                data_gen_func,batch_size:int,\
                data_gen_keys:list,data_gen_kwargs:dict,\
                keys_data:list=None,
                key_batch_data:str="batch_images"):
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
            @param model_configs: dict, 
                functional programming interface, configure pre-processing functions for each model and parameters keys,
                         for example:
                            {"yolo":{"key_data":"yolo","func_pre_proc":=func_yolo,"keys_prepro":["width","height"]},
                            "resnet50":{"key_data":"resnet50","func_pre_proc":fun_resnet}}
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
        self._stop_event = Event()

        self.q_task,self.q_compute=q_task,q_compute
        self.batch_size=batch_size
        self.model_configs=model_configs
        self.data_gen_keys=data_gen_keys
        self.data_gen_kwargs=data_gen_kwargs

        self.keys_data=["batch_images","batch_frames","batch_indecies","batch_src_size","flag_start","flag_end"]

        self.key_batch_data=key_batch_data

        if isinstance(keys_data, (list,tuple)):
            for k in keys_data:
                self.keys_data.append(k)

        self._data_gen_func=data_gen_func

    def stop(self):
        '''
        Stop the thread
        '''
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        if not callable(self._data_gen_func):
            raise NotImplementedError("Please define the data generator function self._data_gen_func")

        while True:
            try:
                try:
                    q_dict_task=self.q_task.get(timeout=1.)
                except Empty:
                    if self.stopped:
                        return
                    continue

                # data_gen_kwargs={}
                data_gen_kwargs=(self.data_gen_kwargs).copy()
                for k in self.data_gen_keys:
                    if k in q_dict_task:
                        data_gen_kwargs[k]=q_dict_task[k]

                data_gen_kwargs["batch_size"]=self.batch_size

                gen=self._data_gen_func(**data_gen_kwargs)
                for data_dict_batch in gen:
                    q_dict_out={'task_id':q_dict_task['task_id']}

                    batch_data=data_dict_batch[self.key_batch_data]

                    for model_name,configs in self.model_configs.items():
                        prepro_kwargs={k:data_dict_batch.get(k,None) for k in configs.get("keys_prepro",[])}
                        prepro_kwargs.update(configs.get("prepro_kwargs",{}))

                        q_dict_out[configs['key_data']]=[configs['func_pre_proc'](frame,**prepro_kwargs) for frame in batch_data]
                    
                    for k in self.keys_data:
                        q_dict_out[k]=data_dict_batch.get(k,None)
                    
                    self.q_compute.put(q_dict_out)
                        

            except KeyboardInterrupt:
                return

            except:
                traceback.print_exc()
                continue


class ThreadVideoLocal(ThreadDATA):
    def __init__(self, q_task: Queue, q_compute: Queue, model_configs: dict, batch_size: int =8, skip :int =1):
        data_gen_keys=["video_path","skip"]
        data_gen_kwargs={"batch_size":batch_size,"skip":skip}
        super().__init__(q_task, q_compute, model_configs, decode_video_batch_local, batch_size, data_gen_keys, data_gen_kwargs,key_batch_data="batch_frames")

class ThreadImgsLocal(ThreadDATA):
    def __init__(self, q_task: Queue, q_compute: Queue, model_configs: dict, batch_size: int =8,shuffle: bool=False):
        data_gen_keys=["pathes_imgs","shuffle"]
        data_gen_kwargs={"batch_size":batch_size,"shuffle":shuffle}
        super().__init__(q_task, q_compute, model_configs, read_images_local, batch_size, data_gen_keys, data_gen_kwargs,key_batch_data="batch_images")

