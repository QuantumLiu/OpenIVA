from threading import Thread, Event
from queue import Queue,Empty
import time
import uuid


class ThreadDATA(Thread):
    def __init__(self,q_task:Queue,q_compute:Queue,data_gen_func,model_configs:dict,batch_size:int,data_gen_keys:list,data_gen_kwargs:dict):
        super().__init__()
        self._stop_event = Event()

        self.q_task,self.q_compute=q_task,q_compute
        self.batch_size=batch_size
        self.model_configs=model_configs
        self.data_gen_keys=data_gen_keys
        self.data_gen_kwargs=data_gen_kwargs

        self.keys_data=["src_size","batch_frames","batch_indecies","flag_start","flag_end"]

        self._data_gen_func=data_gen_func

    def stop(self):
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        if not callable(self._data_gen_func):
            raise NotImplementedError("Please define the data generator function self._data_gen_func")

        while True:
            try:
                q_dict_task=self.q_task.get(timeout=1.)
            except Empty:
                if self.stopped:
                    return
            except KeyboardInterrupt:
                return

            # data_gen_kwargs={}
            data_gen_kwargs=(self.data_gen_kwargs).copy()
            for k in self.data_gen_keys:
                if k in q_dict_task:
                    data_gen_kwargs[k]=q_dict_task[k]

            data_gen_kwargs["batch_size"]=self.batch_size

            gen=self._data_gen_func(**data_gen_kwargs)
            for data_dict_batch in gen:
                q_dict_out={'task_id':q_dict_task['task_id']}

                batch_frames=data_dict_batch["batch_frames"]

                for model_name,configs in self.model_configs.items():
                    prepro_kwargs={k:data_dict_batch.get(k,None) for k in configs["keys_prepro"]}
                    prepro_kwargs.update(configs.get("prepro_kwargs",{}))

                    q_dict_out[configs['key_data']]=[configs['func_pre_proc'](frame,**prepro_kwargs) for frame in batch_frames]
                
                for k in self.keys_data:
                    q_dict_out[k]=data_dict_batch[k]
                
                self.q_compute.put(q_dict_out)
                    



