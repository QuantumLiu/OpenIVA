from threading import Thread
from queue import Queue
import time
import uuid

class ThreadDATA(Thread):
    def __init__(self,q_task,q_compute,model_configs,batch_size=8):
        super().__init__()

        self.q_task,self.q_compute=q_task,q_compute
        self.batch_size=batch_size
        self.model_configs=model_configs

    def _setup(self):
        pass