from threads import ThreadDATA
from queue import Queue

import cv2

from commons.vidcoding import decode_video_batch_local
q_task=Queue(100)
q_compute=Queue(100)

def prepro_func(data,src_size):
    data=cv2.resize(data,(400,400))
    return data/255

model_configs={"test":{"key_data":"test","func_pre_proc":prepro_func,"keys_prepro":["src_size"],}}
data_gen_keys=["video_path"]
data_gen_kwargs={"skip":1,}

th_data=ThreadDATA(q_task,q_compute,decode_video_batch_local,model_configs,8,data_gen_keys,data_gen_kwargs)
th_data.start()

q_task.put({"video_path":"datas/videos_test/dmkj_clip.mp4","task_id":1})

data_batch=q_compute.get()
while not data_batch["flag_end"]:
    data_batch=q_compute.get()

th_data.stop()
quit()
