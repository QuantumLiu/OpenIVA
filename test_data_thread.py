from threads import ThreadDATA
from queue import Queue

import cv2

from commons.vidcoding import decode_video_batch_local

nb_ths=4
nb_tasks=6

q_task=Queue(100)
q_compute=Queue(100)

def prepro_func(data,src_size):
    data=cv2.resize(data,(400,400))
    return data/255

model_configs={"test":{"key_data":"test","func_pre_proc":prepro_func,"keys_prepro":["src_size"],}}
data_gen_keys=["video_path"]
data_gen_kwargs={"skip":1,}

ths_data=[ThreadDATA(q_task,q_compute,decode_video_batch_local,model_configs,8,data_gen_keys,data_gen_kwargs) for _ in range(nb_ths)]
for th_data in ths_data:
    th_data.start()

for task_id in range(nb_tasks):
    print("Putting task: {}".format(task_id))
    q_task.put({"video_path":"datas/videos_test/dmkj_clip.mp4","task_id":task_id})

nb_done=0
while nb_done<nb_tasks:
    data_batch=q_compute.get()
    nb_done+=int(data_batch["flag_end"])

for th_data in ths_data:
    th_data.stop()
quit()