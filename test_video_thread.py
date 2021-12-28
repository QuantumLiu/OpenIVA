from openiva.workers import ThreadVideoLocal
from queue import Queue

import cv2


nb_ths=4
nb_tasks=6

q_task=Queue(100)
q_compute=Queue(100)

def prepro_func(data,width=None,height=None):
    data=cv2.resize(data,(640,640))
    return data/255

model_configs={"yolo_test":{"key_data":"test","func_pre_proc":prepro_func,"prepro_kwargs":{"width":640,"height":640},}}
data_gen_keys=["video_path"]
data_gen_kwargs={"skip":1,}

ths_data=[ThreadVideoLocal(q_task,q_compute,model_configs,batch_size=8,skip=1) for _ in range(nb_ths)]
for th_data in ths_data:
    th_data.start()

for task_id in range(nb_tasks):
    print("Putting task: {}".format(task_id))
    q_task.put({"video_path":"datas/videos_test/inception_clip.mp4","task_id":task_id})

nb_done=0
while nb_done<nb_tasks:
    data_batch=q_compute.get()
    if data_batch["flag_end"]:
        nb_done+=int(data_batch["flag_end"])

for th_data in ths_data:
    th_data.stop()
# quit()