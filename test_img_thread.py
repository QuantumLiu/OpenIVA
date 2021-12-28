from openiva.commons.generators import read_images_local
from openiva.workers import ThreadImgsLocal
from queue import Queue

from openiva.commons.io import get_img_pathes_recursively

import cv2


nb_ths=1
nb_tasks=6

q_task=Queue(100)
q_compute=Queue(100)

def prepro_func(data,width=None,height=None):
    data=cv2.resize(data,(width,height))
    return data/255

model_configs={"yolo_test":{"key_data":"test","func_pre_proc":prepro_func,"prepro_kwargs":{"width":640,"height":640}}}

ths_data=[ThreadImgsLocal(q_task,q_compute,model_configs,batch_size=8,shuffle=True) for _ in range(nb_ths)]
for th_data in ths_data:
    th_data.start()

fns_img=get_img_pathes_recursively("datas/imgs_celebrity")
for task_id in range(nb_tasks):
    print("Putting task: {}".format(task_id))
    q_task.put({"pathes_imgs":fns_img,"task_id":task_id})

nb_done=0
while nb_done<nb_tasks:
    data_batch=q_compute.get()
    if data_batch["flag_end"]:
        nb_done+=int(data_batch["flag_end"])

for th_data in ths_data:
    th_data.stop()
# quit()