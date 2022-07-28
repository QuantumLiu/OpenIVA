from queue import Queue

import cv2

from openiva.commons.io import get_img_pathes_recursively

from openiva.workers import ThreadImgsLocal
from openiva.models.group import ModelConfig

from openiva.models.yolov4 import YOLOV4

import sys

if __name__ == "__main__":

    nb_ths = 6
    nb_tasks = 6

    q_task = Queue(100)
    q_compute = Queue(100)

    input_size = (416, 416)
    model_configs = (ModelConfig("yolo_test", YOLOV4, weights_path="weights\yolov4_1_3_416_416_static.onnx",
                                 preproc_kwargs={"input_size": input_size},
                                 postproc_kwargs={"input_size": input_size}),)

    ths_data = [ThreadImgsLocal(
        q_task, q_compute, model_configs, batch_size=8, shuffle=True) for _ in range(nb_ths)]
    for th_data in ths_data:
        th_data.setDaemon(1)
        th_data.start()

    fns_img = get_img_pathes_recursively(r"E:\DATA\moto_data\imagesets")
    for task_id in range(nb_tasks):
        print("Putting task: {}".format(task_id))
        q_task.put({"pathes_imgs": fns_img, "task_id": task_id})

    nb_done = 0
    while nb_done < nb_tasks:
        data_batch = q_compute.get()
        if data_batch["flag_end"]:
            nb_done += int(data_batch["flag_end"])

    # for th_data in ths_data:
    #     th_data.stop()
    # quit()
