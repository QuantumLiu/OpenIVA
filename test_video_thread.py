from queue import Queue


import cv2
from openiva.workers import ThreadVideoLocal
from openiva.workers import ThreadCompute
from openiva.models.group import ModelConfig

from openiva.models.yolov4 import YOLOV4
import onnxruntime

if __name__ == "__main__":

    nb_ths = 4
    nb_tasks = 4

    q_task = Queue(100)
    q_compute = Queue(100)
    q_post = Queue(100)

    input_size = (416, 416)
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    model_configs = (ModelConfig("yolo_test", YOLOV4, weights_path="weights\yolov4_-1_3_416_416_dynamic.onnx",
                                 preproc_kwargs={"input_size": input_size},
                                 postproc_kwargs={"input_size": input_size}, input_size=(416, 416), sessionOptions=so, providers="cuda"),)

    ths_data = [ThreadVideoLocal(
        q_task, q_compute, model_configs, batch_size=8, skip=1) for _ in range(nb_ths)]
    for th_data in ths_data:
        th_data.start()

    th_compute = ThreadCompute(
        q_compute, q_post, model_config=model_configs[0])
    th_compute.start()

    for task_id in range(nb_tasks):
        print("Putting task: {}".format(task_id))
        q_task.put(
            {"video_path": "datas/videos_test/inception_clip.mp4", "task_id": task_id})

    nb_done = 0
    while nb_done < nb_tasks:
        data_batch = q_post.get()
        if data_batch["flag_end"]:
            nb_done += int(data_batch["flag_end"])

    print("{} tasks done, exit".format(nb_done))
