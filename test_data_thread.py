from openiva.workers import ThreadDATA
from queue import Queue

import cv2

from openiva.commons.videocoding import decode_video_batch_local
from openiva.models.config import ModelDataConfig


if __name__ == "__main__":

    nb_ths=4
    nb_tasks=6

    q_task=Queue(100)
    q_compute=Queue(100)

    def prepro_func(data,w,h):
        data=cv2.resize(data,(640,640))
        return data/255

    model_configs=(ModelDataConfig(model_name="yolo_test",key_data="test",func_pre_proc=prepro_func,prepro_kwargs={"w":640,"h":640}),
                    ModelDataConfig.from_dict({"model_name":"yolo_test2","key_data":"test2","func_pre_proc":prepro_func,"prepro_kwargs":{"w":640,"h":640}}))
    data_gen_keys=["video_path"]
    data_gen_kwargs={"skip":1,}

    ths_data=[ThreadDATA(q_task,q_compute,model_configs,decode_video_batch_local,8,data_gen_keys,data_gen_kwargs,key_batch_data="batch_frames") for _ in range(nb_ths)]
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