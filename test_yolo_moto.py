import os
import traceback

import cv2
import time
import numpy as np
from tqdm import tqdm

import onnxruntime

from openiva.models.yolov4 import YOLOV4
from openiva.models.yolov4.utils import plot_boxes_cv2, crop_boxes

from openiva.commons.io import get_img_pathes_recursively
from openiva.workers import ThreadImgsLocal

batch_size = 1

so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

yolo = YOLOV4("weights\yolov4_-1_3_416_416_dynamic.onnx",
              input_size=(416, 416), sessionOptions=so, providers="cuda")

pathes_imgs = get_img_pathes_recursively(r"E:\DATA\moto_data\imagesets\0505")
out_dir = r"E:\DATA\moto_data\imagesets_crop\0505"
os.makedirs(out_dir, exist_ok=True)

for path_img in tqdm(pathes_imgs):
    try:
        img = cv2.imread(path_img)

        bboxes_batch, cls_confs_batch, cls_ids_batch = yolo.predict(img)

        crops = crop_boxes(img, bboxes_batch[0], cls_confs=cls_confs_batch[0],
                           cls_ids=cls_ids_batch[0], class_names=["Motorcycle"])

        for i, crop in enumerate(crops["Motorcycle"]):
            fn = os.path.basename(path_img)
            img_patch = crop[0]
            path_out = os.path.join(out_dir, os.path.splitext(
                fn)[0]+"_Motorcycle_"+str(i)+os.path.splitext(fn)[-1])
            cv2.imwrite(path_out, img_patch)
    except:
        traceback.print_exc()
        print(path_img)
        pass


# plot_boxes_cv2(img,bboxes_batch[0],cls_confs=cls_confs_batch[0],cls_ids=cls_ids_batch[0],class_names=["Motorcycle"],savename="datas/imgs_results/vis_moto.jpg")
