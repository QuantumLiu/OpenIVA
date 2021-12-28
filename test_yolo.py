import os

import cv2
import time
import numpy as np
from tqdm import tqdm

import onnxruntime

from openiva.models.yolox import YOLOX
from openiva.models.yolox.utils import vis,COCO_CLASSES

batch_size=8

so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

yolo=YOLOX("weights/yolox_m_sim.onnx",input_size=(640,640),sessionOptions=so,providers="cuda")

img=cv2.imread("datas/imgs_test/dog.jpg")



print("Warm up\n")
for _ in tqdm(range(10)):
    boxes_batch,scores_batch,cls_batch=yolo.predict([img]*batch_size)

t_s=time.time()
for _ in tqdm(range(100)):
    boxes_batch,scores_batch,cls_batch=yolo.predict([img]*batch_size)

n_obj=len(boxes_batch[0][:4])
t_e=time.time()
time_batch_det=(t_e-t_s)/100
time_frame_det=time_batch_det/batch_size
time_obj_det=time_frame_det/n_obj

print('Time obj detect cost for batchsize {}, {} objs:{:6f}  \nper frame : {:6f}\nFPS:{:2f}\nObjects per sec: {}\n'.format(\
    batch_size, n_obj, time_batch_det, time_frame_det, 1/time_frame_det, 1/time_obj_det))


boxes,scores,cls_inds=boxes_batch[0],scores_batch[0],cls_batch[0]

origin_img = vis(img, boxes, scores, cls_inds,
                    conf=0.3, class_names=COCO_CLASSES)
cv2.imwrite("datas/imgs_results/vis_dog.jpg", origin_img)
