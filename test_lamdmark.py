
import cv2
import time
import numpy as np
from tqdm import tqdm

import onnxruntime

from models.detector import Detector
from models.alignment import LandmarksExtractor

img=cv2.imread("sample.jpg")

so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
# so.intra_op_num_threads = 12
# so.device_type = "CPU_FP32"
batch_size=1

detector=Detector("weights/face_detector_320_dy_sim.onnx",providers="openvino",sessionOptions=so,input_size=(320,240),top_k=5)
lm_extractor=LandmarksExtractor("weights/landmarks_68_pfld_sim.onnx",sessionOptions=so,providers="openvino")
rectangles_batch, probes_batch=detector.predict([img]*batch_size)
landmarks = lm_extractor.predict(img, rectangles_batch[0])

print("Warm up\n")
for _ in tqdm(range(100)):
    rectangles_batch, probes_batch=detector.predict([img]*batch_size)


t_s=time.time()
for _ in tqdm(range(1000)):
    rectangles_batch, probes_batch=detector.predict([img]*batch_size)

t_e=time.time()
time_batch=(t_e-t_s)/1000
time_frame=time_batch/batch_size


t_s=time.time()
for _ in tqdm(range(1000)):
    landmarks = lm_extractor.predict(img, [rectangles_batch[0][0]])
t_e=time.time()
time_batch_lm=(t_e-t_s)/1000
time_frame_lm=time_batch_lm#/batch_size

landmarks = lm_extractor.predict(img, rectangles_batch[0])


for rect,score,landmark in zip(rectangles_batch[0],probes_batch[0],landmarks):
    if score<0.5:
        continue
    rect = list(map(int, rect))
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
    cx = rect[0]
    cy = rect[1] + 12
    cv2.putText(img, "{:.4f}".format(score), (cx, cy),\
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    for (x, y) in landmark:
        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 0),-1)

name = "vis_face.jpg"
cv2.imwrite(name, img)
#print('Result:\n{}'.format('\n'.join([name+' '*3+'{:4f}'.format(prob) for name,prob in zip(top_names[0],top_probs[0])])))
print('Time face detect cost for batchsize {} :{:6f}  \nper frame : {:6f}\nFPS:{:2f}'.format(batch_size,time_batch,time_frame,1/time_frame))

print('Time face landmark alignment cost for \nper batch : {:6f}\nFPS:{:2f}'.format(time_batch_lm, 1/time_frame_lm))
