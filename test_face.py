import os

import cv2
import time
import numpy as np
from tqdm import tqdm

import onnxruntime

from models.detector import Detector
from models.alignment import LandmarksExtractor
from models.arcface import ArcFace
from models.arcface.utils import l2_norm,face_distance,sub_feature



so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

detector=Detector("weights/face_detector_640_dy_sim.onnx",providers="cuda",sessionOptions=so,input_size=(640,480),top_k=16)
lm_extractor=LandmarksExtractor("weights/landmarks_68_pfld_dy_sim.onnx",sessionOptions=so,providers="cuda")
arcface=ArcFace("weights/arc_mbv2_ccrop_sim.onnx",sessionOptions=so,providers="cuda")


dir_info="datas/imgs_celebrity/姜文_1021999_m"

name="姜文"

imgs=[]
pathes_imgs=[]
for fn in os.listdir(dir_info):
    path_img=os.path.join(dir_info,fn)
    # print("Reading img {}".format(path_img))
    img=cv2.imread(path_img)
    imgs.append(img)
    pathes_imgs.append(path_img)

rectangles_batch, probes_batch=detector.predict(imgs)

person_embs=[]
for rectangles,img,path_img in zip(rectangles_batch,imgs,pathes_imgs):
    if not len(rectangles)==1:
        print("No face or more than 1 face in img {}, PASS".format(path_img))
    lm = lm_extractor.predict(img, rectangles)[0]
    feature_vec=arcface.predict(img,[lm])[0]
    person_embs.append(feature_vec)

print("Testing performance....")
batchsize=8

print("Warm up")
for _ in tqdm(range(10)):
    _=arcface.predict(img,[lm]*batchsize)[0]

t_s=time.time()
for _ in tqdm(range(1000)):
    _=arcface.predict(img,[lm]*batchsize)[0]
t_e=time.time()
time_batch_emb=(t_e-t_s)/1000
time_face_emb=time_batch_emb/batchsize
print("Time face embedding for batchsize 8: {}\nFaces per sec: {}".format(time_batch_emb,1/time_face_emb))


sub_feature_list,mean_feature=sub_feature(person_embs)

img_wild=cv2.imread("datas/imgs_celebrity/wild.jpg")

rectangles_batch, probes_batch=detector.predict([img_wild])

n_faces=len(rectangles_batch[-1])
print("{} faces detected".format(n_faces))

landmarks = lm_extractor.predict(img_wild, rectangles_batch[0])
features_array=arcface.predict(img_wild,landmarks)

dists=face_distance(features_array,mean_feature)


for rect,score,landmark in zip(rectangles_batch[0],dists,landmarks):
    rect = list(map(int, rect))
    cv2.rectangle(img_wild, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
    cx = rect[0]
    cy = rect[1] + 12
    cv2.putText(img_wild, "{:.4f}".format(score), (cx, cy),\
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    if score>0.7:
        cv2.putText(img_wild, "The 9 Dot", (cx, cy-18),\
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

    # for (x, y) in landmark:
    #     cv2.circle(img_wild, (int(x), int(y)), 1, (255, 255, 0),-1)

name = "datas/imgs_results/vis_recog.jpg"
cv2.imwrite(name, img_wild)
