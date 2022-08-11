import os

import cv2
import time
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont

from tqdm import tqdm

import onnxruntime

from openiva.commons.facial.database import FacialDB

from detector import Detector
from alignment import LandmarksExtractor
from arcface import ArcFace

from openiva.tools.register_face import register_all

if __name__ == "__main__":
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    detector = Detector("weights/face_detector_640_dy_sim.onnx",
                        providers="cuda", sessionOptions=so, input_size=(640, 480), top_k=16)
    lm_extractor = LandmarksExtractor(
        "weights/landmarks_68_pfld_dy_sim.onnx", sessionOptions=so, providers="cuda")
    arcface = ArcFace("weights/arc_mbv2_ccrop_sim.onnx",
                      sessionOptions=so, providers="cuda")

    dir_info = "datas/imgs_celebrity"
    path_json_tmp = "t.json"

    db_dict = register_all(dir_info, path_json_tmp,
                           detector, lm_extractor, arcface, over_write=True)
    facial_db = FacialDB(db_dict=db_dict)

    img_wild = cv2.imread("datas/imgs_celebrity/wild.jpg")

    rectangles_batch, probes_batch = detector.predict([img_wild])
    landmarks = lm_extractor.predict_single(img_wild, rectangles_batch[0])

    print("Testing performance....")
    batchsize = 8
    lm = landmarks[0]

    print("Warm up")
    for _ in tqdm(range(10)):
        _ = arcface.predict_single(img_wild, [lm]*batchsize)[0]

    t_s = time.time()
    for _ in tqdm(range(1000)):
        _ = arcface.predict_single(img_wild, [lm]*batchsize)[0]
    t_e = time.time()
    time_batch_emb = (t_e-t_s)/1000
    time_face_emb = time_batch_emb/batchsize
    print("Time face embedding for batchsize 8: {}\nFaces per sec: {}".format(
        time_batch_emb, 1/time_face_emb))

    rectangles_batch, probes_batch = detector.predict([img_wild])

    n_faces = len(rectangles_batch[-1])
    print("{} faces detected".format(n_faces))

    landmarks = lm_extractor.predict_single(img_wild, rectangles_batch[0])
    features_array = arcface.predict_single(img_wild, landmarks)

    inds, knowns, dists_max = facial_db.query_N2N(features_array)

    for rect, ind, known, score, landmark in zip(rectangles_batch[0], inds, knowns, dists_max, landmarks):
        rect = list(map(int, rect))
        cv2.rectangle(img_wild, (rect[0], rect[1]),
                      (rect[2], rect[3]), (0, 0, 255), 2)
        cx = rect[0]
        cy = rect[1] + 12
        cv2.putText(img_wild, "{:.4f}".format(score), (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        if score > 0.7:
            name = facial_db.ind2name[ind]
            image = Image.fromarray(cv2.cvtColor(img_wild, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            fontText = ImageFont.truetype(
                "datas/fonts/SourceHanSansHWSC-Regular.otf", 18, encoding="utf-8")
            draw.text((cx, cy-32), name, font=fontText, fill=(0, 255, 255))
            img_wild = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        for (x, y) in landmark:
            cv2.circle(img_wild, (int(x), int(y)), 3, (255, 255, 0), -1)

    name = "datas/imgs_results/vis_recog.jpg"
    cv2.imwrite(name, img_wild)
