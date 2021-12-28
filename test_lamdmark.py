
import cv2
import time
import numpy as np
from tqdm import tqdm

import onnxruntime

from openiva.models.detector import Detector
from openiva.models.alignment import LandmarksExtractor

if __name__ == "__main__":
    img=cv2.imread("datas/imgs_test/lumia.jpg")

    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    batch_size=8

    detector=Detector("weights/face_detector_640_dy_sim.onnx",providers="cuda",sessionOptions=so,input_size=(640,480),top_k=68)
    lm_extractor=LandmarksExtractor("weights/landmarks_68_pfld_dy_sim.onnx",sessionOptions=so,providers="cuda")
    rectangles_batch, probes_batch=detector.predict([img]*batch_size)
    landmarks = lm_extractor.predict(img, rectangles_batch[-1])

    n_faces=len(rectangles_batch[-1])
    print("{} faces detected".format(n_faces))

    print("Warm up\n")
    for _ in tqdm(range(10)):
        rectangles_batch, probes_batch=detector.predict([img]*batch_size)


    t_s=time.time()
    for _ in tqdm(range(100)):
        rectangles_batch, probes_batch=detector.predict([img]*batch_size)

    t_e=time.time()
    time_batch_det=(t_e-t_s)/100
    time_frame_det=time_batch_det/batch_size
    time_face_det=time_frame_det/n_faces


    t_s=time.time()
    for _ in tqdm(range(100)):
        landmarks = lm_extractor.predict(img, rectangles_batch[-1])
    t_e=time.time()
    time_batch_lm=(t_e-t_s)/100
    # time_frame_lm=time_batch_lm/batch_size
    time_face_lm=time_batch_lm/n_faces

    landmarks = lm_extractor.predict(img, rectangles_batch[-1])


    for rect,score,landmark in zip(rectangles_batch[-1],probes_batch[-1],landmarks):
        rect = list(map(int, rect))
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
        cx = rect[0]
        cy = rect[1] + 12
        cv2.putText(img, "{:.4f}".format(score), (cx, cy),\
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        for (x, y) in landmark:
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 0),-1)

    name = "datas/imgs_results/vis_landmark.jpg"
    cv2.imwrite(name, img)
    #print('Result:\n{}'.format('\n'.join([name+' '*3+'{:4f}'.format(prob) for name,prob in zip(top_names[0],top_probs[0])])))
    print('Time face detect cost for batchsize {}, {} faces:{:6f}  \nper frame : {:6f}\nFPS:{:2f}\nFaces per sec: {}\n'.format(\
        batch_size, n_faces, time_batch_det, time_frame_det, 1/time_frame_det, 1/time_face_det))

    print('Time face landmark alignment cost for {} faces\nper batch : {:6f}\nFaces per sec: {}\n'.format(\
        n_faces,time_batch_lm, 1/time_face_lm))
