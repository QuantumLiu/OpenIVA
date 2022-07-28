import json
import os
import traceback

import cv2
import time
import numpy as np

import h5py

from tqdm import tqdm

import onnxruntime
from openiva.commons import facial

from openiva.models.detector import Detector
from openiva.models.alignment import LandmarksExtractor
from openiva.models.arcface import ArcFace
from openiva.models.arcface.utils import l2_norm, face_distance, sub_feature

from openiva.commons.facial import FacialInfo, FacialDB, parse_filename, remove_old

from openiva.commons.io import imread


def worker_dir(path_dir, detector: Detector, lm_extractor: LandmarksExtractor, arcface: ArcFace, over_write=False):
    facial_info = FacialInfo()

    print('Registing dir:{}'.format(path_dir))
    info_path = os.path.join(path_dir, 'info.json')

    empty_flag_path = os.path.join(path_dir, 'empty.flag')
    nb_imgs = 0
    name_celeba, id_douban, gender_id = parse_filename(path_dir)

    # and not (os.path.exists(empty_flag_path)):
    if ((not os.path.exists(info_path)) or over_write):
        if os.path.exists(info_path):
            os.remove(info_path)
        path_list = [os.path.join(path_dir, fn) for fn in os.listdir(path_dir)]

        feature_list = []
        for path in path_list:
            try:
                if path[-4:] in ['flag', 'json']:
                    continue
                # print('Working on img:{}'.format(path))
                img_src = imread(path)
                rectangles_batch, probes_batch = detector.predict([img_src])
                rectangles = rectangles_batch[0]

                if not len(rectangles) == 1:
                    # print("More than 1 face or no face in img, PASS")
                    continue

                lm = lm_extractor.predict_single(img_src, rectangles)[0]
                feature_vec = arcface.predict_single(img_src, [lm])[0]
                feature_list.append(feature_vec)
                nb_imgs += 1

            except [KeyboardInterrupt, IndexError]:
                traceback.print_exc()
                quit()

            except:
                traceback.print_exc()
                continue

        if len(feature_list):
            feature_list = np.asarray(feature_list)
            mean_feature = np.mean(feature_list, axis=0)
            #print(path_dir+' nb feature:{}'.format(len(feature_list)))
            feature_list, mean_feature = sub_feature(feature_list)
            print("Computed facial feature vector from {} images from directory {}, got {} faces.\nActor's name is {}.".format(
                nb_imgs, path_dir, len(feature_list), name_celeba))

            info_dict = {"name": name_celeba, "id": id_douban, "gender_id": gender_id,
                         "feature_vector": mean_feature.tolist(), "feature_list": feature_list.tolist()}
            facial_info.update(info_dict)
            facial_info.save_info(path_json=info_path)

            flag_empty = False
            if os.path.exists(empty_flag_path):
                os.remove(empty_flag_path)
        else:
            flag_empty = True
            with open(empty_flag_path, 'w') as fp:
                fp.write('e')
            mean_feature = None
    elif os.path.exists(empty_flag_path):
        flag_empty = True
        mean_feature = None
    else:
        info_dict = facial_info.load_info(info_path)
        flag_empty = False
        if os.path.exists(empty_flag_path):
            os.remove(empty_flag_path)

    return facial_info.info_dict, flag_empty


def register_all(root_dir, out_path, detector: Detector, lm_extractor: LandmarksExtractor, arcface: ArcFace, use_par=False, over_write=False):
    facial_db = FacialDB()
    if over_write:
        remove_old(root_dir)
    pathes_dir = [path_dir for path_dir in [os.path.join(
        root_dir, name_dir) for name_dir in os.listdir(root_dir)] if os.path.isdir(path_dir)]

    results = []

    print('Registing {} by serial processing'.format(root_dir))
    results = [worker_dir(path_dir, detector, lm_extractor,
                          arcface, over_write) for path_dir in pathes_dir]

    db_dict = {}

    for i, (info_dict, flag_empty) in enumerate(results):
        if not flag_empty:
            db_dict.setdefault(info_dict["id"], info_dict)

    with open(out_path, "w") as fp:
        json.dump(db_dict, fp)

    with h5py.File("debug.h5", "w") as fph5:
        for i, info_dict in db_dict.items():
            grp = fph5.create_group(i)
            for k, v in info_dict.items():
                ds = grp.create_dataset(k, data=v)

    return db_dict


if __name__ == "__main__":
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    detector = Detector("weights/face_detector_640_dy_sim.onnx",
                        providers="cuda", sessionOptions=so, input_size=(640, 480), top_k=16)
    lm_extractor = LandmarksExtractor(
        "weights/landmarks_68_pfld_dy_sim.onnx", sessionOptions=so, providers="cuda")
    arcface = ArcFace("weights/arc_mbv2_ccrop_sim.onnx",
                      sessionOptions=so, providers="cuda")
    import sys
    root_dir = "datas/celebrity_images"
    out_path = "t.json"
    register_all(root_dir, out_path, detector, lm_extractor,
                 arcface, use_par=False, over_write=True)
