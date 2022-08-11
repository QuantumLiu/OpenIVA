import numpy as np
import cv2

from openiva.models.base import BaseNetNew
from openiva.engines import EngineORT

from .utils import get_transform_mat, warp_img, l2_norm, INDS_68_5, MEAN_PTS_5

__all__ = ["LandmarksExtractor", "pre_process", "post_process"]


class ArcFace(BaseNetNew):
    ENGINE_CLASS = EngineORT

    def __init__(self, onnx_path, input_size=(112, 112), channel_first=False, sessionOptions=None, providers="cpu"):
        super().__init__(onnx_path, sessionOptions=sessionOptions, providers=providers)

        self.input_size = input_size
        self.channel_first = channel_first

    def pre_process(self, data):
        """
        Returns pre-processed ndarray (n,h,w,c).
        Args:
            data: raw data ndarray (n,h,w,c) 
        Returns:
            pre-processed ndarray (n,h,w,c)
        """
        return pre_process(data)

    def post_process(self, data):
        """
        Returns results (68,2).
        Args:
            outputs: (136,2) Net outputs ndarrays 
        Returns:
            results: landmarks ndarrays (68,2)
        """

        return post_process(data)

    def predict_single(self, image, landmarks):
        return self.predict({"batch_images": [image], "batch_landms": [landmarks]})[0]

    @classmethod
    def func_pre_process(cls):
        return pre_process

    @classmethod
    def func_post_process(cls):
        return post_process


def post_process(data):
    outputs = data["outputs"][0]
    embs_all = l2_norm(outputs)

    batch_landms = data["batch_landms"]

    batch_embs = []
    n = 0
    for landms in batch_landms:
        embs = []
        for _ in landms:
            embs.append(embs_all[n])
        batch_embs.append(embs)

    return batch_embs


def pre_process(data):
    batch_images = data["batch_images"]
    batch_landms = data["batch_landms"]

    face_imgs = []
    for img, landms in zip(batch_images, batch_landms):
        for landm in landms:
            face_imgs.append(_pre_proc_frame(img, landm))
    return np.ascontiguousarray(face_imgs)


def _pre_proc_frame(img, landm):
    if not landm is None:
        mat = get_transform_mat(
            landm[INDS_68_5].reshape(5, 2), MEAN_PTS_5, 112)
        face_img = warp_img(mat, img, (112, 112)).astype(np.float32)/255.0
        # face_imgs.append(face_img)
    else:
        face_img = cv2.resize(
            img, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    return face_img
