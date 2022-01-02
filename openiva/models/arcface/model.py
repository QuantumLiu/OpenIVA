import numpy as np
import cv2

from openiva.models.base import BaseNet

from .utils import get_transform_mat,warp_img,l2_norm,INDS_68_5,MEAN_PTS_5

__all__ = ["LandmarksExtractor"]

class ArcFace(BaseNet):
    def __init__(self, onnx_path, input_size=(112,112),channel_first=False, sessionOptions=None,providers="cpu"):
        super().__init__(onnx_path, sessionOptions=sessionOptions,providers=providers)

        self.input_size=input_size
        self.channel_first=channel_first

    def predict(self, data,landm=None):
        face_img=self.pre_process(data,landm)
        outputs=self._infer(face_img)
        outputs=self.post_process(outputs)
        return outputs

    @staticmethod
    def pre_process(data_raw,landms=None):
        face_imgs=[]
        if not landms is None and len(landms):
            for landm in landms:
                mat=get_transform_mat(landm[INDS_68_5].reshape(5,2),MEAN_PTS_5,112)
                face_img=warp_img(mat,data_raw,(112,112)).astype(np.float32)/255.0
                face_imgs.append(face_img)
        else:
            face_imgs.append(data_raw.astype(np.float32)/255.0)
        return np.ascontiguousarray(face_imgs)

    @staticmethod
    def post_process(outputs):
        return l2_norm(outputs[0])

    def _pre_proc_frame(self, img,landms):
        face_imgs=[]
        if not landms is None and len(landms):
            for landm in landms:
                mat=get_transform_mat(landm[INDS_68_5].reshape(5,2),MEAN_PTS_5,112)
                face_img=warp_img(mat,img,(112,112)).astype(np.float32)/255.0
                face_imgs.append(face_img)
        else:
            face_imgs.append(img.astype(np.float32)/255.0)
        return np.ascontiguousarray(face_imgs)