# import sys
# import os
# import time
# import argparse
# import numpy as np
# import cv2
# # from PIL import Image
# import tensorrt as trt
# print("TensorRT init")
# from trt_backend.engine import load_engine
# from trt_backend.engine import cuda, autoinit
# from trt_backend.engine import HostDeviceMem, GiB
# from trt_backend.engine import DEFAULT_TRT_LOGGER,DEFAULT_TRT_RUNTIME

# from trt_backend.engine import do_inference,allocate_buffers
import numpy as np
from models.base import BaseNet


try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class ResNet50ArcFace(BaseNet):
    def __init__(self,engine_path,image_size=(112,112),batch_size=1,channel_first=False):
        super().__init__(engine_path,image_size,batch_size,channel_first=channel_first)
    def predict(self,batch_img_in):
        trt_outputs=super().predict(batch_img_in)
        #trt_outputs[0] = trt_outputs[0].reshape(self.batch_size,self.num_classes)
        
        return trt_outputs










