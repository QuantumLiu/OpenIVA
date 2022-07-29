import onnxruntime
import numpy as np
import cv2


class BaseNet(object):

    def __init__(self, engine_path, options=None):
        """
        Initializes face detector.
        Args:
            engine_path: string, path of model weights file
            nmsThreshold: NonMaxSuppression threshold (defaults to 0.5)
            sessionOptions: Session options.
        """

        self.engine_path = engine_path

        self.options = options

        self.engine = None
        self.load_engine()

    def load_engine(self):
        pass

    def _infer(self, data: dict):
        """
        Returns onnx inference outputs.
        Args:
            data_infer: pre-processed ndarray float32 (b,h,w,c) 0.~255.
        Returns:
            Net outputs ndarrays
        """
        pass

    def pre_process(self, data: dict):
        """
        Returns pre-processed ndarray (b,h,w,c).
        Args:
            data_raw: raw data ndarray (h,w,c) or list 
        Returns:
            pre-processed ndarray (b,h,w,c)
        """
        pass

    def _pre_proc_frame(self, img):
        pass

    def post_process(self, data: dict):
        """
        Returns results (b,).
        Args:
            outputs: Net outputs ndarrays
        Returns:
            results
        """
        pass

    def predict(self, data: dict):

        if not isinstance(data, dict):
            data_dict = {}
            data_dict["batch_images"] = data
            data = data_dict

        data_infer = self.pre_process(data)
        if isinstance(data_infer, dict):
            data.update(data_infer)
        else:
            data["data_infer"] = data_infer

        outputs = self._infer(data)
        if isinstance(outputs, dict):
            data.update(outputs)
        else:
            data["outputs"] = outputs

        results = self.post_process(data)
        if isinstance(results, dict):
            data.update(results)
        else:
            data["results"] = results

        return results

    # @classmethod
    # def func_pre_process(self):
    #     def func_pre_process(data):
    #         return self.pre_process(data)
    #     return func_pre_process

    # @classmethod
    # def func_post_process(self):
    #     def func_post_process(data):
    #         return self.pre_process(data)
    #     return func_post_process

    @staticmethod
    def warp_batch(data):
        if isinstance(data, np.ndarray):
            if len(data.shape) == 3:
                return data[None]
            elif len(data.shape) > 4:
                raise ValueError(
                    "Got error data dims expect 3 or 4, got {}".format(len(data)))
        elif isinstance(data, list):
            return data


class Engine(object):
    def __init__(self, engine_path, options=None):
        pass
