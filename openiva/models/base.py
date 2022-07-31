import onnxruntime
import numpy as np
import cv2

from openiva.engines import Engine


class BaseNet():

    DICT_PROVIDERS = {
        "cpu": ("CPUExecutionProvider", {}),
        "openvino": ("OpenVINOExecutionProvider", {}),
        "tensorrt": (
            'TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
                'trt_max_workspace_size': 2147483648*4}
        ),
        "cuda": (
            'CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True, }
        )
    }

    def __init__(self, onnx_path, sessionOptions=None, providers="cpu"):
        """
        Initializes face detector.
        Args:
            confidenceThreshold: Confidence threshold (defaults to 0.95)
            nmsThreshold: NonMaxSuppression threshold (defaults to 0.5)
            sessionOptions: Session options.
        """

        self.onnx_path = onnx_path

        self.provider = self.DICT_PROVIDERS.get(
            providers.lower(), "CPUExecutionProvider")

        self.__session = onnxruntime.InferenceSession(
            onnx_path, sessionOptions, providers=[self.provider])

        # print(self.__session.get_providers())
        self.provider = (self.provider if self.provider[0] in self.__session.get_providers(
        ) else "CPUExecutionProvider")

        self.__input_name = self.__session.get_inputs()[0].name
        # self.__session.set_providers(providers=[self.provider])
        print("Using {} external provider".format(
            self.__session.get_providers()))

    def _infer(self, data: dict):
        """
        Returns onnx inference outputs.
        Args:
            data_infer: pre-processed ndarray float32 (b,h,w,c) 0.~255.
        Returns:
            Net outputs ndarrays
        """
        if isinstance(data, dict):
            data_infer = data["data_infer"]
        elif isinstance(data, np.ndarray):
            data_infer = data
        outputs = self.__session.run(None, {self.__input_name: data_infer})

        return outputs

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


class BaseNetNew():
    ENGINE_CLASS = Engine

    def __init__(self, engine_path, **engine_options):
        """
        Initializes face detector.
        Args:
            confidenceThreshold: Confidence threshold (defaults to 0.95)
            nmsThreshold: NonMaxSuppression threshold (defaults to 0.5)
            sessionOptions: Session options.
        """

        self.engine_path = engine_path

        if engine_options is None:
            self.engine_options = {}
        else:
            self.engine_options = engine_options

        self.engine = self.load_engine(self.engine_path, **self.engine_options)

    def load_engine(self, engine_path, **engine_options):
        return self.ENGINE_CLASS(engine_path, **engine_options)

    def infer(self, data: dict):
        """
        Returns onnx inference outputs.
        Args:
            data_infer: pre-processed ndarray float32 (b,h,w,c) 0.~255.
        Returns:
            Net outputs ndarrays
        """
        if isinstance(data, dict):
            data_infer = data["data_infer"]
        elif isinstance(data, np.ndarray):
            data_infer = data
        outputs = self.engine.run(data_infer)

        return outputs

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

        outputs = self.infer(data)
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

    @property
    def inputs_names(self):
        return self.engine.inputs_names

    @property
    def outputs_names(self):
        return self.engine.outputs_names

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
