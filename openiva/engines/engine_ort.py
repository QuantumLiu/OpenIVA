import onnxruntime
import numpy as np
import cv2

from .base import Engine


class EngineORT(Engine):

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

        self.inputs_names = [x.name for x in self.__session.get_inputs()]
        assert len(self.inputs_names) > 0

        self.outputs_names = [x.name for x in self.__session.get_outputs()]
        assert len(self.inputs_names) > 0
        # self.__session.set_providers(providers=[self.provider])
        print("Using {} external provider".format(
            self.__session.get_providers()))

    def run(self, data: dict):
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
        if len(self.inputs_names) == 1:
            outputs = self.__session.run(
                None, {self.inputs_names[0]: data_infer})
        else:
            outputs = self.__session.run(
                None, {n: data_infer[n] for n in self.inputs_names})

        return outputs
