import onnxruntime
import numpy as np
import cv2

class BaseNet():
    DICT_PROVIDERS={"cpu":"CPUExecutionProvider","openvino":"OpenVINOExecutionProvider","tensorrt":"TensorrtExecutionProvider","cuda":"CUDAExecutionProvider"}
    def __init__(self, onnx_path, sessionOptions = None,providers="cpu"):
        """
        Initializes face detector.
        Args:
            confidenceThreshold: Confidence threshold (defaults to 0.95)
            nmsThreshold: NonMaxSuppression threshold (defaults to 0.5)
            sessionOptions: Session options.
        """
        
        self.onnx_path = onnx_path

        self.provider=self.DICT_PROVIDERS.get(providers.lower(),"CPUExecutionProvider")

        self.__session = onnxruntime.InferenceSession(onnx_path, sessionOptions)#,providers=[self.provider])
        
        # print(self.__session.get_providers())
        self.provider=(self.provider if self.provider in self.__session.get_providers() else "CPUExecutionProvider")

        self.__input_name = self.__session.get_inputs()[0].name
        self.__session.set_providers(providers=[self.provider])
        print("Using {} external provider".format(self.__session.get_providers()[0]))

    def _infer(self, data_infer):
        """
        Returns onnx inference outputs.
        Args:
            data_infer: pre-processed ndarray float32 (b,h,w,c) 0.~255.
        Returns:
            Net outputs ndarrays
        """
        # print(data_infer.shape)
        outputs=self.__session.run(None, {self.__input_name: data_infer})

        return outputs


    def _pre_process(self,data_raw):
        """
        Returns pre-processed ndarray (b,h,w,c).
        Args:
            data_raw: raw data ndarray (h,w,c) or list 
        Returns:
            pre-processed ndarray (b,h,w,c)
        """
        pass 

    def _post_process(self,outputs):
        """
        Returns results (b,).
        Args:
            outputs: Net outputs ndarrays
        Returns:
            results
        """
    
    def predict(self,data):
        if isinstance(data,np.ndarray):
            if len(data.shape)==3:
                data_raw=data[None]
            elif len(data)>4:
                raise ValueError("Got error data dims expect 3 or 4, got {}".format(len(data)))
        elif isinstance(data,list):
            data_raw=np.ascontiguousarray(data,dtype=np.float32)
            
        data_infer=self._pre_process(data_raw)

        outputs=self._infer(data_infer)

        results=self._post_process(outputs)

        return results

        
