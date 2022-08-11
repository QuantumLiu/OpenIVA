import numpy as np
import cv2

from openiva.models.base import BaseNetNew
from openiva.engines import EngineORT

__all__ = ["LandmarksExtractor", "pre_process", "post_process"]


class LandmarksExtractor(BaseNetNew):
    ENGINE_CLASS = EngineORT

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

    def predict_single(self, image: np.ndarray, rectangles: list):
        return self.predict({"batch_images": [image], "batch_rectangles": [rectangles]})[0]

    @classmethod
    def func_pre_process(cls):
        return pre_process

    @classmethod
    def func_post_process(cls):
        return post_process


def pre_process(data):
    """
    Returns pre-processed ndarray (n,h,w,c).
    Args:
        data: raw data ndarray (n,h,w,c) 
    Returns:
        pre-processed ndarray (n,h,w,c)
    """
    batch_images = data["batch_images"]
    batch_rectangles = data["batch_rectangles"]

    face_imgs = []
    sizes = []

    for image, rectangles in zip(batch_images, batch_rectangles):
        for rectangle in rectangles:
            cropped = crop(image, rectangle)

            h, w = cropped.shape[:2]
            sizes.append((w, h))

            face_img = _transform(cropped)
            face_imgs.append(face_img)

    data_infer = np.ascontiguousarray(face_imgs, dtype=np.float32)

    return {"data_infer": data_infer, "sizes": sizes}


def post_process(data):
    """
    Returns results (68,2).
    Args:
        outputs: (136,2) Net outputs ndarrays 
    Returns:
        results: landmarks ndarrays (68,2)
    """
    # output=outputs[0]
    outputs, sizes = data["outputs"][0], data["sizes"]
    batch_rectangles = data["batch_rectangles"]

    batch_lms = []
    n = 0
    for rectangles in batch_rectangles:
        lms = []
        for output, rectangle in zip(outputs, rectangles):
            (w, h) = sizes[n]
            points = output.reshape(-1, 2) * (w, h)
            for i in range(len(points)):
                points[i] += (rectangle[0], rectangle[1])

            lms.append(points)
            n += 1
        batch_lms.append(lms)
    return batch_lms


def predict_single(self, image: np.ndarray, rectangles: list):
    return self.predict({"batch_images": [image], "batch_rectangles": [rectangles]})[0]


def _transform(img):
    """
    Returns pre-processed ndarray (h,w,3).
    Args:
        data_raw: raw data ndarray (h,w,3) 
    Returns:
        pre-processed ndarray (3,h,w)
    """
    data_raw = img
    data_raw = cv2.resize(data_raw, (112, 112),
                          interpolation=cv2.INTER_LINEAR)
    data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
    data_raw = data_raw.astype(np.float32)
    data_raw = data_raw/255.
    data_infer = np.transpose(data_raw, [2, 0, 1])  # [None]
    return data_infer


def crop(image, rectangle):
    """
    Returns cropped image.
    Args:
        image: Bitmap
        rectangle: Rectangle

    Returns:
        Bitmap
    """
    h, w, _ = image.shape

    x0 = max(min(w, rectangle[0]), 0)
    x1 = max(min(w, rectangle[2]), 0)
    y0 = max(min(h, rectangle[1]), 0)
    y1 = max(min(h, rectangle[3]), 0)

    num = image[y0:y1, x0:x1]
    return num
