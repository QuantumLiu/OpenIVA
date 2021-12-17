import numpy as np
import cv2

from models.base import BaseNet


class LandmarksExtractor(BaseNet):


    def _pre_process(self, data_raw):
        data_raw = cv2.resize(data_raw, (112, 112),interpolation=cv2.INTER_LINEAR)
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
        data_raw=data_raw.astype(np.float32)
        data_raw=data_raw/255.
        data_infer=np.transpose(data_raw, [2, 0, 1])[None]
        return data_infer

    def _post_process(self, outputs, w, h):
        output=outputs[0][0]
        points = output.reshape(-1, 2) * (w, h)
        return points


    # @profile
    def predict(self,data,rectangles):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

                

        lms = []

        for rectangle in rectangles:
            cropped = Crop(data, rectangle)
            h,w=cropped.shape[:2]
            data_infer=self._pre_process(cropped)
            outputs = self._infer(data_infer)
            points=self._post_process(outputs,w,h)

            for i in range(len(points)):
                points[i] += (rectangle[0], rectangle[1])

            lms.append(points)

        return lms


def Crop(image, rectangle):
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