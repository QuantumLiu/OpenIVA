import numpy as np
import cv2

from models.base import BaseNet


class LandmarksExtractor(BaseNet):

    @staticmethod
    def pre_process(data_raw):
        """
        Returns pre-processed ndarray (n,h,w,c).
        Args:
            data_raw: raw data ndarray (n,h,w,c) 
        Returns:
            pre-processed ndarray (n,h,w,c)
        """
        data_raw = cv2.resize(data_raw, (112, 112),interpolation=cv2.INTER_LINEAR)
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
        data_raw=data_raw.astype(np.float32)
        data_raw=data_raw/255.
        data_infer=np.transpose(data_raw, [2, 0, 1])#[None]
        return data_infer

    @staticmethod
    def post_process(output, w, h):
        """
        Returns results (68,2).
        Args:
            outputs: (136,2) Net outputs ndarrays 
        Returns:
            results: landmarks ndarrays (68,2)
        """
        # output=outputs[0]
        points = output.reshape(-1, 2) * (w, h)
        return points


    # @profile
    def predict(self,data,rectangles):
        """
        Returns face landmarks.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

                

        lms = []
        face_imgs=[]
        sizes=[]
        for rectangle in rectangles:
            cropped = Crop(data, rectangle)

            h,w=cropped.shape[:2]
            sizes.append((w,h))

            face_img=self.pre_process(cropped)
            face_imgs.append(face_img)
        
        data_infer=np.ascontiguousarray(face_imgs,dtype=np.float32)

        outputs = self._infer(data_infer)

        output_batch=outputs[0]
        for output,rectangle,(w,h) in zip(output_batch,rectangles,sizes):
            points=self.post_process(output,w,h)

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