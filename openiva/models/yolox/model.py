import numpy as np
import cv2

from openiva.models.base import BaseNetNew
from openiva.engines import EngineORT
from .utils import multiclass_nms

__all__ = ["YOLOX", "pre_process", "post_process"]


class YOLOX(BaseNetNew):
    ENGINE_CLASS = EngineORT

    def __init__(self, onnx_path, input_size=(640, 640), with_p6=False, sessionOptions=None, providers="cpu"):
        super().__init__(onnx_path, sessionOptions=sessionOptions, providers=providers)
        self.input_size = input_size
        self.with_p6 = with_p6

    def pre_process(self, data: dict):
        return pre_process(data, self.input_size)

    def post_process(self, data: dict):
        return post_process(data, self.input_size, self.with_p6)

    @classmethod
    def func_pre_process(cls):
        return pre_process

    @classmethod
    def func_post_process(cls):
        return post_process


def pre_process(data, input_size):
    ratios_batch = []
    data_raw = data["batch_images"]
    data_batch = YOLOX.warp_batch(data_raw)
    data_infer = []
    for img in data_batch:
        img_padded, ratio = _pre_proc_frame(img, input_size)
        data_infer.append(img_padded)
        ratios_batch.append(ratio)

    data_infer = np.ascontiguousarray(data_infer, dtype=np.float32)
    ratios_batch = np.asarray(ratios_batch)
    return {"data_infer": data_infer, "ratios_batch": ratios_batch}


def post_process(data, input_size, with_p6):

    ratios_batch = data['ratios_batch']
    outputs = data["outputs"]

    predictions = _demo_postprocess(
        outputs[0], input_size, p6=with_p6)
    boxes = predictions[..., :4]
    scores = predictions[..., 4:5] * predictions[..., 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2]/2.
    boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3]/2.
    boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2]/2.
    boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3]/2.
    boxes_xyxy /= ratios_batch[:, None, None]

    boxes_batch = []
    scores_batch = []
    cls_batch = []
    for b, s in zip(boxes_xyxy, scores):
        dets = multiclass_nms(b, s, nms_thr=0.45, score_thr=0.1)
        final_boxes, final_scores, final_cls_inds = dets[:,
                                                         :4], dets[:, 4], dets[:, 5]

        boxes_batch.append(final_boxes)
        scores_batch.append(final_scores)
        cls_batch.append(final_cls_inds)

    return boxes_batch, scores_batch, cls_batch


def _pre_proc_frame(img, input_size):
    if len(img.shape) == 3:
        padded_img = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0],
            input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r),
               : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def _demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs
