import numpy as np
import cv2

from openiva.models.base import BaseNetNew
from openiva.engines import EngineORT

__all__ = ["YOLOV4", "pre_process", "post_process"]


class YOLOV4(BaseNetNew):
    ENGINE_CLASS = EngineORT

    def __init__(self, onnx_path, input_size=(416, 416), sessionOptions=None, providers="cpu"):
        super().__init__(onnx_path, sessionOptions=sessionOptions, providers=providers)
        self.input_size = input_size

    def pre_process(self, data):

        return pre_process(data, self.input_size)

    def post_process(self, data):

        return post_process(data)

    @classmethod
    def func_pre_process(cls):
        return pre_process

    @classmethod
    def func_post_process(cls):
        return post_process


def pre_process(data, input_size):
    ratios_batch = []
    data_raw = data["batch_images"]
    data_batch = YOLOV4.warp_batch(data_raw)
    data_infer = []
    for img in data_batch:
        img_padded = _pre_proc_frame(img, input_size)
        data_infer.append(img_padded)

    data_infer = np.ascontiguousarray(data_infer, dtype=np.float32)
    return {"data_infer": data_infer}


def _pre_proc_frame(img, input_size):

    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32)/255.
    return img


def post_process(data, conf_thresh=0.4, nms_thresh=0.6):

    outputs = data["outputs"]

    bboxes_batch, cls_confs_batch, cls_ids_batch = _post_processing(
        conf_thresh, nms_thresh, outputs)

    return bboxes_batch, cls_confs_batch, cls_ids_batch


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def _post_processing(conf_thresh, nms_thresh, output):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    cls_confs_batch = []
    cls_ids_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        cls_ids = []
        cls_confs = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    box_list = [ll_box_array[k, 0], ll_box_array[k,
                                                                 1], ll_box_array[k, 2], ll_box_array[k, 3]]
                    box_list = [max(x, 0) for x in box_list]
                    bboxes.append(box_list)

                    cls_confs.append(ll_max_conf[k])
                    cls_ids.append(ll_max_id[k])
                    # bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)
        cls_confs_batch.append(cls_confs)
        cls_ids_batch.append(cls_ids)
    return bboxes_batch, cls_confs_batch, cls_ids_batch
