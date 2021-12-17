import numpy as np
import cv2

from models.base import BaseNet

class Detector(BaseNet):
    def __init__(self, onnx_path, input_size=(640,480), confidenceThreshold = 0.95, nmsThreshold = 0.5, top_k=5, sessionOptions=None,providers="cpu"):
        super().__init__(onnx_path, sessionOptions=sessionOptions,providers=providers)

        self.input_size=input_size
        self.confidenceThreshold=confidenceThreshold
        self.nmsThreshold=nmsThreshold
        self.top_k=top_k
        

    def _pre_process(self, data_raw):
        data_raw = cv2.resize(data_raw, self.input_size,interpolation=cv2.INTER_LINEAR)
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
        data_raw=data_raw.astype(np.float32)
        data_raw=(data_raw-127.)/128.
        data_infer=np.transpose(data_raw, [2, 0, 1])#[None]
        return data_infer

    def _post_process(self, outputs, w, h):
        boxes_batch, confidences_batch=outputs
        rectangles_batch, probes_batch = _parse_result(w, h, boxes_batch, confidences_batch, self.confidenceThreshold, self.nmsThreshold, self.top_k)
        return rectangles_batch, probes_batch


    # @profile
    def predict(self,data):
        if isinstance(data,np.ndarray):
            if len(data.shape)==3:
                h,w=data.shape[:2]

                data_infer=self._pre_process(data)
                data_infer=data_infer[None]
                
            elif len(data)>4:
                raise ValueError("Got error data dims expect 3 or 4, got {}".format(len(data)))

        elif isinstance(data,list):
            h,w=data[0].shape[:2]

            data_infer=[]
            for data_raw in data:
                data_infer.append(self._pre_process(data_raw))

            # data_infer=np.ascontiguousarray(data_infer,dtype=np.float32)
            data_infer=np.ascontiguousarray(data_infer,dtype=np.float32)
        
        # print(data_raw.shape)
        # h,w,_=data_raw.shape[1:]
            
        # data_infer=self._pre_process(data_raw)

        outputs=self._infer(data_infer)

        rectangles_batch, probes_batch=self._post_process(outputs,w,h)

        # boxes=[]
        # for rect in rectangles:
        #     boxes.append(ToBox(rect))

        return  rectangles_batch, probes_batch

def _parse_result(width, height, boxes_batch, confidences_batch, prob_threshold, iou_threshold=0.5, top_k=5):
    """
    Selects boxes that contain human faces.
    Args:
        width: original image width
        height: original image height
        boxes_batch (N, K, 4): boxes array in corner-form
        confidences_batch (N, K, 2): confidence array
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (N, K, 4): an array of boxes kept
        probs (N, K): an array of probabilities for each boxes being in corresponding labels
    """
    picked_box_batch=[]
    picked_probs_batch=[]
    for boxes, confidences in zip(boxes_batch,confidences_batch):
        picked_box_probs = []
        
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = __hard_nms(box_probs,
            iou_threshold=iou_threshold,
            top_k=top_k,
            )
            picked_box_probs.append(box_probs)
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        
        picked_box_batch.append(ToBoxN(picked_box_probs[:, :4].astype(np.int32)))
        picked_probs_batch.append(picked_box_probs[:, 4])
    return picked_box_batch,picked_probs_batch

def __area_of(left_top, right_bottom):
    """
    Computes the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def __iou_of(boxes0, boxes1, eps=1e-5):
    """
    Returns intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = __area_of(overlap_left_top, overlap_right_bottom)
    area0 = __area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = __area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def __hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Performs hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = __iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def ToBox(rectangle):
    """
    Returns rectangle scaled to box.
    Args:
        rectangle: Rectangle
    Returns:
        Rectangle
    """
    width = rectangle[:,2] - rectangle[:,0]
    height = rectangle[:,3] - rectangle[:,1]
    m = max(width, height)
    dx = int((m - width)/2)
    dy = int((m - height)/2)
    
    return [rectangle[0] - dx, rectangle[1] - dy, rectangle[2] + dx, rectangle[3] + dy]

def ToBoxN(rectangle_N):
    """
    Returns rectangle scaled to box.
    Args:
        rectangle: Rectangle
    Returns:
        Rectangle
    """
    width = rectangle_N[:,2] - rectangle_N[:,0]
    height = rectangle_N[:,3] - rectangle_N[:,1]
    m=np.max([width,height],axis=0)
    dx = ((m - width)/2).astype("int32")
    dy = ((m - height)/2).astype("int32")

    rectangle_N[:,0] -= dx
    rectangle_N[:,1] -= dy
    rectangle_N[:,2] += dx
    rectangle_N[:,3] += dy
    return rectangle_N