import numpy as np
import cv2
import math


def plot_boxes_cv2(img, boxes, cls_confs=None, cls_ids=None, class_names=None, savename=None, color=None):
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [
                      1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if class_names:
            cls_id = cls_ids[i]
            cls_conf = cls_confs[i]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(
                img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def get_area(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    return (x2-x1)*(y2-y1)


def crop_boxes(img, boxes, cls_confs, cls_ids, class_names):

    width = img.shape[1]
    height = img.shape[0]

    areas = []
    crops = {cls: [] for cls in class_names}

    for i in range(len(boxes)):
        box = boxes[i]
        areas.append(get_area(box))

        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        area = (x2-x1)*(y2-y1)

        cls_id = cls_ids[i]
        cls_conf = cls_confs[i]
        cls = class_names[cls_id]

        img_patch = img[y1:y2, x1:x2].copy()
        crops[cls].append([img_patch, area, cls, cls_conf])

    for cls in crops.keys():
        crops[cls] = sorted(crops[cls], key=lambda x: -1*x[1])

    return crops


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names
