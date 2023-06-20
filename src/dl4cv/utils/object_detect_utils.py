from PIL import Image
import numpy as np


def fix_orientation(filename, orientation):
    I = Image.open(filename)
    if I._getexif():
        exif = dict(I._getexif().items())
        # Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                I = I.rotate(180, expand=True)
            if exif[orientation] == 6:
                I = I.rotate(270, expand=True)
            if exif[orientation] == 8:
                I = I.rotate(90, expand=True)
    return I.convert("RGB")


def get_iou(bb1, bb2):
    # assuring for proper dimension.
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]
    # calculating dimension of common area between these two boxes.
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])
    # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # calculating intersection area.
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # individual areas of both these bounding boxes.
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])
    # union area = area of bb1_+ area of bb2 - intersection of bb1 and bb2.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def NMS(P, iou_threshold=0.5, background_class=28):
    # P: list of dicts {'bbox':(x1,y1,x2,y2), 'conf':float, 'pred_class':int, 'true_class':int, 'image_id':int}
    conf_list = np.array([x["conf"] for x in P])
    classes_list = np.array([x["pred_class"] for x in P])
    conf_order = (-conf_list).argsort()  # apply minus to reverse order
    isremoved = [False for _ in range(len(P))]
    keep = []

    for idx in range(len(P)):
        to_keep = conf_order[idx]
        class_to_keep = classes_list[to_keep]
        if isremoved[to_keep]:
            continue

        if class_to_keep == background_class:
            continue

        # append to keep list
        keep.append(P[to_keep])
        isremoved[to_keep] = True
        # remove overlapping bboxes
        for order in range(idx + 1, len(P)):
            bbox_idx = conf_order[order]
            if isremoved[bbox_idx] == False:  # if not removed yet
                # check overlapping
                iou = get_iou(P[to_keep]["bbox"], P[bbox_idx]["bbox"])
                if iou > iou_threshold:
                    isremoved[bbox_idx] = True
    return keep
