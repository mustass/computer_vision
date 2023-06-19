from PIL import Image
import numpy as np

CAT_MAPPING = {0: 23,
 1: 6,
 2: 19,
 3: 19,
 4: 13,
 5: 13,
 6: 13,
 7: 4,
 8: 4,
 9: 24,
 10: 20,
 11: 20,
 12: 20,
 13: 9,
 14: 9,
 15: 9,
 16: 9,
 17: 9,
 18: 9,
 19: 9,
 20: 16,
 21: 16,
 22: 16,
 23: 16,
 24: 16,
 25: 21,
 26: 2,
 27: 10,
 28: 10,
 29: 26,
 30: 8,
 31: 8,
 32: 8,
 33: 8,
 34: 11,
 35: 11,
 36: 14,
 37: 14,
 38: 14,
 39: 14,
 40: 14,
 41: 14,
 42: 14,
 43: 0,
 44: 0,
 45: 0,
 46: 0,
 47: 0,
 48: 3,
 49: 7,
 50: 1,
 51: 22,
 52: 18,
 53: 25,
 54: 12,
 55: 17,
 56: 17,
 57: 15,
 58: 5,
 59: 27}


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


def NMS(P, iou_threshold = 0.5, background_class = 28):
  # P: list of dicts {'bbox':(x1,y1,x2,y2), 'conf':float, 'pred_class':int, 'true_class':int, 'image_id':int}
  conf_list = np.array([x['conf'] for x in P])
  classes_list = np.array([x['pred_class'] for x in P])
  conf_order = (-conf_list).argsort() # apply minus to reverse order
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
      if isremoved[bbox_idx]==False:  # if not removed yet
        # check overlapping
        iou = get_iou(P[to_keep]['bbox'], P[bbox_idx]['bbox'])
        if iou > iou_threshold:
          isremoved[bbox_idx] = True
  return keep
