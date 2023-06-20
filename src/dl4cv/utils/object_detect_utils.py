from PIL import Image
import numpy as np
import torch

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
            if classes_list[bbox_idx] != class_to_keep:
                continue
            if isremoved[bbox_idx] == False:  # if not removed yet
                # check overlapping
                iou = get_iou(P[to_keep]["bbox"], P[bbox_idx]["bbox"])
                if iou > iou_threshold:
                    isremoved[bbox_idx] = True
    return keep



def calc_mAPs_image(pred,truth):
    preds_by_class, truths_by_class = split_by_class(pred,truth)
    mAP = {}
    for c in set(list(preds_by_class.keys()) + list(truths_by_class.keys())):
        if c not in preds_by_class.keys() or c not in truths_by_class.keys():
            mAP[c] = 0
        else:
            mAP[c] = calc_mAP_class(preds_by_class[c],truths_by_class[c])
    return mAP

def calc_total_mAP(pred, truth):
    image_mAPs = {}
    for i in range(len(pred)):
        image_mAPs[i] = calc_mAPs_image(pred[i],truth[i])

    print(f'Image mAPs: {image_mAPs}')
    # mean to get class average
    class_mAPs = {i:None for i in range(29)}
    for i in range(29):
        class_mAP = [image_mAPs[j][i] for j in range(len(image_mAPs)) if i in image_mAPs[j].keys()]
        class_mAPs[i] = np.mean(class_mAP)
    
    # remove nan values
    class_mAPs = {k:v for k,v in class_mAPs.items() if not(np.isnan(v))}
    print(f'Class mAPs: {class_mAPs}')
    return  np.mean(list(class_mAPs.values()))

from sklearn.metrics import auc

def calc_mAP_class(pred,truth):
    # sort predictions by confidence
    detections = sorted(pred, key=lambda k: k['conf'], reverse=True)

    total_truth = len(truth) # TP + FN
    
    true_pos = 0
    false_pos = 0

    recalls = []
    precisions = []

    removed_truths = [False for t in truth]

    # compute precision and recall for each detection
    for d_idx, d in enumerate(detections):
        ious = {}
        for t_idx, t in enumerate(truth):
            if removed_truths[t_idx]:
                continue
            
            iou = get_iou(d['bbox'], t['bbox'])

            if iou > 0.5:
                ious[t_idx] = iou
        
        if len(ious) == 0:
            false_pos += 1
        elif len(ious) == 1:
            true_pos += 1
            removed_truths[list(ious.keys())[0]] = True
        elif len(ious) > 1:
            true_pos += 1
            # find index with hishest iou
            max_iou = 0
            max_idx = 0
            for idx, iou in ious.items():
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx
            removed_truths[max_idx] = True
        else:
            raise Exception('Something went wrong')
        
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / total_truth

        recalls.append(recall)
        precisions.append(precision)

    if len(precisions) == 1 and  precisions[0]==1.0:
        return 1
    elif len(precisions) == 1 and  precisions[0]==0.0:
        return 0
    else:
        return auc(recalls, precisions)
    
def split_by_class(preds,truths):
    pred_by_class = {i:[] for i in range(29)}
    truth_by_class = {i:[] for i in range(29)}
    for pred in preds:
        if pred['true_class'] == 28:
            continue
        pred_by_class[pred['pred_class']].append(pred)
    
    for truth in truths:
        truth_by_class[truth['pred_class']].append(truth)

    valids_preds = []
    for i in range(len(pred_by_class)):
        if len(pred_by_class[i]) > 0:
            valids_preds.append(i)
    
    valids_truths = []
    for i in range(len(truth_by_class)):
        if len(truth_by_class[i]) > 0:
            valids_truths.append(i)

    preds_by_class = {i: pred_by_class[i] for i in valids_preds}
    truth_by_class ={i: truth_by_class[i] for i in valids_truths}
    
    return preds_by_class,truth_by_class


def mAP_calc(predictions,truth,iou_thresh=0.5, num_classes=29):
    average_precisions = []

    # do per class so:
    for c in range(0, num_classes-1):

        TP = 0
        FP = 0

        total_true_bboxes = 0

        detections = []

        for idx, prs in enumerate(predictions):
            for pr in prs:
                if pr['pred_class'] == c:
                    detections.append((pr['conf'], idx, pr['bbox']))

        is_detected = []
        for ground_truth in truth:
            is_detected.append([False for region in ground_truth if region['pred_class']==c]) 
            total_true_bboxes += sum([region['pred_class']==c for region in ground_truth]) # count how many true bboxes there are

        detections.sort(reverse=True)

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            best_iou = 0
            for idx, gt in enumerate([region for region in truth[detection[1]] if region['pred_class']==c]):
                iou = get_iou(gt['bbox'], detection[2])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_thresh:
                if not is_detected[detection[1]][best_gt_idx]:
                    TP[detection_idx] = 1
                    is_detected[detection[1]][best_gt_idx] = True
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
                

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + 1e-6)  # ratio of detected objects!
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)         # ratio of predictions that are true objects!
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
    return sum(average_precisions) / len(average_precisions)