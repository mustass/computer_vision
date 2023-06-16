import json
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image, ExifTags

def split_tacos(dataset_path = '/dtu/datasets1/02514/data_wastedetection', outpath='/dtu/blackhole/0f/160495/s210527/taco',  splits = [1000,250,250] ,seed = 8008):
    """Split the taco dataset into train, val, and test sets."""
    print('Splitting dataset into train, val, and test sets')
    anns_file_path = dataset_path + '/' + 'annotations.json'
    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())
    # Split annotations
    np.random.seed(seed)
    indices = np.arange(0,1500,1)
    np.random.shuffle(indices)
    train_indices = indices[:splits[0]]
    val_indices = indices[splits[0]:splits[0]+splits[1]]
    test_indices = indices[splits[0]+splits[1]:]

    train_images = [ dataset['images'][i] for i in train_indices ]
    val_images = [ dataset['images'][i] for i in val_indices ]
    test_images = [ dataset['images'][i] for i in test_indices ]
    

    train_annotations = _get_annotations_for_images(train_indices, dataset['annotations'])
    val_annotations = _get_annotations_for_images(val_indices, dataset['annotations'])
    test_annotations = _get_annotations_for_images(test_indices, dataset['annotations'])

    train_dataset = { 
        'images': train_images,
        'annotations': train_annotations,
        'categories': dataset['categories']
    }
    val_dataset = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': dataset['categories']
    }
    test_dataset = {
        'images': test_images,
        'annotations': test_annotations,
        'categories': dataset['categories']
    }

    print('Train dataset size: ', len(train_dataset['images']))
    print('Val dataset size: ', len(val_dataset['images']))
    print('Test dataset size: ', len(test_dataset['images']))
    print(f'Train annotations size: {len(train_dataset["annotations"])}')
    print(f'Val annotations size: {len(val_dataset["annotations"])}')
    print(f'Test annotations size: {len(test_dataset["annotations"])}')

    print('Saving the json files')
    # Write annotations
    with open(outpath + '/' + 'train_annotations.json', 'w') as f:
        f.write(json.dumps(train_dataset))
    with open(outpath + '/' + 'val_annotations.json', 'w') as f:
        f.write(json.dumps(val_dataset))
    with open(outpath + '/' + 'test_annotations.json', 'w') as f:
        f.write(json.dumps(test_dataset))

    return train_dataset, val_dataset, test_dataset


def _get_annotations_for_images(image_ids, annotations):
    anns = [a for a in annotations if a['image_id'] in image_ids]
    return anns

def run_selective_search(dataset, outpath, target_size=[800,600], dataset_root_path= '/dtu/datasets1/02514/data_wastedetection'):

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break


    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    regions=[]
    labels=[]

    for e,i in tqdm(enumerate(dataset['images']),desc='running selective search',total=len(dataset['images'])):
        annots = [box['bbox'] for box in dataset['annotations'] if box['image_id'] == i['id']]
        filename = os.path.join(dataset_root_path, i['file_name']) 
        try:
            I = Image.open(filename)
            if I._getexif():
                exif = dict(I._getexif().items())
                # Rotate portrait and upside down images if necessary
                if orientation in exif:
                    if exif[orientation] == 3:
                        I = I.rotate(180,expand=True)
                    if exif[orientation] == 6:
                        I = I.rotate(270,expand=True)
                    if exif[orientation] == 8:
                        I = I.rotate(90,expand=True)
            I = I.convert('RGB')
            image = cv2.cvtColor(np.array(I), cv2.COLOR_RGB2BGR)
            scale_factor = (image.shape[0]/target_size[0], image.shape[1]/target_size[1])
            image = cv2.resize(image, (target_size[0], target_size[1]), interpolation = cv2.INTER_AREA)
            gtvalues=[]
            for annot in annots:
                x1 = int(annot[0]/scale_factor[0])
                y1 = int(annot[1]/scale_factor[1])
                x2 = int(annot[0]/scale_factor[0] + annot[2]/scale_factor[0])
                y2 = int(annot[1]/scale_factor[1] + annot[3]/scale_factor[1])
                gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
            ss.setBaseImage(image)   # setting given image as base image
            ss.switchToSelectiveSearchFast()     # running selective search on bae image 
            ssresults = ss.process()     # processing to get the outputs
            imout = image.copy()   
            
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0

            print(f'Got {len(ssresults)} regions')
            for e,result in enumerate(ssresults):
                if e < 2000 and flag == 0:     # till 2000 to get top 2000 regions only
                    for gtval in gtvalues:
                        x,y,w,h = result
                        iou = get_iou(gtval,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})  # calculating IoU for each of the proposed regions
                        if counter < 30:       # getting only 30 psoitive examples
                            if iou > 0.70:     # IoU or being positive is 0.7
                                timage = imout[x:x+w,y:y+h]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                images.append(resized)
                                labels.append(1)
                                counter += 1
                        else :
                            fflag =1              # to insure we have collected all psotive examples
                        if falsecounter <30:      # 30 negatve examples are allowed only
                            if iou < 0.3:         # IoU or being negative is 0.3
                                timage = imout[x:x+w,y:y+h]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                images.append(resized)
                                labels.append(0)
                                falsecounter += 1
                        else :
                            bflag = 1             #to ensure we have collected all negative examples
                    if fflag == 1 and bflag == 1:  
                        print("inside")
                        flag = 1        # to signal the complition of data extaction from a particular image
        except Exception as e:
            print(e)
            print("error in "+filename)
            continue

    X_new = np.array(images)
    y_new = np.array(labels)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    np.save(outpath + '/' + 'X_new.npy', X_new)
    np.save(outpath + '/' + 'y_new.npy', y_new)

    return X_new, y_new

def get_iou(bb1, bb2):
  # assuring for proper dimension.
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
  # calculating dimension of common area between these two boxes.
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
  # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
  # calculating intersection area.
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
  # individual areas of both these bounding boxes.
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
  # union area = area of bb1_+ area of bb2 - intersection of bb1 and bb2.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def main():
    train_dataset, val_dataset, test_dataset = split_tacos()
    print('Running selective search')
    X_train, y_train = run_selective_search(train_dataset, outpath='/dtu/blackhole/0f/160495/s210527/taco/train')
    X_val, y_val = run_selective_search(val_dataset, outpath='/dtu/blackhole/0f/160495/s210527/taco/val')
    X_test, y_test = run_selective_search(test_dataset, outpath='/dtu/blackhole/0f/160495/s210527/taco/test')

if __name__ == '__main__':
    main()
