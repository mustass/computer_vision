import json
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image, ExifTags
import pickle
from pathlib import Path

from dl4cv.utils.object_detect_utils import get_iou, fix_orientation


def split_tacos(
    dataset_path="/dtu/datasets1/02514/data_wastedetection",
    outpath="/dtu/blackhole/0f/160495/s210527/taco",
    splits=[1000, 250, 250],
    seed=8008,
):
    """Split the taco dataset into train, val, and test sets."""
    print("Splitting dataset into train, val, and test sets")
    anns_file_path = dataset_path + "/" + "annotations.json"
    # Read annotations
    with open(anns_file_path, "r") as f:
        dataset = json.loads(f.read())
    # Split annotations
    np.random.seed(seed)
    indices = np.arange(0, 1500, 1)
    np.random.shuffle(indices)
    train_indices = indices[: splits[0]]
    val_indices = indices[splits[0] : splits[0] + splits[1]]
    test_indices = indices[splits[0] + splits[1] :]

    train_images = [dataset["images"][i] for i in train_indices]
    val_images = [dataset["images"][i] for i in val_indices]
    test_images = [dataset["images"][i] for i in test_indices]

    train_annotations = _get_annotations_for_images(
        train_indices, dataset["annotations"]
    )
    val_annotations = _get_annotations_for_images(val_indices, dataset["annotations"])
    test_annotations = _get_annotations_for_images(test_indices, dataset["annotations"])

    train_dataset = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": dataset["categories"],
    }
    val_dataset = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": dataset["categories"],
    }
    test_dataset = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": dataset["categories"],
    }

    print("Train dataset size: ", len(train_dataset["images"]))
    print("Val dataset size: ", len(val_dataset["images"]))
    print("Test dataset size: ", len(test_dataset["images"]))
    print(f'Train annotations size: {len(train_dataset["annotations"])}')
    print(f'Val annotations size: {len(val_dataset["annotations"])}')
    print(f'Test annotations size: {len(test_dataset["annotations"])}')

    print("Saving the json files")
    # Write annotations
    with open(outpath + "/train/" + "train_annotations.json", "w") as f:
        f.write(json.dumps(train_dataset))
    with open(outpath + "/val/" + "val_annotations.json", "w") as f:
        f.write(json.dumps(val_dataset))
    with open(outpath + "/test/" + "test_annotations.json", "w") as f:
        f.write(json.dumps(test_dataset))

    supercats = set([cat["supercategory"] for cat in dataset["categories"]])
    cat2supercat = {cat["id"]: cat["supercategory"] for cat in dataset["categories"]}
    supercat2id = {sc: i for i, sc in enumerate(supercats)}
    cat2supercat_encoded = {
        cat_id: supercat2id[cat2supercat[cat_id]] for cat_id in cat2supercat
    }

    return train_dataset, val_dataset, test_dataset, cat2supercat_encoded, supercat2id


def _get_annotations_for_images(image_ids, annotations):
    anns = [a for a in annotations if a["image_id"] in image_ids]
    return anns


def run_selective_search(
    dataset,
    catid_to_supercat_label_mapping,
    outpath,
    target_size=[1280, 720],
    dataset_root_path="/dtu/datasets1/02514/data_wastedetection",
    train=False,
):
    BACKGROUND_LABEL = -1
    label_map = catid_to_supercat_label_mapping

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            break

    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    out = {}

    for image_index, i in tqdm(
        enumerate(dataset["images"]),
        desc="running selective search",
        total=len(dataset["images"]),
    ):
        annot_boxes = [
            annotation["bbox"]
            for annotation in dataset["annotations"]
            if annotation["image_id"] == i["id"]
        ]
        annots_labels = [
            label_map[annotation["category_id"]]
            for annotation in dataset["annotations"]
            if annotation["image_id"] == i["id"]
        ]
        filename = os.path.join(dataset_root_path, i["file_name"])

        try:
            I = fix_orientation(filename, orientation)

            image = cv2.cvtColor(np.array(I), cv2.COLOR_RGB2BGR)

            scale_factor = (
                image.shape[0] / target_size[1],  # height
                image.shape[1] / target_size[0],  # width
            )
            image = cv2.resize(
                image, (target_size[0], target_size[1]), interpolation=cv2.INTER_AREA
            )

            gtvalues = []
            for a_index, annot in enumerate(annot_boxes):
                x1 = int(annot[0] / scale_factor[1])
                y1 = int(annot[1] / scale_factor[0])
                x2 = int(annot[0] / scale_factor[1] + annot[2] / scale_factor[1])
                y2 = int(annot[1] / scale_factor[0] + annot[3] / scale_factor[0])

                assert (
                    y2 <= target_size[1] and y1 <= target_size[1]
                ), f"y2 = {y2} and y1 = {y1} for gt of image {filename} with resized dims {image.shape}"
                assert (
                    x2 <= target_size[0] and x1 <= target_size[0]
                ), f"x2 = {x2} and x1 = {x1} for gt of image {filename} with resized dims {image.shape}"

                gtvalues.append(
                    {
                        "x1": x1,
                        "x2": x2,
                        "y1": y1,
                        "y2": y2,
                        "label": annots_labels[a_index],
                    }
                )

            ss.setBaseImage(image)  # setting given image as base image
            ss.switchToSelectiveSearchFast()  # running selective search on bae image
            ssresults = ss.process()  # processing to get the outputs

            regions = {}
            for e, result in enumerate(ssresults):
                x, y, w, h = result
                result_coords = {"x1": x, "x2": x + w, "y1": y, "y2": y + h}
                assert (
                    y + h <= target_size[1] and y <= target_size[1]
                ), f"y + h = {y + h} and y = {y} for ss of image {filename} with resized dims {image.shape}"
                assert (
                    x + w <= target_size[0] and x <= target_size[0]
                ), f"x + w = {x + w} and x = {x} for ss of image {filename} with resized dims {image.shape}"
                regions[e] = {
                    "coordinates": result_coords,
                    "label": BACKGROUND_LABEL,
                    "iou": 0.0,
                }
                for gtval in gtvalues:
                    iou = get_iou(
                        gtval, result_coords
                    )  # calculating IoU for each of the proposed regions
                    if regions[e]["iou"] < iou and iou > 0.5:
                        regions[e]["iou"] = iou
                        regions[e]["label"] = gtval["label"]
            if train:
                last_index = len(regions)
                for indx, gtval in enumerate(gtvalues):
                    regions[last_index + indx] = {
                        "coordinates": gtval,
                        "label": gtval["label"],
                        "iou": 1.0,
                    }

            # print(f'For Image {image_index} with filename {filename} got {len(regions)} regions')
            # high_iou_regions = [region for region in regions.values() if region['iou'] > 0.5]
            # low_iou_regions = [region for region in regions.values() if region['iou'] < 0.5]
            # print(f'For Image {image_index} with filename {filename} got {len(high_iou_regions)} regions with iou > 0.5')
            # print(high_iou_regions)
            # print(f'For Image {image_index} with filename {filename} got {len(low_iou_regions)} regions with iou < 0.5')
            # print(low_iou_regions[:5])
            out[image_index] = {
                "regions": regions,
                "filename": filename,
                "image_id": i["id"],
            }

        except Exception as excpt:
            print(excpt)
            print("error in " + filename)
            continue

    outpath = Path(outpath)
    outfile_path = outpath / f"ss_regions_{outpath.parts[-1]}.pkl"

    with open(str(outfile_path), "wb") as fp:
        pickle.dump(out, fp)
    return out


def main():
    train_dataset, val_dataset, test_dataset, cat_mapping, _ = split_tacos()
    print("Running selective search")
    train_ss = run_selective_search(
        train_dataset,
        cat_mapping,
        outpath="/dtu/blackhole/0f/160495/s210527/taco/train",
        train=True,
    )
    val_ss = run_selective_search(
        val_dataset, cat_mapping, outpath="/dtu/blackhole/0f/160495/s210527/taco/val"
    )
    test_ss = run_selective_search(
        test_dataset, cat_mapping, outpath="/dtu/blackhole/0f/160495/s210527/taco/test"
    )


if __name__ == "__main__":
    main()
