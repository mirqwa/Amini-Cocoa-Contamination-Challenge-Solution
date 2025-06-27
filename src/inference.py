import os
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import pandas as pd

from ultralytics import YOLO


IMAGE_SIZE = 640


def do_prediction(
    models_project,
    images_paths,
    output_path,
    confidence,
    iou_threshold,
    max_detection,
    validation_df=pd.DataFrame(),
):
    models = [
        (model_path, YOLO(model_path))
        for model_path in Path(models_project).rglob("*/weights/best.pt")
    ]
    model_paths = [
        model_path for model_path in Path(models_project).rglob("*/weights/best.pt")
    ]
    # folds_map_df = pd.read_csv(f"{models_project}/fold_class_maps.csv")
    dataset_paths = []
    for model_path in model_paths:
        if len(images_paths) == 1:
            dataset_paths.append(images_paths[0])
            continue
        split = str(model_path).split("/")[-3][-1]
        split = split if split.isdigit() else 1
        dataset = str(
            list(
                Path("data/dataset_cross_validation").rglob(
                    f"*/split_{split}/val/images"
                )
            )[0]
        )
        dataset_paths.append(dataset)
    image_files = []
    for images_path in images_paths:
        image_files.extend(os.listdir(images_path))
    image_files = list(set(image_files))
    all_data = []
    image_files = (
        image_files
        if validation_df.empty
        else [
            image_file
            for image_file in image_files
            if image_file.split(".")[0] in validation_df.index
        ]
    )
    # test_images = ["ID_iIZUc1.jpeg", "ID_DitJb1.jpeg", "ID_WU55ux.jpg", "ID_IClv1d.jpg", "ID_YIrpjW.jpg"]
    # image_files = test_images
    for image_file in tqdm(image_files):
        all_boxes = []
        all_classes = []
        all_confidences = []
        model_detections = []
        class_votes = defaultdict(int)
        for model_and_path, dataset_path in zip(models, dataset_paths):
            img_path = Path(f"{dataset_path}/{image_file}")
            model = model_and_path[1]
            results = model(
                img_path,
                imgsz=IMAGE_SIZE,
                verbose=False,
                conf=confidence,
                iou=iou_threshold,
                max_det=max_detection,
            )
            # Extract bounding boxes, confidence scores, and class labels
            classes = (
                results[0].boxes.cls.tolist() if results[0].boxes else []
            )  # Class indices
            all_classes.extend(classes)
            for class_ in classes:
                class_votes[int(class_)] += 1
            boxes = (
                results[0].boxes.xyxy.tolist() if results[0].boxes else []
            )  # Bounding boxes in xyxy format
            all_boxes.extend(boxes)
            confidences = (
                results[0].boxes.conf.tolist() if results[0].boxes else []
            )  # Confidence scores
            all_confidences.extend(confidences)
            model_name = model_and_path[0].parent.parent.name
            model_detections.extend([model_name] * len(boxes))
            names = results[0].names  # Class names dictionary

        if all_boxes:
            boxes, classes, confidences, max_ious = weighted_fussion(
                all_boxes, all_classes, all_confidences, iou_threshold
            )
            # boxes, classes, confidences = split_detections_by_conf(
            #     boxes, classes, confidences, 0.2, 0.3
            # )
            classes_confs = defaultdict(list)
            confs = {0: 0, 1: 0, 2: 0}
            for cls, conf in zip(classes, confidences):
                confs[cls] = max(confs[cls], conf)
                classes_confs[cls].append(conf)
            classes_min_confs = {}
            class_with_max_conf = 0
            max_conf = 0
            for cls, confs in classes_confs.items():
                sorted_confs = sorted(confs, reverse=True)
                cls_min_conf = (
                    sorted_confs[CLASS_MAX_DETECTION[cls]]
                    if len(sorted_confs) > CLASS_MAX_DETECTION[cls]
                    else sorted_confs[-1]
                )
                classes_min_confs[cls] = cls_min_conf
                if max(sorted_confs) > max_conf:
                    max_conf = max(sorted_confs)
                    class_with_max_conf = cls
            class_with_most_votes = max(class_votes, key=class_votes.get)

            for box, cls, conf in zip(boxes, classes, confidences):
                # if conf < classes_min_confs[cls]:
                #     continue
                x1, y1, x2, y2 = box
                detected_class = names[
                    int(cls)
                ]  # Get the class name from the names dictionary

                # Add the result to the all_data list
                all_data.append(
                    {
                        "Image_ID": str(image_file),
                        "class": detected_class,
                        "confidence": conf,
                        "ymin": y1,
                        "xmin": x1,
                        "ymax": y2,
                        "xmax": x2,
                        # "box_size": int((x2 - x1) * (y2 - y1))
                    }
                )
        else:  # If no objects are detected
            all_data.append(
                {
                    "Image_ID": str(image_file),
                    "class": "None",
                    "confidence": None,
                    "ymin": None,
                    "xmin": None,
                    "ymax": None,
                    "xmax": None,
                    # "box_size": None
                }
            )
    predictions = pd.DataFrame(all_data)
    predictions.to_csv(output_path, index=False)
