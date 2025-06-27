import os
import typing
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from ultralytics import YOLO


IMAGE_SIZE = 640


def bb_intersection_over_union(A: typing.List[float], B: typing.List[float]) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def find_matching_box_from_boxes_list(
    boxes_list: typing.List[list],
    weighted_classes: typing.List[float],
    new_box: typing.List[float],
    new_class: float,
    match_iou: float,
) -> tuple:
    best_iou = match_iou
    max_iou = 0
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        iou = (
            bb_intersection_over_union(box, new_box)
            if weighted_classes[i] == new_class
            else 0
        )
        max_iou = max(max_iou, iou)
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou, max_iou


def get_weighted_box(boxes: typing.List[list], scores: typing.List[float]) -> tuple:
    box = np.zeros(4, dtype=np.float32)
    conf = 0
    conf_list = []
    for b, s in zip(boxes, scores):
        box += s * np.array(b)
        conf += s
        conf_list.append(s)
    score = np.max(conf_list)
    box = box / conf
    return box, score


def weighted_fussion(
    all_boxes: typing.List[list],
    all_classes: typing.List[list],
    all_confidences: typing.List[float],
    iou_threshold: float,
) -> float:
    new_boxes = []
    new_classes = []
    new_confidences = []
    weighted_classes = []
    weighted_boxes = []
    weighted_scores = []
    max_ious = []

    for i in range(len(all_boxes)):
        conf = all_confidences[i]
        index, best_iou, max_iou = find_matching_box_from_boxes_list(
            weighted_boxes,
            weighted_classes,
            all_boxes[i],
            all_classes[i],
            iou_threshold,
        )
        if index != -1:
            new_boxes[index].append(all_boxes[i])
            new_classes[index].append(all_classes[i])
            new_confidences[index].append(all_confidences[i])
            weighted_classes[index] = all_classes[i]
            weighted_boxes[index], weighted_scores[index] = get_weighted_box(
                new_boxes[index], new_confidences[index]
            )
            max_ious[index] = max_iou
        else:
            new_boxes.append([all_boxes[i]])
            new_classes.append([all_classes[i]])
            new_confidences.append([all_confidences[i]])
            weighted_classes.append(all_classes[i])
            weighted_boxes.append(all_boxes[i])
            weighted_scores.append(all_confidences[i])
            max_ious.append(max_iou)
    weighted_boxes = [list(weighted_box) for weighted_box in weighted_boxes]
    return weighted_boxes, weighted_classes, weighted_scores, max_ious


def get_models_and_dataset(
    models_project: str, images_paths: typing.List[str], validation_df: pd.DataFrame
) -> tuple:
    models = [
        (model_path, YOLO(model_path))
        for model_path in Path(models_project).rglob("*/weights/best.pt")
    ]
    model_paths = [
        model_path for model_path in Path(models_project).rglob("*/weights/best.pt")
    ]
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
    image_files = (
        image_files
        if validation_df.empty
        else [
            image_file
            for image_file in image_files
            if image_file.split(".")[0] in validation_df.index
        ]
    )

    return models, dataset_paths, image_files


def predict_for_image(
    models: typing.List[tuple],
    dataset_paths: typing.List[str],
    image_file: str,
    confidence: float,
    iou_threshold: float,
    max_detection: int,
) -> typing.Tuple[list]:
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
    return all_boxes, all_classes, all_confidences, names


def do_prediction(
    models_project: str,
    images_paths: typing.List[str],
    output_path: str,
    confidence: float,
    iou_threshold: float,
    max_detection: int,
    validation_df: pd.DataFrame = pd.DataFrame(),
) -> None:
    models, dataset_paths, image_files = get_models_and_dataset(
        models_project, images_paths, validation_df
    )
    all_data = []
    for image_file in tqdm(image_files):
        all_boxes, all_classes, all_confidences, names = predict_for_image(
            models, dataset_paths, image_file, confidence, iou_threshold, max_detection
        )

        if all_boxes:
            boxes, classes, confidences, _ = weighted_fussion(
                all_boxes, all_classes, all_confidences, iou_threshold
            )

            for box, cls, conf in zip(boxes, classes, confidences):
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
                }
            )
    predictions = pd.DataFrame(all_data)
    predictions.to_csv(output_path, index=False)


if __name__ == "__main__":
    do_prediction(
        "runs/train/7fold/2025-05-07 01:14",
        ["data/dataset/images/test"],
        "output/Submission151.csv",
        confidence=0.001,
        iou_threshold=0.5,
        max_detection=int(300),
    )
