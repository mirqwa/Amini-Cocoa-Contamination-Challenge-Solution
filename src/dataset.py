import collections
import datetime
import shutil
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image, ExifTags
from sklearn import model_selection


INPUT_DATA_DIR = Path("data")
DATASETS_DIR = Path("data/dataset_cross_validation")
TRAIN_IMAGES_DIR = DATASETS_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATASETS_DIR / "labels" / "train"
TEST_IMAGES_DIR = DATASETS_DIR / "images" / "test"
SPLITS = 7


def create_dataset_directories():
    for DIR in [TRAIN_IMAGES_DIR, TEST_IMAGES_DIR, DATASETS_DIR]:
        if DIR.exists():
            shutil.rmtree(DIR)
        DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(INPUT_DATA_DIR / "dataset.zip", DATASETS_DIR)


def get_labels() -> list:
    labels = sorted(DATASETS_DIR.rglob("*labels/*/*.txt"))
    return labels


def create_yaml_file(train):
    # Create a data.yaml file required by yolo
    class_names = sorted(train["class"].unique().tolist())
    num_classes = len(class_names)
    class_weights = get_class_weights(class_names, get_percentages(train))

    data_yaml = {
        "path": str(DATASETS_DIR.absolute()),
        "train": str(TRAIN_IMAGES_DIR.absolute()),
        "val": str(VAL_IMAGES_DIR.absolute()),
        "test": str(TEST_IMAGES_DIR.absolute()),
        "nc": num_classes,
        "names": class_names,
        "weights": class_weights,
    }

    yaml_path = "data.yaml"
    with open(yaml_path, "w") as file:
        yaml.dump(data_yaml, file, default_flow_style=False)


def get_class_labels():
    yaml_file = "data.yaml"
    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    return [i for i in range(len(classes))], classes


def count_label_instances(labels, cls_idx):
    index = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)

    for label in labels:
        lbl_counter = collections.Counter()

        with open(label, "r") as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ")[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`
    return labels_df, index


def split_data(labels_df: pd.DataFrame) -> list:
    labels_df["Image_ID"] = labels_df.index
    train_names, val_names = model_selection.train_test_split(
        labels_df["Image_ID"].unique(), test_size=0.1, random_state=42
    )
    train_labels_df = labels_df.copy()
    valid_labels_df = labels_df[labels_df["Image_ID"].isin(val_names)]
    boxes_summary_df = pd.read_csv("analysis/overall_train_with_box_sizes.csv")
    boxes_summary_df = boxes_summary_df.fillna("")
    boxes_summary_df = boxes_summary_df[["Image_ID", "day", "hour", "camera_model"]]
    boxes_summary_df["Image_ID"] = boxes_summary_df["Image_ID"].str.split(".").str[0]
    for col in ["day", "hour", "camera_model"]:
        train_labels_df[col] = train_labels_df.apply(
            lambda row: boxes_summary_df[
                boxes_summary_df["Image_ID"] == row["Image_ID"]
            ].iloc[0][col],
            axis=1,
        )
    train_labels_df["class"] = np.where(
        train_labels_df[0] > 0, 0, np.where(train_labels_df[1] > 0, 1, 2)
    )
    train_labels_df["object_count"] = train_labels_df[[0, 1, 2]].sum(axis=1)
    train_labels_df["stratify_label"] = np.where(
        train_labels_df["object_count"] > 4, 5, train_labels_df["object_count"]
    )
    skf = model_selection.StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=0)
    kfolds = list(skf.split(train_labels_df, train_labels_df[["stratify_label"]]))
    return kfolds, train_labels_df, valid_labels_df


def get_folds_df(kfolds, index, labels_df):
    folds = [f"split_{n}" for n in range(1, SPLITS + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"
    return folds_df


def get_data_folds():
    labels = get_labels()
    cls_idx, classes = get_class_labels()
    labels_df, index = count_label_instances(labels, cls_idx)
    kfolds, labels_df_with_counts, valid_labels_df = split_data(labels_df.copy())
    folds_df = get_folds_df(kfolds, labels_df_with_counts.index, labels_df_with_counts)
    return (
        labels,
        classes,
        kfolds,
        labels_df,
        labels_df_with_counts,
        cls_idx,
        folds_df,
        valid_labels_df,
    )


def create_yml_directories(folds_df, classes):
    images = sorted(DATASETS_DIR.rglob("*images/train/*"))
    sorted(images)

    save_path = Path(
        DATASETS_DIR / f"{datetime.date.today().isoformat()}_{SPLITS}-Fold_Cross-val"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": str(split_dir.absolute()),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )
    return images, save_path, ds_yamls


def load_image(filepath):
    image = Image.open(filepath)

    for flag in ExifTags.TAGS.keys():
        if ExifTags.TAGS[flag] == "Orientation":
            break
    orientation = flag
    exif = image._getexif()
    if not exif:
        return image
    orientation_value = exif.get(orientation, None)

    if orientation_value == 3:
        image = image.rotate(180, expand=True)
    elif orientation_value == 6:
        image = image.rotate(270, expand=True)
    elif orientation_value == 8:
        image = image.rotate(90, expand=True)
    return image


def copy_validation_data(images, labels, validation_df, save_path):
    for image, label in zip(images, labels):
        if image.stem not in validation_df.index:
            continue
        img_to_path = save_path / "validation" / "images"
        lbl_to_path = save_path / "validation" / "labels"
        img_to_path.mkdir(parents=True, exist_ok=True)
        lbl_to_path.mkdir(parents=True, exist_ok=True)
        img = load_image(image)
        img.save(img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)


def main():
    create_dataset_directories()
    (
        labels,
        classes,
        kfolds,
        labels_df,
        labels_df_with_counts,
        cls_idx,
        folds_df,
        valid_labels_df,
    ) = get_data_folds()
    images, save_path, ds_yamls = create_yml_directories(folds_df, classes)
    copy_validation_data(images, labels, valid_labels_df, save_path)


if __name__ == "__main__":
    main()
