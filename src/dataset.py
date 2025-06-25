import shutil
from pathlib import Path

import pandas as pd


INPUT_DATA_DIR = Path("data")
DATASETS_DIR = Path("data/dataset_cross_validation")
TRAIN_IMAGES_DIR = DATASETS_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATASETS_DIR / "labels" / "train"
TEST_IMAGES_DIR = DATASETS_DIR / "images" / "test"


def create_dataset_directories():
    for DIR in [TRAIN_IMAGES_DIR, TEST_IMAGES_DIR, DATASETS_DIR]:
        if DIR.exists():
            shutil.rmtree(DIR)
        DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(INPUT_DATA_DIR / "dataset.zip", DATASETS_DIR)


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
    fold_lbl_distrb = get_class_distributions(kfolds, labels_df, cls_idx)
    images, save_path, ds_yamls = create_yml_directories(folds_df, classes)
    copy_validation_data(images, labels, valid_labels_df, save_path)


if __name__ == "__main__":
    main()
