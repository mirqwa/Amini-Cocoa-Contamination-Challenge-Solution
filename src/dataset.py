import pandas as pd


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
