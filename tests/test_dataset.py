import pathlib

import pandas as pd
from PIL import Image

from main import FoodX251Dataset, dataset_paths


def _write_split(root: pathlib.Path, image_dir_name: str, labels_name: str) -> None:
    image_dir = root / image_dir_name
    image_dir.mkdir(parents=True)
    Image.new('RGB', (8, 8)).save(image_dir / 'img0.jpg')

    meta_dir = root / 'meta'
    meta_dir.mkdir(exist_ok=True)
    pd.DataFrame({'image': ['img0.jpg'], 'label': [3]}).to_csv(meta_dir / labels_name, index=False)


def test_dataset_paths_derive_from_supplied_root(tmp_path: pathlib.Path) -> None:
    (train_dir, train_labels), (val_dir, val_labels) = dataset_paths(tmp_path)

    assert train_dir == tmp_path / 'train_set'
    assert train_labels == tmp_path / 'meta' / 'train_labels.csv'
    assert val_dir == tmp_path / 'val_set'
    assert val_labels == tmp_path / 'meta' / 'val_labels.csv'


def test_train_and_val_roots_differ(tmp_path: pathlib.Path) -> None:
    (train_dir, train_labels), (val_dir, val_labels) = dataset_paths(tmp_path)

    assert train_dir != val_dir
    assert train_labels != val_labels


def test_val_dataset_reads_val_set_not_train_set(tmp_path: pathlib.Path) -> None:
    _write_split(tmp_path, 'train_set', 'train_labels.csv')
    _write_split(tmp_path, 'val_set', 'val_labels.csv')

    (train_dir, train_labels), (val_dir, val_labels) = dataset_paths(tmp_path)

    train_dataset = FoodX251Dataset(train_dir, train_labels)
    val_dataset = FoodX251Dataset(val_dir, val_labels)

    assert train_dataset.image_dir == tmp_path / 'train_set'
    assert val_dataset.image_dir == tmp_path / 'val_set'
    assert train_dataset.image_dir != val_dataset.image_dir
