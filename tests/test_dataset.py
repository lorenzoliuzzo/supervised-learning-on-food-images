import pathlib

import pandas as pd
from PIL import Image

from main import FoodX251Dataset, dataset_paths


def _write_split(
    root: pathlib.Path,
    image_dir_name: str,
    labels_name: str,
    image_name: str,
    label: int,
) -> None:
    image_dir = root / image_dir_name
    image_dir.mkdir(parents=True)
    Image.new('RGB', (8, 8)).save(image_dir / image_name)

    meta_dir = root / 'meta'
    meta_dir.mkdir(exist_ok=True)
    pd.DataFrame({'image': [image_name], 'label': [label]}).to_csv(
        meta_dir / labels_name, index=False
    )


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
    _write_split(tmp_path, 'train_set', 'train_labels.csv', 'train_000000.jpg', 11)
    _write_split(tmp_path, 'val_set', 'val_labels.csv', 'val_000000.jpg', 42)

    (train_dir, train_labels), (val_dir, val_labels) = dataset_paths(tmp_path)

    train_dataset = FoodX251Dataset(train_dir, train_labels)
    val_dataset = FoodX251Dataset(val_dir, val_labels)

    assert train_dataset.image_dir == tmp_path / 'train_set'
    assert val_dataset.image_dir == tmp_path / 'val_set'
    assert train_dataset.image_dir != val_dataset.image_dir

    # The directories were only half the bug: the original code paired val_set
    # with train_labels.csv, so validation scored the model against the wrong
    # labels. Pin the labels themselves, not just the directory they sit next to.
    assert train_dataset.labels == [11]
    assert val_dataset.labels == [42]
    assert train_dataset.image_names == ['train_000000.jpg']
    assert val_dataset.image_names == ['val_000000.jpg']


def test_val_dataset_yields_the_val_label(tmp_path: pathlib.Path) -> None:
    # End to end through __getitem__: a dataset wired to the wrong CSV returns
    # the wrong target even when every path looks right.
    _write_split(tmp_path, 'val_set', 'val_labels.csv', 'val_000000.jpg', 42)

    _, (val_dir, val_labels) = dataset_paths(tmp_path)
    _, target = FoodX251Dataset(val_dir, val_labels)[0]

    assert target == 42
