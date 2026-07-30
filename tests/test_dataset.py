import pathlib

import pandas as pd
from PIL import Image

from main import FoodX251Dataset, dataset_paths, load_val_split


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


def _write_multi_image_split(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    image_dir = root / 'val_set'
    image_dir.mkdir(parents=True)
    names = ['val_000000.jpg', 'val_000001.jpg', 'val_000002.jpg']
    for name in names:
        Image.new('RGB', (8, 8)).save(image_dir / name)

    meta_dir = root / 'meta'
    meta_dir.mkdir(exist_ok=True)
    labels_path = meta_dir / 'val_labels.csv'
    pd.DataFrame({'img_name': names, 'label': [1, 2, 3]}).to_csv(labels_path, index=False)
    return image_dir, labels_path


def test_dataset_subset_keeps_only_the_named_images(tmp_path: pathlib.Path) -> None:
    image_dir, labels_path = _write_multi_image_split(tmp_path)

    dataset = FoodX251Dataset(image_dir, labels_path, subset={'val_000000.jpg', 'val_000002.jpg'})

    assert dataset.image_names == ['val_000000.jpg', 'val_000002.jpg']
    assert dataset.labels == [1, 3]


def test_dataset_without_subset_keeps_every_image(tmp_path: pathlib.Path) -> None:
    image_dir, labels_path = _write_multi_image_split(tmp_path)

    dataset = FoodX251Dataset(image_dir, labels_path)

    assert len(dataset) == 3


def test_load_val_split_dev_and_test_are_disjoint(tmp_path: pathlib.Path) -> None:
    split_path = tmp_path / 'val_split.csv'
    pd.DataFrame({
        'img_name': ['val_000000.jpg', 'val_000001.jpg', 'val_000002.jpg'],
        'label': [1, 2, 3],
        'split': ['dev', 'test', 'dev'],
    }).to_csv(split_path, index=False)

    dev = load_val_split(split_path, 'dev')
    test = load_val_split(split_path, 'test')

    assert dev == {'val_000000.jpg', 'val_000002.jpg'}
    assert test == {'val_000001.jpg'}
    assert dev.isdisjoint(test)


def test_load_val_split_all_returns_none() -> None:
    assert load_val_split('splits/val_split.csv', 'all') is None


def test_load_val_split_missing_file_falls_back_to_none(tmp_path: pathlib.Path) -> None:
    assert load_val_split(tmp_path / 'does_not_exist.csv', 'dev') is None
