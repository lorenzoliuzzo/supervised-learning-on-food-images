# Supervised learning on food images

Final project for a university course on supervised learning: a CNN with fewer
than 10M parameters that classifies the 251 classes of the
[FoodX-251](https://github.com/karansikka1/iFood_2019) dataset.

## Layout

| Path | What |
|---|---|
| `src/model.py` | `FoodCNN`, the from-scratch architecture under the 10M-parameter budget |
| `src/main.py` | Training / evaluation entry point |
| `src/simsiam.py` | SimSiam self-supervised pretraining (Phase E) |
| `src/resnet.py` | The PyTorch ImageNet reference script, kept as a baseline |
| `src/food101.py` | Food-101 experiments |
| `src/main.ipynb` | Exploratory notebook |
| `benchmarks/` | Throughput and parameter measurements backing the plans |
| `notebooks/` | Colab notebooks (GPU throughput probe for Phase E) |
| `plans/` | Dated roadmaps |
| `report/` | Typst report sources and built PDF |

## Dataset

`food251/` is not committed (~5.5 GB). Download the FoodX-251 train/val/test
archives and unpack them so the tree looks like:

```
food251/
├── meta/{class_list.txt,train_labels.csv,val_labels.csv}
├── train_set/
└── val_set/
```

## Running

```bash
python src/main.py food251 --epochs 90 -b 256
```

## Benchmarks

Every number in `plans/` comes from one of these:

```bash
python benchmarks/trunk_variants.py      # params vs throughput vs VRAM
python benchmarks/loader_throughput.py   # is the loader ahead of the GPU?
```
