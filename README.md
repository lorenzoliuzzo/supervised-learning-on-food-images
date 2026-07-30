# Supervised learning on food images

Final project for a university course on supervised learning: a CNN with fewer
than 10M parameters that classifies the 251 classes of the
[FoodX-251](https://github.com/karansikka1/iFood_2019) dataset.

## Layout

| Path | What |
|---|---|
| `model.py` | `FoodCNN`, the from-scratch architecture under the 10M-parameter budget |
| `main.py` | Training / evaluation entry point |
| `resnet.py` | The PyTorch ImageNet reference script, kept as a baseline |
| `food101.py` | Food-101 experiments |
| `main.ipynb` | Exploratory notebook |
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
python main.py food251 --epochs 90 -b 256
```
