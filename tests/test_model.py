import torch.nn as nn

from model import FoodCNN


def test_foodcnn_is_a_module() -> None:
    assert issubclass(FoodCNN, nn.Module)
