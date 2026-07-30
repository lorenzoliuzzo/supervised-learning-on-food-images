import torch
import torch.nn as nn

from model import FoodCNN


def test_foodcnn_is_a_module() -> None:
    assert issubclass(FoodCNN, nn.Module)


def test_parameter_budget() -> None:
    model = FoodCNN(num_classes=251)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params < 10_000_000


def test_forward_shape() -> None:
    model = FoodCNN(num_classes=251)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 251)
