import torch
import torch.nn as nn

from model import FoodCNN, ResidualBlock


def test_foodcnn_is_a_module() -> None:
    assert issubclass(FoodCNN, nn.Module)


def test_residual_block_adds_shortcut_when_conv_path_is_zeroed() -> None:
    block = ResidualBlock(128, 128)

    last_conv = block.conv[-2]
    assert isinstance(last_conv, nn.Conv2d)
    nn.init.zeros_(last_conv.weight)
    nn.init.zeros_(last_conv.bias)

    block.eval()
    x = torch.randn(2, 128, 8, 8)
    out = block(x)

    # With the conv path zeroed, only the identity shortcut survives.
    assert torch.allclose(out, x, atol=1e-5)
