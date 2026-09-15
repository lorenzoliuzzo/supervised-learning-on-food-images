import pathlib

import pytest
import torch
from ssl_probe_calibration import RANDOM_INIT, load_encoder

from model import FoodCNN
from simsiam import SimSiamModel


def test_load_encoder_random_init_needs_no_checkpoint() -> None:
    model = load_encoder(RANDOM_INIT, torch.device('cpu'))

    assert isinstance(model, FoodCNN)


def test_load_encoder_reads_a_supervised_checkpoint(tmp_path: pathlib.Path) -> None:
    trained = FoodCNN(num_classes=251)
    path = tmp_path / 'supervised.pth.tar'
    torch.save({'epoch': 90, 'state_dict': trained.state_dict()}, path)

    loaded = load_encoder(str(path), torch.device('cpu'))

    for key, value in trained.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_load_encoder_strips_the_simsiam_encoder_prefix(tmp_path: pathlib.Path) -> None:
    pretrained = SimSiamModel(FoodCNN(num_classes=251))
    path = tmp_path / 'simsiam.pth.tar'
    torch.save({'epoch': 50, 'state_dict': pretrained.state_dict()}, path)

    loaded = load_encoder(str(path), torch.device('cpu'))

    for key, value in pretrained.encoder.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_load_encoder_rejects_a_foreign_checkpoint(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'not_a_foodcnn.pth.tar'
    torch.save({'state_dict': {'some.other.layer.weight': torch.randn(4, 4)}}, path)

    with pytest.raises(RuntimeError, match='not a FoodCNN or SimSiamModel'):
        load_encoder(str(path), torch.device('cpu'))
