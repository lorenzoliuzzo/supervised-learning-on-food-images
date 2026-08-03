import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import Normalize

from main import load_encoder_weights
from model import FoodCNN
from simsiam import (
    PredictionMLP,
    ProjectionMLP,
    SimSiamModel,
    TwoCropsTransform,
    build_ssl_transform,
    negative_cosine_similarity,
    simsiam_loss,
)


def test_forward_features_output_shape() -> None:
    model = FoodCNN(num_classes=251)
    model.eval()
    with torch.no_grad():
        out = model.forward_features(torch.randn(2, 3, 176, 176))
    assert out.shape == (2, 512)


def test_forward_still_produces_class_logits_after_the_refactor() -> None:
    model = FoodCNN(num_classes=251)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 176, 176))
    assert out.shape == (2, 251)


def test_projection_mlp_output_shape() -> None:
    projector = ProjectionMLP(in_dim=512, hidden_dim=64, out_dim=128)
    projector.eval()
    out = projector(torch.randn(4, 512))
    assert out.shape == (4, 128)


def test_prediction_mlp_output_shape() -> None:
    predictor = PredictionMLP(in_dim=128, hidden_dim=32)
    predictor.eval()
    out = predictor(torch.randn(4, 128))
    assert out.shape == (4, 128)


def test_negative_cosine_similarity_is_minus_one_for_identical_vectors() -> None:
    x = torch.randn(4, 16)
    loss = negative_cosine_similarity(x, x)
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)


def test_negative_cosine_similarity_does_not_backprop_into_z() -> None:
    p = torch.randn(4, 16, requires_grad=True)
    z = torch.randn(4, 16, requires_grad=True)
    negative_cosine_similarity(p, z).backward()
    assert p.grad is not None
    assert z.grad is None


def test_simsiam_loss_decreases_when_optimized_on_a_single_batch() -> None:
    torch.manual_seed(0)
    encoder = FoodCNN(num_classes=251)
    model = SimSiamModel(encoder, proj_dim=64, pred_hidden_dim=32)
    model.train()

    view1 = torch.randn(4, 3, 176, 176)
    view2 = torch.randn(4, 3, 176, 176)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    first_loss = None
    last_loss = None
    for _ in range(20):
        optimizer.zero_grad()
        p1, p2, z1, z2 = model(view1, view2)
        loss = simsiam_loss(p1, p2, z1, z2)
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert last_loss < first_loss


def test_two_crops_transform_produces_two_different_views() -> None:
    torch.manual_seed(0)
    image = Image.fromarray(np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8), mode='RGB')
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    two_crops = TwoCropsTransform(build_ssl_transform(normalize))

    view1, view2 = two_crops(image)

    assert view1.shape == (3, 176, 176)
    assert view2.shape == (3, 176, 176)
    assert not torch.equal(view1, view2)


def test_load_encoder_weights_transfers_encoder_but_not_classifier(tmp_path) -> None:
    torch.manual_seed(0)
    pretrained_encoder = FoodCNN(num_classes=251)
    pretrained_model = SimSiamModel(pretrained_encoder, proj_dim=64, pred_hidden_dim=32)

    checkpoint_path = tmp_path / 'simsiam.pth.tar'
    torch.save({'epoch': 1, 'state_dict': pretrained_model.state_dict()}, checkpoint_path)

    fresh_model = FoodCNN(num_classes=251)
    load_encoder_weights(fresh_model, str(checkpoint_path))

    for key, value in pretrained_encoder.features.state_dict().items():
        assert torch.equal(fresh_model.features.state_dict()[key], value)

    assert not torch.equal(
        fresh_model.classifier[-1].weight, pretrained_model.encoder.classifier[-1].weight
    )
