import pytest
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import v2

from main import (
    WARMUP_EPOCHS,
    GeneralizedCrossEntropyLoss,
    build_train_transform,
    warmup_cosine_lr,
)
from model import FoodCNN


def test_single_batch_overfit_gate() -> None:
    # The cheapest possible proof the trunk can learn at all -- the one thing
    # that would otherwise waste a 2-hour run discovering a wiring bug.
    torch.manual_seed(0)
    model = FoodCNN(num_classes=251)
    model.train()

    images = torch.randn(8, 3, 176, 176)
    target = torch.randint(0, 251, (8,))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(images), target)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1


def test_channels_last_survives_the_forward_pass() -> None:
    model = FoodCNN(num_classes=251).to(memory_format=torch.channels_last)
    model.eval()
    x = torch.randn(2, 3, 176, 176).to(memory_format=torch.channels_last)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 251)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (0, 1 / WARMUP_EPOCHS),
        (WARMUP_EPOCHS, 1.0),
        (89, pytest.approx(0.0, abs=1e-3)),
    ],
)
def test_warmup_cosine_lr_hits_expected_multipliers(epoch: int, expected: float) -> None:
    # 90-epoch run: linear warmup through epoch 4, peak at epoch 5 (cosine
    # progress 0), decayed to ~0 by the last epoch (cosine progress ~1).
    assert warmup_cosine_lr(epoch, epochs=90) == expected


def test_gce_loss_is_near_zero_for_a_confident_correct_prediction() -> None:
    criterion = GeneralizedCrossEntropyLoss(q=0.7)
    output = torch.zeros(1, 4)
    output[0, 2] = 20.0  # softmax ~= 1 on the true class

    assert criterion(output, torch.tensor([2])).item() < 1e-3


def test_gce_loss_is_bounded_unlike_cross_entropy() -> None:
    # The whole point of GCE: a confidently *wrong* prediction costs at most
    # 1/q, not the unbounded loss CE would assign it -- that cap is what
    # limits how much a single mislabeled example can dominate the gradient.
    criterion = GeneralizedCrossEntropyLoss(q=0.7)
    output = torch.zeros(1, 4)
    output[0, 0] = 20.0  # confidently predicts class 0

    loss = criterion(output, torch.tensor([2]))  # true class is 2
    assert loss.item() < 1 / 0.7 + 1e-3
    assert torch.isfinite(loss)


def test_mixup_replaces_hard_targets_with_a_soft_label_over_the_batch() -> None:
    mixer = v2.MixUp(num_classes=251)
    images = torch.randn(4, 3, 8, 8)
    target = torch.randint(0, 251, (4,))

    _, mixed_target = mixer(images, target)

    assert mixed_target.shape == (4, 251)
    assert torch.allclose(mixed_target.sum(dim=1), torch.ones(4), atol=1e-5)


def test_cutmix_replaces_hard_targets_with_a_soft_label_over_the_batch() -> None:
    mixer = v2.CutMix(num_classes=251)
    images = torch.randn(4, 3, 8, 8)
    target = torch.randint(0, 251, (4,))

    _, mixed_target = mixer(images, target)

    assert mixed_target.shape == (4, 251)
    assert torch.allclose(mixed_target.sum(dim=1), torch.ones(4), atol=1e-5)


def test_ema_model_moves_toward_new_weights_without_jumping_there() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9)
    )
    # The first update_parameters() call seeds the EMA at the current weights
    # rather than blending -- there is nothing to blend with yet. Only the
    # second call onward applies the decay, which is what train() relies on
    # over many SGD steps per epoch.
    ema_model.update_parameters(model)
    seeded = ema_model.module.weight.clone()

    with torch.no_grad():
        model.weight.add_(1.0)
    ema_model.update_parameters(model)

    # Decay 0.9 means the EMA should sit close to the seeded weights, not
    # jump straight to the new ones -- that's the smoothing EMA exists for.
    assert not torch.allclose(ema_model.module.weight, model.weight)
    assert not torch.allclose(ema_model.module.weight, seeded)
    distance_to_new = (ema_model.module.weight - model.weight).abs().mean()
    distance_to_old = (ema_model.module.weight - seeded).abs().mean()
    assert distance_to_old < distance_to_new


@pytest.mark.parametrize("augment", ["none", "trivial", "rand"])
def test_build_train_transform_survives_a_real_image(augment: str) -> None:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = build_train_transform(augment, normalize)
    image = Image.new("RGB", (200, 150))

    out = transform(image)

    assert out.shape == (3, 176, 176)


def test_build_train_transform_none_has_no_augmentation_op() -> None:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = build_train_transform("none", normalize)

    names = [type(t).__name__ for t in transform.transforms]
    assert "TrivialAugmentWide" not in names
    assert "RandAugment" not in names
