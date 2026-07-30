import pytest
import torch
import torch.nn as nn

from main import WARMUP_EPOCHS, warmup_cosine_lr
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
