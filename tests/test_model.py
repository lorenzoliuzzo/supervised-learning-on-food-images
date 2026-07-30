import torch
import torch.nn as nn

from model import FoodCNN, ResidualBlock


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


def test_residual_block_adds_shortcut_when_conv_path_is_zeroed() -> None:
    block = ResidualBlock(128, 128)

    last_conv = block.conv[-2]
    assert isinstance(last_conv, nn.Conv2d)
    nn.init.zeros_(last_conv.weight)
    nn.init.zeros_(last_conv.bias)

    block.eval()
    x = torch.randn(2, 128, 8, 8)
    out = block(x)

    # With the conv path zeroed, only the shortcut survives -- through the
    # post-addition ReLU, which is why this is relu(x) and not x.
    assert torch.allclose(out, torch.relu(x), atol=1e-5)


def test_residual_block_activates_after_the_addition() -> None:
    # relu(conv(x) + shortcut(x)), not conv(x) + shortcut(x): without this the
    # block feeds an unactivated sum straight into the next pooling stage.
    block = ResidualBlock(128, 128)
    block.eval()
    out = block(torch.randn(2, 128, 8, 8))

    assert (out >= 0).all()


def test_pooling_is_global() -> None:
    # (7, 7) here is what put 94% of the budget into a single Linear.
    model = FoodCNN(num_classes=251)
    assert model.avgpool.output_size == (1, 1)


def test_budget_is_spent_on_features_not_the_classifier() -> None:
    # The original model had this backwards: 4.8% features, 95.2% classifier.
    model = FoodCNN(num_classes=251)
    features = sum(p.numel() for p in model.features.parameters())
    classifier = sum(p.numel() for p in model.classifier.parameters())

    assert features > 10 * classifier


def test_every_parameter_receives_a_gradient() -> None:
    # A residual branch wired up wrongly can leave whole blocks unreachable
    # while the forward pass still returns the right shape.
    model = FoodCNN(num_classes=251)
    model.train()
    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 3, 176, 176)), torch.randint(0, 251, (4,)))
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
