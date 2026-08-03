import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # Both convolutions are 3x3, and the stride lives on the first of them.
        # A strided 1x1 would step over three quarters of its input without ever
        # reading it; at stride 1 that is invisible, but this trunk downsamples
        # three times, so it would throw away most of the signal.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Zero-init this BN's gamma (Goyal et al., "Bag of Tricks", 2018): the
        # residual branch starts at zero, so each block begins as an identity
        # map. That is what makes warmup plus a high LR safe.
        nn.init.zeros_(self.conv[-1].weight)

        # Only project the shortcut when the addition would otherwise be
        # shape-mismatched. BatchNorm after the projection, same as
        # torchvision's ResNet `downsample` -- without it an unnormalized
        # branch was being added to a BN'd one.
        self.shortcut: nn.Module = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        # Activation goes after the addition, not inside self.conv: that is what
        # makes the block a nonlinearity rather than an affine detour.
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class FoodCNN(nn.Module):
    def __init__(self, num_classes: int = 251) -> None:
        super().__init__()

        # A residual trunk of plain convolutions, sized to spend the parameter
        # budget where it can affect accuracy. Stage widths double as the spatial
        # resolution halves; the final stage carries one block rather than two,
        # which is what keeps the model under 10M.
        self.features = nn.Sequential(
            # Stem: 224 -> 112 -> 56 before any residual stage, so the expensive
            # stages never run at full resolution.
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            self._stage(64, 64, blocks=2, stride=1),     # 56x56
            self._stage(64, 128, blocks=2, stride=2),    # 28x28
            self._stage(128, 256, blocks=2, stride=2),   # 14x14
            self._stage(256, 512, blocks=1, stride=2),   # 7x7
        )

        # Genuinely global: (1, 1), not (7, 7). At (7, 7) the flatten produced
        # 512*7*7 = 25088 features and the first Linear alone was 6.4M
        # parameters -- 94% of the model, for the least useful layer in it.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def _stage(self, in_channels: int, out_channels: int, *, blocks: int, stride: int
               ) -> nn.Sequential:
        # Only the first block of a stage changes width or resolution; the rest
        # refine at constant shape, so their shortcuts stay identity.
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        layers += [ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    # --- Retired blocks, kept as the documented ablation -------------------
    #
    # These built the earlier trunk and are no longer used by `features`. They
    # are kept because the comparison is a result, not dead code: measured on
    # this box (RTX 5050 Laptop, 176 px, batch 160, bf16, channels_last), a
    # depthwise-separable design is the *worse* trade under a parameter cap --
    # MobileNetV2 at 2.55M runs 621 img/s and peaks at 3.83 GiB, while a
    # plain-conv residual net at 6.58M runs 918 img/s and peaks at 1.28 GiB.
    # Depthwise convolutions cost activation memory and wall-clock, not
    # parameters, and throughput is the binding constraint here, not the budget.

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Essential for fast convergence
            nn.ReLU(inplace=True)
        )

    def _conv_block_dilated(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels), # Essential for fast convergence
            nn.ReLU(inplace=True)
        )

    def _conv_separable_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            # depthwise: one 3x3 filter per input channel, no cross-channel mixing
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # pointwise: 1x1 mixes channels, expands to out_channels
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Pooled, flattened 512-d embedding before the classifier head --
        # what SimSiam's projector attaches to (src/simsiam.py).
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.classifier(x)
        return x