import torch.nn as nn


class FoodCNN(nn.Module):
    def __init__(self, num_classes=251):
        super().__init__()

        # We use a modular approach with Batch Normalization
        self.features = nn.Sequential(
            # Block 1: 64 filters
            self._conv_block(3, 64),
            nn.MaxPool2d(2, 2),

            # Block 2: 128 filters
            self._conv_block_dilated(64, 128),
            nn.MaxPool2d(2, 2),

            # Block 3: Identity Block
            self._identity_block(128, 128),
            nn.MaxPool2d(2, 2),

            # Block 4: Depthwise Separable Convolution
            self._conv_separable_block(128 ,512),
            nn.MaxPool2d(2, 2)
        )

        # Global Average Pooling: Dramatically reduces parameter count
        # (saves VRAM) compared to flattening 112x112 features.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Reduce from 512 * 7 * 7 to 256 or 512
            nn.Linear(512 * 7 * 7, 256),
            nn.BatchNorm1d(256), # Add Batchnorm for speed
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),     # Slightly lower dropout
            nn.Linear(256, num_classes)
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Essential for fast convergence
            nn.ReLU(inplace=True)
        )

    def _conv_block_dilated(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels), # Essential for fast convergence
            nn.ReLU(inplace=True)
        )

    def _identity_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),

            # identity layer
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def _conv_separable_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, groups=in_channels, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x