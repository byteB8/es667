"""ResNet implementation with regularization hooks."""

from typing import Optional
import torch
import torch.nn as nn
from .base_model import BaseModel


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize basic block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for convolution
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 = no dropout)
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = self.shortcut(x)

        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(BaseModel):
    """ResNet architecture with configurable regularization."""

    def __init__(
        self,
        num_classes: int,
        layers: list[int] = [2, 2, 2, 2],
        base_channels: int = 64,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize ResNet.

        Args:
            num_classes: Number of output classes
            layers: Number of blocks in each layer
            base_channels: Base number of channels
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 = no dropout)
        """
        super().__init__(num_classes)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.in_channels = base_channels

        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            BasicBlock, base_channels, layers[0], stride=1, use_batch_norm=use_batch_norm, dropout_rate=dropout_rate
        )
        self.layer2 = self._make_layer(
            BasicBlock, base_channels * 2, layers[1], stride=2, use_batch_norm=use_batch_norm, dropout_rate=dropout_rate
        )
        self.layer3 = self._make_layer(
            BasicBlock, base_channels * 4, layers[2], stride=2, use_batch_norm=use_batch_norm, dropout_rate=dropout_rate
        )
        self.layer4 = self._make_layer(
            BasicBlock, base_channels * 8, layers[3], stride=2, use_batch_norm=use_batch_norm, dropout_rate=dropout_rate
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * BasicBlock.expansion, num_classes)

    def _make_layer(
        self,
        block: type,
        channels: int,
        num_blocks: int,
        stride: int,
        use_batch_norm: bool,
        dropout_rate: float,
    ) -> nn.Sequential:
        """Create a layer of blocks."""
        layers = []
        layers.append(
            block(
                self.in_channels,
                channels,
                stride,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
            )
        )
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def create_resnet(
    num_classes: int,
    variant: str = "resnet18",
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
) -> ResNet:
    """
    Create ResNet model.

    Args:
        num_classes: Number of output classes
        variant: ResNet variant ('resnet18', 'resnet34')
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate (0.0 = no dropout)

    Returns:
        ResNet model
    """
    configs = {
        "resnet18": [2, 2, 2, 2],
        "resnet34": [3, 4, 6, 3],
    }

    if variant not in configs:
        raise ValueError(f"Unknown ResNet variant: {variant}")

    return ResNet(
        num_classes=num_classes,
        layers=configs[variant],
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
    )

