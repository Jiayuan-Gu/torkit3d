from torch import nn

__all__ = ["Conv1dBNReLU", "Conv2dBNReLU"]


class Conv1dBNReLU(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes,
    optionally followed by batch normalization and ReLU activation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        relu=True,
        bn=True,
        dropout_p=0.0,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, bias=(not bn), **kwargs
        )
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Conv2dBNReLU(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        relu=True,
        bn=True,
        dropout_p=0.0,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, bias=(not bn), **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
