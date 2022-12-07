import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _NormBase

__all__ = ["LayerNorm"]


class LayerNorm(nn.Module):
    """Custom LayerNorm.

    See also:
        https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_first"):
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

        assert data_format in ["channels_first", "channels_last"], data_format
        self._data_format = data_format

    def to(self, **kwargs):
        if kwargs.get("memory_format") == torch.channels_last:
            self._data_format = "channels_last"
        super().to(**kwargs)

    @staticmethod
    def normalize(x: torch.Tensor, eps):
        u = x.mean(1, keepdim=True)
        x = x - u
        s = x.pow(2).mean(1, keepdim=True)
        x = x / torch.sqrt(s + eps)
        return x

    def forward(self, x: torch.Tensor):
        if self._data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self._data_format == "channels_first":
            if x.ndim > 2:
                dims = [0] + [i for i in range(2, x.ndim)] + [1]
                x = x.permute(dims).contiguous()
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            if x.ndim > 2:
                dims = [0, x.ndim - 1] + [i for i in range(1, x.ndim - 1)]
                x = x.permute(dims).contiguous()
            return x
            # x = self.normalize(x, eps=self.eps)
            # # Reshape weight and bias
            # weight, bias = self.weight, self.bias
            # other_dims = x.dim() - 2
            # for _ in range(other_dims):
            #     weight = weight.unsqueeze(-1)
            #     bias = bias.unsqueeze(-1)
            # x = weight * x + bias
            # return x

    def extra_repr(self) -> str:
        return f"data_format={self._data_format}"


def test_LayerNorm():
    B, D, N = 4, 3, 10
    x = torch.rand(B, D, N)

    builtin_ln = nn.LayerNorm(D)
    custom_ln = LayerNorm(D)
    custom_ln.weight = builtin_ln.weight
    custom_ln.bias = builtin_ln.bias

    desired = builtin_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
    actual = custom_ln(x)
    torch.testing.assert_allclose(actual, desired)
