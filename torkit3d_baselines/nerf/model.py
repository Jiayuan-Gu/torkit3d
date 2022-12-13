from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torkit3d.nn import LinearNormAct


def march_rays(
    attenuation: torch.Tensor,
    radiance,
    segment_lengths,
    background=None,
):
    if attenuation.ndim == 3:
        attenuation = attenuation.squeeze(-1)
    alpha = 1.0 - torch.exp(-segment_lengths * attenuation)  # [N_rays, N_samples]
    opacity = 1.0 - alpha
    transparency = torch.cumprod(opacity, dim=-1)
    transparency = F.pad(transparency, [1, 0], mode="constant", value=1.0)
    weights = alpha * transparency[..., :-1]

    rgb = torch.sum(weights.unsqueeze(-1) * radiance, dim=-2)  # [N_rays, 3]
    if background is not None:
        rgb = rgb + transparency[-1:].unsqueeze(-1) * background

    return dict(rgb=rgb, weights=weights)


class PositionalEncoding(nn.Module):
    def __init__(self, n_embed, dim=1, include_input=True):
        super().__init__()
        self.n_embed = n_embed
        self.dim = dim
        self.include_input = include_input
        freq = torch.pow(2, torch.arange(0, self.n_embed))
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, x: torch.Tensor):
        _shape = [(-1 if i == self.dim else 1) for i in range(x.ndim)]
        freq = self.freq.reshape(_shape).unsqueeze(self.dim)
        t = x.unsqueeze(self.dim + 1) * freq
        t = t.flatten(self.dim, self.dim + 1)
        embed = [torch.sin(t), torch.cos(t)]
        if self.include_input:
            embed.append(x)
        return torch.cat(embed, dim=self.dim)


def inverse_transform_sampling(bins, cdf, n_samples, eps=1e-5):
    assert bins.shape == cdf.shape, (bins.shape, cdf.shape)

    # Sample x from a uniform distribution
    x = torch.rand(*cdf.shape[:-1], n_samples, dtype=cdf.dtype, device=cdf.device)
    inds = torch.searchsorted(cdf, x, right=True)  # [..., n_samples]
    lower = torch.clamp(inds - 1, min=0)
    upper = torch.clamp(inds, max=int(cdf.shape[-1] - 1))

    # Not efficient, but easy to read
    cdf_l = torch.gather(cdf, dim=-1, index=lower)  # [..., n_samples]
    cdf_u = torch.gather(cdf, dim=-1, index=upper)  # [..., n_samples]
    bins_l = torch.gather(bins, dim=-1, index=lower)  # [..., n_samples]
    bins_u = torch.gather(bins, dim=-1, index=upper)  # [..., n_samples]

    cdf_d = cdf_u - cdf_l
    t = (x - cdf_l) / torch.where(cdf_d < eps, 1.0, cdf_d)
    y = bins_l + t * (bins_u - bins_l)
    return y


class NeRF(nn.Module):
    def __init__(
        self,
        in_channels=3,
        in_channels_viewdir=3,
        hidden_size=256,
        hidden_layers=8,
        skip_layers=(5,),
        pe_dim=-1,
        pe_dim_viewdir=-1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.in_channels_viewdir = in_channels_viewdir
        self.skip_layers = skip_layers

        self.mlp = nn.ModuleList()
        c_in = in_channels
        if pe_dim > 0:
            self.pe = PositionalEncoding(pe_dim)
            c_in = c_in + 2 * self.in_channels * pe_dim
        else:
            self.pe = None
        _c_in = c_in  # Keep for skip layers
        for i in range(hidden_layers):
            # NOTE(jigu): The original implementation adds the skip after the layer.
            if i in skip_layers:
                c_in = c_in + _c_in
            layer = LinearNormAct(c_in, hidden_size)
            c_in = hidden_size
            self.mlp.append(layer)

        self.output_alpha = nn.Linear(hidden_size, 1)

        if self.in_channels_viewdir > 0:
            self.output_feat = nn.Linear(hidden_size, hidden_size)
            c_in = hidden_size + self.in_channels_viewdir
            if pe_dim_viewdir > 0:
                self.pe_viewdir = PositionalEncoding(pe_dim_viewdir)
                c_in = c_in + 2 * self.in_channels_viewdir * pe_dim_viewdir
            else:
                self.pe_viewdir = None
            self.mlp_viewdir = nn.Sequential(LinearNormAct(c_in, hidden_size // 2))
            self.output_rgb = nn.Linear(hidden_size // 2, 3)
        else:
            self.output_rgb = nn.Linear(hidden_size, 3)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        rays: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor],
        n_segments=64,
        background=None,
    ):
        assert rays.ndim == 2, rays.shape
        return self.render_rays(
            rays[..., 0:3],
            rays[..., 3:6],
            near,
            far,
            n_segments,
            background=background,
            perturb=self.training,
        )

    def encode_points(self, points: torch.Tensor, viewdirs=None):
        ori_shape = list(points.shape)  # (N_rays, N_samples, 3)
        points = points.flatten(0, -2)
        x = points

        # Positional embedding
        if self.pe is not None:
            x = self.pe(x)

        # Keep for skip layers
        _x = x

        if viewdirs is not None:
            viewdirs = viewdirs.unsqueeze(-2).expand(ori_shape)
            viewdirs = viewdirs.flatten(0, -2)

            if self.pe_viewdir is not None:
                viewdirs = self.pe_viewdir(viewdirs)

        for i, layer in enumerate(self.mlp):
            if i in self.skip_layers:
                x = torch.cat([x, _x], dim=1)
            x = layer(x)

        alpha = self.output_alpha(x)

        # NOTE(jigu): Outputs are not actually alpha and rgb
        if self.in_channels_viewdir > 0:
            feat = self.output_feat(x)
            feat = self.mlp_viewdir(torch.cat([feat, viewdirs], dim=1))
            rgb = self.output_rgb(feat)
        else:
            rgb = self.output_rgb(x)

        # Reshape
        alpha = alpha.reshape(ori_shape[:-1] + [alpha.shape[-1]])
        rgb = rgb.reshape(ori_shape[:-1] + [rgb.shape[-1]])

        # Activation
        alpha = F.relu(alpha)
        rgb = torch.sigmoid(rgb)

        return dict(alpha=alpha, rgb=rgb)

    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor],
        n_segments: int,
        background=None,
        perturb=True,
    ):
        # NOTE(jigu): Different from the original implementation
        # Sample points (along the ray)
        if perturb:
            t = torch.linspace(
                0, 1, n_segments, dtype=rays_o.dtype, device=rays_o.device
            )
            t = t.expand(*rays_o.shape[:-1], n_segments)
            t_rand = torch.rand(
                *rays_o.shape[:-1],
                n_segments - 1,
                dtype=rays_o.dtype,
                device=rays_o.device
            )
            t = t[..., :-1] + t_rand * (t[..., 1:] - t[..., :-1])
            t = torch.cat([t[..., :1], t, t[..., -1:]], dim=-1)
        else:
            t = torch.linspace(
                0, 1, n_segments + 1, dtype=rays_o.dtype, device=rays_o.device
            )
            t = t.expand(*rays_o.shape[:-1], n_segments + 1)

        bins = near * (1 - t) + far * t  # [..., n_segments + 1]
        z = 0.5 * (bins[..., :-1] + bins[..., 1:])
        dists = bins[..., 1:] - bins[..., :-1]  # [..., n_segments]

        # Points in the world frame
        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z.unsqueeze(-1)

        outs_e = self.encode_points(points, viewdirs=rays_d)
        outs_r = march_rays(
            outs_e["alpha"], outs_e["rgb"], dists, background=background
        )

        # Output bins for stratified sampling
        outs_r["bins"] = bins

        return outs_r

    def render_rays_stratified(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_importance: int,
        background=None,
    ):
        # Stratified sampling
        with torch.no_grad():
            pdf = weights / torch.clamp(weights.sum(-1, keepdim=True), min=1e-5)
            cdf = torch.cumsum(pdf, dim=-1)
            cdf = F.pad(cdf, [1, 0], mode="constant", value=0.0)
            bins2 = inverse_transform_sampling(bins, cdf, n_importance)

        # Merge two bins
        bins, _ = torch.sort(torch.cat([bins, bins2], dim=-1), dim=-1)
        z = 0.5 * (bins[..., :-1] + bins[..., 1:])
        dists = bins[..., 1:] - bins[..., :-1]  # [..., n_importance + (n_bins - 1)]

        # Points in the world frame
        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z.unsqueeze(-1)

        outs_e = self.encode_points(points, viewdirs=rays_d)
        outs_r = march_rays(
            outs_e["alpha"], outs_e["rgb"], dists, background=background
        )

        # Output bins for stratified sampling
        outs_r["bins"] = bins

        return outs_r


def test():
    N_rays = 10
    rays_d = torch.randn(N_rays, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = -rays_d * (0.5 + 0.5 * torch.rand(N_rays, 1))
    rays = torch.cat([rays_o, rays_d], dim=-1)

    # model = NeRF()
    model = NeRF(pe_dim=10, pe_dim_viewdir=4)
    print(model)
    ret = model.forward(rays, near=0.1, far=1.0)
    for k, v in ret.items():
        print(k, v.shape)

    # Stratified sampling
    ret = model.render_rays_stratified(rays_o, rays_d, ret["bins"], ret["weights"], 128)
    for k, v in ret.items():
        print(k, v.shape)


# def test_inverse_transform_sampling():
#     bins = torch.linspace(0, 1, 11)
#     # cdf = torch.linspace(0, 1, 11)
#     cdf = torch.zeros(11)
#     cdf[-2:] = 0.5
#     samples = inverse_transform_sampling(bins, cdf, n_samples=1000)
#     print(torch.histogram(samples, bins))
