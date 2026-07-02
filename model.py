"""
cINN architecture for Hjj unfolding.

Ported directly from the training notebook (section "7. Build cINN /
conditional flow with FrEIA" — the name is stale, the implementation is
pure PyTorch with no FrEIA dependency).

Every class here is structurally identical to the notebook version, so a
checkpoint trained there will load here with a plain `load_state_dict`.
The only change is that hyperparameters (widths, depths, spline bins,
number of blocks, dims) are passed in explicitly via a config dict
instead of being read off module-level globals. This is what lets one
codebase serve multiple model variants (e.g. the 66-dim/12-dim
no-energy model and a future with-energy model) by swapping a YAML file
instead of editing code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Conditioning subnet
# ──────────────────────────────────────────────────────────────────────────

class ConditioningSubnet(nn.Module):
    """
    MLP that preprocesses the reco conditioning vector.
    Input:  (batch, context_dim)   e.g. 66
    Output: (batch, width)
    """

    def __init__(self, context_dim: int, width: int, depth: int):
        super().__init__()
        layers = []
        in_dim = context_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.GELU()]
            in_dim = width
        self.net = nn.Sequential(*layers)
        self.out_dim = width

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────
# Coupling subnet (shared helper used by both coupling block types)
# ──────────────────────────────────────────────────────────────────────────

def make_subnet(in_dim: int, out_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.GELU(),
        nn.Dropout(p=0.1),
        nn.Linear(width, width),
        nn.GELU(),
        nn.Dropout(p=0.1),
        nn.Linear(width, out_dim),
    )


# ──────────────────────────────────────────────────────────────────────────
# Affine coupling block (kept for completeness / future models; the
# current no-energy model uses "rqs")
# ──────────────────────────────────────────────────────────────────────────

class AffineCouplingBlock(nn.Module):
    def __init__(self, x_dim, context_dim, width, clamp=2.0, **_ignored):
        super().__init__()
        self.x1_dim = x_dim // 2
        self.x2_dim = x_dim - self.x1_dim
        self.clamp = clamp
        self.subnet = make_subnet(self.x1_dim + context_dim, self.x2_dim * 2, width=width)

    def _st(self, x1, context):
        h = self.subnet(torch.cat([x1, context], dim=-1))
        s, t = h.chunk(2, dim=-1)
        s = self.clamp * torch.tanh(s / self.clamp)
        return s, t

    def forward(self, x, context):
        x1, x2 = x[:, :self.x1_dim], x[:, self.x1_dim:]
        s, t = self._st(x1, context)
        z2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return torch.cat([x1, z2], dim=-1), log_det

    def inverse(self, z, context):
        z1, z2 = z[:, :self.x1_dim], z[:, self.x1_dim:]
        s, t = self._st(z1, context)
        x2 = (z2 - t) * torch.exp(-s)
        return torch.cat([z1, x2], dim=-1)


# ──────────────────────────────────────────────────────────────────────────
# Rational Quadratic Spline (Durkan et al. 2019 — Neural Spline Flows)
# ──────────────────────────────────────────────────────────────────────────

def _rqs_forward_batched(x, widths, heights, derivatives, tail):
    """
    Vectorised RQS forward pass.
    x:           (batch, D)
    widths:      (batch, D, K)
    heights:     (batch, D, K)
    derivatives: (batch, D, K+1)
    returns:     z (batch, D), log_det (batch, D)
    """
    cum_w = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0) - tail
    cum_h = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) - tail

    bin_idx = (x.unsqueeze(-1) >= cum_w[..., :-1]).sum(dim=-1) - 1
    bin_idx = bin_idx.clamp(0, widths.shape[-1] - 1)

    idx = bin_idx.unsqueeze(-1)
    w_k = widths.gather(-1, idx).squeeze(-1)
    h_k = heights.gather(-1, idx).squeeze(-1)
    d_k = derivatives[..., :-1].gather(-1, idx).squeeze(-1)
    d_k1 = derivatives[..., 1:].gather(-1, idx).squeeze(-1)
    x_k = cum_w[..., :-1].gather(-1, idx).squeeze(-1)
    y_k = cum_h[..., :-1].gather(-1, idx).squeeze(-1)

    s_k = h_k / w_k
    xi = ((x - x_k) / w_k).clamp(0.0, 1.0)

    num = h_k * (s_k * xi ** 2 + d_k * xi * (1 - xi))
    den = s_k + ((d_k + d_k1 - 2 * s_k) * xi * (1 - xi))
    z = y_k + num / den

    dnum = 2 * s_k * xi * (1 - xi) + d_k * (1 - xi) ** 2 + d_k1 * xi ** 2
    log_dz = torch.log(s_k ** 2 * dnum) - 2 * torch.log(den.abs() + 1e-8)

    return z, log_dz


def _rqs_inverse_batched(z, widths, heights, derivatives, tail):
    """
    Vectorised RQS inverse pass.
    z:           (batch, D)
    widths:      (batch, D, K)
    heights:     (batch, D, K)
    derivatives: (batch, D, K+1)
    returns:     x (batch, D)
    """
    cum_w = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0) - tail
    cum_h = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) - tail

    bin_idx = (z.unsqueeze(-1) >= cum_h[..., :-1]).sum(dim=-1) - 1
    bin_idx = bin_idx.clamp(0, widths.shape[-1] - 1)

    idx = bin_idx.unsqueeze(-1)
    w_k = widths.gather(-1, idx).squeeze(-1)
    h_k = heights.gather(-1, idx).squeeze(-1)
    d_k = derivatives[..., :-1].gather(-1, idx).squeeze(-1)
    d_k1 = derivatives[..., 1:].gather(-1, idx).squeeze(-1)
    x_k = cum_w[..., :-1].gather(-1, idx).squeeze(-1)
    y_k = cum_h[..., :-1].gather(-1, idx).squeeze(-1)

    s_k = h_k / w_k
    zeta = z - y_k

    a = h_k * (s_k - d_k) + zeta * (d_k + d_k1 - 2 * s_k)
    b = h_k * d_k - zeta * (d_k + d_k1 - 2 * s_k)
    c = -s_k * zeta

    disc = (b ** 2 - 4 * a * c).clamp(min=0.0)
    xi = (2 * c / (-b - torch.sqrt(disc))).clamp(0.0, 1.0)

    return xi * w_k + x_k


class RQSCouplingBlock(nn.Module):
    def __init__(self, x_dim, context_dim, width, K=16, tail=6.0, **_ignored):
        super().__init__()
        self.x1_dim = x_dim // 2
        self.x2_dim = x_dim - self.x1_dim
        self.K = K
        self.tail = tail
        self.n_params_per_dim = 3 * K + 1
        self.subnet = make_subnet(
            self.x1_dim + context_dim,
            self.x2_dim * self.n_params_per_dim,
            width=width,
        )

    def _params(self, x1, context):
        raw = self.subnet(torch.cat([x1, context], dim=-1))
        raw = raw.view(*raw.shape[:-1], self.x2_dim, self.n_params_per_dim)
        W = raw[..., :self.K]
        H = raw[..., self.K: 2 * self.K]
        D = raw[..., 2 * self.K:]
        return (
            F.softmax(W, dim=-1) * 2 * self.tail,
            F.softmax(H, dim=-1) * 2 * self.tail,
            F.softplus(D),
        )

    def forward(self, x, context):
        x1, x2 = x[:, :self.x1_dim], x[:, self.x1_dim:]
        widths, heights, derivatives = self._params(x1, context)

        inside = ((x2 >= -self.tail) & (x2 <= self.tail)).float()
        z2_in, ld_in = _rqs_forward_batched(x2, widths, heights, derivatives, self.tail)
        z2 = inside * z2_in + (1.0 - inside) * x2
        log_det = (inside * ld_in).sum(dim=-1)

        return torch.cat([x1, z2], dim=-1), log_det

    def inverse(self, z, context):
        z1, z2 = z[:, :self.x1_dim], z[:, self.x1_dim:]
        widths, heights, derivatives = self._params(z1, context)

        inside = ((z2 >= -self.tail) & (z2 <= self.tail)).float()
        x2_in = _rqs_inverse_batched(z2, widths, heights, derivatives, self.tail)
        x2 = inside * x2_in + (1.0 - inside) * z2

        return torch.cat([z1, x2], dim=-1)


# ──────────────────────────────────────────────────────────────────────────
# Fixed random permutation
# ──────────────────────────────────────────────────────────────────────────

class Permutation(nn.Module):
    def __init__(self, dim: int, seed: int):
        super().__init__()
        rng = np.random.RandomState(seed)
        perm = rng.permutation(dim).astype(np.int64)
        inv = np.argsort(perm).astype(np.int64)
        self.register_buffer("perm", torch.tensor(perm))
        self.register_buffer("inv", torch.tensor(inv))

    def forward(self, x):
        return x[:, self.perm]

    def inverse(self, x):
        return x[:, self.inv]


# ──────────────────────────────────────────────────────────────────────────
# cINN
# ──────────────────────────────────────────────────────────────────────────

COUPLING_REGISTRY = {
    "affine": AffineCouplingBlock,
    "rqs": RQSCouplingBlock,
}


class cINN(nn.Module):
    def __init__(
        self,
        target_dim: int,
        context_dim: int,
        n_blocks: int = 24,
        coupling: str = "rqs",
        cond_width: int = 512,
        cond_depth: int = 3,
        subnet_width: int = 256,
        spline_bins: int = 16,
        spline_tail: float = 6.0,
        affine_clamp: float = 2.0,
    ):
        super().__init__()

        if coupling not in COUPLING_REGISTRY:
            raise ValueError(
                f"Unknown coupling type '{coupling}'. "
                f"Available: {list(COUPLING_REGISTRY.keys())}"
            )

        self.cond_net = ConditioningSubnet(context_dim, width=cond_width, depth=cond_depth)
        c_dim = self.cond_net.out_dim

        CouplingClass = COUPLING_REGISTRY[coupling]
        self.blocks = nn.ModuleList()
        self.perms = nn.ModuleList()

        for k in range(n_blocks):
            self.blocks.append(
                CouplingClass(
                    target_dim,
                    c_dim,
                    width=subnet_width,
                    K=spline_bins,
                    tail=spline_tail,
                    clamp=affine_clamp,
                )
            )
            self.perms.append(Permutation(target_dim, seed=k))

    def forward(self, x, context):
        c = self.cond_net(context)
        log_det = torch.zeros(x.shape[0], device=x.device)

        for block, perm in zip(self.blocks, self.perms):
            x, ld = block(x, c)
            log_det += ld
            x = perm(x)

        return x, log_det

    def inverse(self, z, context):
        c = self.cond_net(context)

        for block, perm in zip(reversed(self.blocks), reversed(self.perms)):
            z = perm.inverse(z)
            z = block.inverse(z, c)

        return z


# ──────────────────────────────────────────────────────────────────────────
# Config-driven builder
# ──────────────────────────────────────────────────────────────────────────

def build_model_from_config(config: dict, device: str = "cpu") -> cINN:
    """
    Build a cINN from a loaded YAML config (see configs/no_energy.yaml).

    Expects config["model"] to hold the architecture hyperparameters and
    config["truth"]["dim"] / config["reco"]["dim"] for target_dim /
    context_dim. This mirrors MODEL_CONFIG as saved in the training
    checkpoint, so the same config also documents what a checkpoint was
    trained with.
    """
    m = config["model"]
    model = cINN(
        target_dim=config["truth"]["dim"],
        context_dim=config["reco"]["dim"],
        n_blocks=m.get("n_blocks", 24),
        coupling=m.get("coupling", "rqs"),
        cond_width=m.get("cond_width", 512),
        cond_depth=m.get("cond_depth", 3),
        subnet_width=m.get("subnet_width", 256),
        spline_bins=m.get("spline_bins", 16),
        spline_tail=m.get("spline_tail", 6.0),
        affine_clamp=m.get("affine_clamp", 2.0),
    )
    return model.to(device)


def load_checkpoint(model: cINN, checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Load a training checkpoint's model_state_dict into `model` in place.
    Returns the full checkpoint dict (epoch, val_loss, model_config, ...)
    in case the caller wants to inspect / sanity-check it.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint
