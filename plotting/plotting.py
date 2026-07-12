"""
Usage
    from plotting import plot_closure
    fig = plot_closure(truth_obs, reco_obs, flow_obs)
    fig.savefig("closure.png")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import core.kinematics as kinematics


# ──────────────────────────────────────────────────────────────────────────
# Histogram helpers
# ──────────────────────────────────────────────────────────────────────────

def hist_density(values, bins):
    counts, edges = np.histogram(values, bins=bins, density=False)
    widths = np.diff(edges)
    area = np.sum(counts * widths)
    density = counts / area if area > 0 else np.zeros_like(counts, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return density, centers


def safe_ratio(num, den):
    out = np.full_like(num, np.nan, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


# ──────────────────────────────────────────────────────────────────────────
# Default plot specs — (observable key, bin edges, x-axis label, title)
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_PLOT_SPECS = [
    ("dphi", np.linspace(-np.pi, np.pi, 20), r"$\Delta\phi_{jj}$", r"$\Delta\phi_{jj}$"),
    ("deta", np.linspace(-8, 8, 33), r"$\Delta\eta_{jj}$", r"$\Delta\eta_{jj}$"),
    ("j1_pt", np.linspace(0, 300, 31), r"$p_{T,j1}$ [GeV]", r"$p_{T,j1}$"),
    ("j2_pt", np.linspace(0, 300, 31), r"$p_{T,j2}$ [GeV]", r"$p_{T,j2}$"),
    ("j1_eta", np.linspace(-6, 6, 31), r"$\eta_{j1}$", r"$\eta_{j1}$"),
    ("j2_eta", np.linspace(-6, 6, 31), r"$\eta_{j2}$", r"$\eta_{j2}$"),
    ("H_pt", np.linspace(0, 400, 41), r"$p_{T,H}$ [GeV]", r"$p_{T,H}$"),
    ("H_eta", np.linspace(-5, 5, 31), r"$\eta_H$", r"$\eta_H$"),
    ("H_phi", np.linspace(-np.pi, np.pi, 5), r"$\phi_H$", r"$\phi_H$"),
]


# ──────────────────────────────────────────────────────────────────────────
# Residual / error histograms — per-event fidelity, as opposed to
# plot_closure's marginal-distribution comparison.
# ──────────────────────────────────────────────────────────────────────────

ANGULAR_OBSERVABLES = {"H_phi", "dphi", "dphi_eta_ordered"}

DEFAULT_ERROR_SPECS = [
    ("dphi", np.linspace(-1.0, 1.0, 41), r"$\Delta\phi_{jj}$ residual"),
    ("deta", np.linspace(-2.0, 2.0, 41), r"$\Delta\eta_{jj}$ residual"),
    ("j1_pt", np.linspace(-100, 100, 41), r"$p_{T,j1}$ residual [GeV]"),
    ("j2_pt", np.linspace(-100, 100, 41), r"$p_{T,j2}$ residual [GeV]"),
    ("j1_eta", np.linspace(-2, 2, 41), r"$\eta_{j1}$ residual"),
    ("j2_eta", np.linspace(-2, 2, 41), r"$\eta_{j2}$ residual"),
    ("H_pt", np.linspace(-100, 100, 41), r"$p_{T,H}$ residual [GeV]"),
    ("H_eta", np.linspace(-2, 2, 41), r"$\eta_H$ residual"),
    ("H_phi", np.linspace(-1.0, 1.0, 41), r"$\phi_H$ residual"),
]


def observable_residual(key: str, reference: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    """
    reference - comparison, wrapping correctly for angular observables (a
    plain subtraction is wrong there, same issue as kinematics.circular_mean
    solves for averaging). comparison may carry an extra trailing axis --
    reference shape (N,), comparison shape (N,) for one point/event (reco,
    or a per-event mean) or (N, n_samples) for the full posterior, in which
    case every sample gets its own residual against that event's single
    reference value.
    """
    if comparison.ndim > reference.ndim:
        reference = reference[..., None]
    if key in ANGULAR_OBSERVABLES:
        return kinematics.delta_phi(reference, comparison)
    return reference - comparison


def plot_error_histograms(reference_obs: dict, comparisons: list,
                           reference_label: str = "truth",
                           error_specs=None, title: str | None = None, ncols: int = 3):
    """
    Grid of (reference - comparison) residual histograms, one panel per
    observable, with every (comparison_obs, label) pair in `comparisons`
    overlaid in the same panel -- e.g.
        [(reco_obs, "reco"), (mean_obs, "unfolded (mean)"),
         (samples_obs, "unfolded (all samples)")]
    to directly compare which reduction strategy sits tighter around zero,
    rather than plot_closure's marginal-shape comparison.
    """
    if error_specs is None:
        error_specs = DEFAULT_ERROR_SPECS

    nplots = len(error_specs)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i, (key, bins, xlabel) in enumerate(error_specs):
        ax = axes[i // ncols][i % ncols]
        stats = []
        for comparison_obs, label in comparisons:
            diff = observable_residual(key, reference_obs[key], comparison_obs[key]).reshape(-1)
            # density-normalized: comparisons carry wildly different sample
            # counts (one residual/event for reco/mean vs. one/sample for
            # the full posterior), so raw counts would just show whichever
            # has the most points -- only the shape is comparable here
            ax.hist(diff, bins=bins, density=True, histtype="step", linewidth=1.5,
                    label=f"{reference_label} - {label}")
            stats.append(f"{label}: $\\mu$={diff.mean():.3g}, $\\sigma$={diff.std():.3g}")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{xlabel}\n" + "\n".join(stats), fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    for j in range(nplots, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_loss_curve(train_losses: list, val_losses: list, train_eval_losses: list | None = None,
                     title: str | None = None):
    """
    Per-epoch train/val NLL curve, for spotting under/overfitting and
    convergence. train_losses is measured with dropout ON (the actual
    optimization signal, noisier/inflated by dropout); val_losses is
    always dropout-OFF (model.eval()). Those two aren't directly
    comparable on their own -- pass train_eval_losses (the training set's
    NLL also measured in eval mode, dropout OFF) to plot an apples-to-apples
    third curve against val, isolating the real generalization gap from the
    dropout-inflation artifact.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = np.arange(len(train_losses))
    ax.plot(epochs, train_losses, label="train NLL (dropout on)")
    if train_eval_losses is not None:
        ax.plot(epochs, train_eval_losses, label="train NLL (eval, dropout off)")
    ax.plot(epochs, val_losses, label="val NLL")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def plot_closure(truth_obs: dict, reco_obs: dict, flow_obs: dict,
                  plot_specs=None, title: str | None = None, ncols: int = 3):
    """
    Grid of density histograms (truth / detector / flow) with ratio
    panels underneath each, one panel per observable in plot_specs.
    """
    if plot_specs is None:
        plot_specs = DEFAULT_PLOT_SPECS

    nplots = len(plot_specs)
    nrows = int(np.ceil(nplots / ncols))

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.35)

    for i, (key, bins, xlabel, subtitle) in enumerate(plot_specs):
        col = i % ncols
        row = i // ncols

        inner = outer[row, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(inner[0])
        rax = fig.add_subplot(inner[1], sharex=ax)

        truth_h, centers = hist_density(truth_obs[key], bins)
        reco_h, _ = hist_density(reco_obs[key], bins)
        flow_h, _ = hist_density(flow_obs[key], bins)

        reco_ratio = safe_ratio(reco_h, truth_h)
        flow_ratio = safe_ratio(flow_h, truth_h)

        ax.step(centers, truth_h, where="mid", linewidth=1.5, label="hard truth")
        ax.step(centers, reco_h, where="mid", linewidth=1.5, label="detector")
        ax.step(centers, flow_h, where="mid", linewidth=1.5, label="flow")
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(subtitle, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.tick_params(labelbottom=False)

        rax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        rax.step(centers, reco_ratio, where="mid", linewidth=1.5, label="reco/truth")
        rax.step(centers, flow_ratio, where="mid", linewidth=1.5, label="flow/truth")
        rax.set_xlabel(xlabel, fontsize=8)
        rax.set_ylabel("ratio", fontsize=8)
        rax.set_ylim(0.5, 1.5)
        rax.grid(alpha=0.3)
        rax.legend(fontsize=6)

    if title:
        fig.suptitle(title, fontsize=13)

    return fig
