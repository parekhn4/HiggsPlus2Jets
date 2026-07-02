"""
Closure plots comparing hard truth, detector-level reco, and flow (unfolded)
observables. Ported from the training notebook, "Cell 14: Closure plots".

This is a diagnostic tool for when truth is available (e.g. running the
model on a held-out validation split during development), not part of
the plain inference path — Pratik's inference-only ROOT files have no
truth branches, so these plots aren't part of `inference.py`'s flow.
Kept here so the repo can grow into full closure-testing later without
restructuring.

Typical usage, given truth_obs / reco_obs / flow_obs dicts of the form
{"dphi": array, "H_pt": array, ...} (see get_observables() below to build
these from truth/reco/flow kinematic arrays):

    from plotting import plot_closure
    fig = plot_closure(truth_obs, reco_obs, flow_obs)
    fig.savefig("closure.png")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────
# Observable dict builder
# ──────────────────────────────────────────────────────────────────────────

def get_observables(pt_H, eta_H, phi_H, pt_j1, eta_j1, phi_j1,
                     pt_j2, eta_j2, dphi_jj) -> dict:
    """
    Build the standard observable dict from raw kinematics. `dphi_jj`
    should already be the wrapped delta_phi(phi_j1, phi_j2) — pass
    phi_from_features(block(arr, "j2")) directly for truth/flow samples,
    since that slot already stores delta_phi_jj.
    """
    return {
        "H_pt": pt_H, "H_eta": eta_H, "H_phi": phi_H,
        "j1_pt": pt_j1, "j1_eta": eta_j1,
        "j2_pt": pt_j2, "j2_eta": eta_j2,
        "dphi": dphi_jj,
        "deta": eta_j1 - eta_j2,
    }


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
