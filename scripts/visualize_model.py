"""
Usage
    python scripts/visualize_model.py --checkpoint best_model.pt --output arch.png --depth 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchview import draw_graph

from core.model import build_model_from_config
from core.kinematics import total_dim


def load_model_for_visualization(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved = checkpoint["resolved_config"]
    max_jets = checkpoint["max_jets"]
    reco_dim = total_dim(resolved["reco"], max_jets=max_jets)
    truth_dim = total_dim(resolved["truth"], max_jets=max_jets)

    model = build_model_from_config(
        {"model": checkpoint["model_config"]}, target_dim=truth_dim, context_dim=reco_dim, device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, truth_dim, reco_dim


def visualize(args: argparse.Namespace) -> None:
    model, truth_dim, reco_dim = load_model_for_visualization(args.checkpoint)
    print(f"truth_dim={truth_dim}, reco_dim={reco_dim}, depth={args.depth}")

    x = torch.randn(args.batch_size, truth_dim)
    c = torch.randn(args.batch_size, reco_dim)

    graph = draw_graph(
        model, input_data=(x, c), depth=args.depth, expand_nested=False,
        graph_name=Path(args.output).stem, save_graph=False,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = out_path.suffix.lstrip(".") or "png"
    graph.visual_graph.render(str(out_path.with_suffix("")), format=fmt, cleanup=True)
    print(f"wrote {out_path}")

    if args.depth >= 2:
        print("note: depth >= 2 unfolds each block's internals -- this can get very "
              "large for a real (many-block) model; depth=1 is recommended there.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a torchview architecture diagram from a checkpoint's own embedded config."
    )
    p.add_argument("--checkpoint", required=True, help="Path to a trained model checkpoint (.pt)")
    p.add_argument("--output", default="model_architecture.png",
                    help="Output image path (default: model_architecture.png)")
    p.add_argument("--depth", type=int, default=1,
                    help="Module-hierarchy depth to unfold (default: 1, recommended for the full "
                         "model -- see module docstring for why higher values explode in size)")
    p.add_argument("--batch-size", type=int, default=2,
                    help="Fake batch size for the example trace (default: 2, doesn't affect architecture)")
    return p


def main():
    args = build_arg_parser().parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
