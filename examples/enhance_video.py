from __future__ import annotations

import argparse
from pathlib import Path

from nightjet.inference import DEFAULT_WEIGHTS_PATH, NightJetEnhancer


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance one video with NightJet.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/nightjet-video.mp4"))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--preserve-color", action="store_true")
    args = parser.parse_args()

    enhancer = NightJetEnhancer.from_checkpoint(args.weights, device=args.device)
    enhancer.enhance_video(
        args.input,
        args.output,
        fps=args.fps,
        side_by_side=args.side_by_side,
        preserve_color=args.preserve_color,
    )
    print(args.output)


if __name__ == "__main__":
    main()
