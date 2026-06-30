#!/usr/bin/env python3
"""
Render the overall-framework and hybrid-architecture diagrams used in the
README to `assets/`.

Usage:
    python scripts/00_render_diagrams.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wtpf.visualization.architecture_diagram import render_all  # noqa: E402


def main() -> None:
    render_all(output_dir=Path(__file__).resolve().parents[1] / "assets")


if __name__ == "__main__":
    main()
