"""Resolve original model inputs without changing model or analysis settings."""
import os
from pathlib import Path

def data_path(filename):
    root = Path(os.environ.get("DANN_DATA_DIR", Path(__file__).resolve().parents[1] / "data")).expanduser().resolve()
    matches = sorted(p for p in root.rglob(filename) if p.is_file()) if root.is_dir() else []
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(f"Missing {filename}. Place the published input in {root}, or set DANN_DATA_DIR to its directory.")
    raise ValueError(f"Multiple files named {filename} under {root}. Use a directory containing one authoritative copy per input.")
