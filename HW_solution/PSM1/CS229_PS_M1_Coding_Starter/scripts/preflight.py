#!/usr/bin/env python3
"""Check dependency versions and integrity of the supplied datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
import pytest  # noqa: E402
from src.data import read_split  # noqa: E402


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("PyTorch:", torch.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("pytest:", pytest.__version__)
    print("CUDA available:", torch.cuda.is_available())
    meta = json.loads((ROOT / "data" / "manifest.json").read_text())
    for name in ("train", "val"):
        file = ROOT / "data" / f"{name}.npz"
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        expected = meta["files"][file.name]["sha256"]
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {file}")
        x, comp = read_split(name)
        assert len(x) == meta[f"{name}_size"]
        assert x.dtype == np.float32 and np.isfinite(x).all()
        assert len(comp) == len(x) and comp.min() >= 0 and comp.max() < 8
        print(f"{name}: {len(x):,} rows, checksum OK")
    print("Preflight passed (student TODOs are not checked here).")


if __name__ == "__main__":
    main()
