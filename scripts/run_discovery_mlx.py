#!/usr/bin/env python3
"""Repository-local entry point for blinded discovery."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.discovery_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
