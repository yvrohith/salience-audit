#!/usr/bin/env python3
"""Repository-local entry point for the Stage 2 candidate tournament."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.tournament_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
