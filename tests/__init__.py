"""Test package bootstrap for converter project."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when tests are executed directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

