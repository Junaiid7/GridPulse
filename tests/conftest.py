"""Test bootstrap: make ``tests/`` importable so tests use
``from support.synthetic_electricity import ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))