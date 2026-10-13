"""Ensure the backend package directory is importable when running pytest."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
