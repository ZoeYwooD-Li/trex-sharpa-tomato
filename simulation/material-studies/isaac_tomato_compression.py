#!/usr/bin/env python3
"""Compatibility entry point for the corrected single-tomato FEM baseline."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("tomato_fem_compression.py")), run_name="__main__")
