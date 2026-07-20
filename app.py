#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible Streamlit entry — delegates to frontend/app.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "frontend" / "app.py"), run_name="__main__")
