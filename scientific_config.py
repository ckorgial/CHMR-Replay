"""Reproducible run configuration shared by the original scientific experiments."""
from __future__ import annotations
import os
from pathlib import Path

SUITES = ("MR", "PA", "KITTI")


def selected_suites():
    value = os.environ.get("CHMR_SUITES", "MR,PA,KITTI")
    suites = tuple(token.strip().upper() for token in value.split(",") if token.strip())
    if not suites or len(set(suites)) != len(suites) or any(s not in SUITES for s in suites):
        raise ValueError("CHMR_SUITES must be a nonempty subset of MR,PA,KITTI")
    return suites


def cache_directory():
    return Path(os.environ.get("CHMR_CACHE", "cache"))
