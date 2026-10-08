# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared variables and utility functions for CUTracer."""

import importlib.util


def is_fbcode() -> bool:
    """Check if running in fbcode environment."""
    return importlib.util.find_spec("cutracer.analyze.fb") is not None
