# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLP archive validation utilities for CUTracer trace files.

This module provides validation functions for CLP (Compressed Log Processor)
archive format, which is Mode 3 trace compression in CUTracer.
"""

from pathlib import Path
from typing import Union


class ClpValidationError(Exception):
    """Raised when CLP archive validation fails."""

    pass


def detect_clp_archive(filepath: Union[str, Path]) -> bool:
    """
    Detect if a file is a CLP archive.

    Detection is based on the .clp file extension.

    Args:
        filepath: Path to the file to check

    Returns:
        True if file is a CLP archive, False otherwise

    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return filepath.suffix.lower() == ".clp"
