"""
Copyright (c) 2025 Simon Rothman. All rights reserved.

directcov: c++ backend for direct binned covariance
"""

from __future__ import annotations

from ._version import version as __version__
from ._core import DirectCov

__all__ = ["__version__", "DirectCov"]
