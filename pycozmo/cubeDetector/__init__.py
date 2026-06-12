"""
Cube detection and tracking module.

This module provides advanced 3D cube detection and tracking capabilities
using ArUco-like markers and sensor fusion.
"""

from .cubeDetector import CubeFusionTracker, CozmoCubeDetector

__all__ = [
    "CubeFusionTracker",
    "CozmoCubeDetector",
]
