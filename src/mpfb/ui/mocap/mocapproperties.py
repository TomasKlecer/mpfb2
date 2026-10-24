# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  scene properties of the motion capture panel
# ------------------------------------------------------------------------------
"""The inputs of the motion capture panel, stored on the scene (definitions in properties/*.json)."""

import os
import bpy

from ...services.sceneconfigset import SceneConfigSet

MOCAP_PROPERTIES_DIR = os.path.join(os.path.dirname(__file__), "properties")
MOCAP_PROPERTIES = SceneConfigSet.from_definitions_in_json_directory(MOCAP_PROPERTIES_DIR, prefix="MOCAP_")

_DEFAULTS = {
    "video_path": "", "video_pose": True, "video_hands": True, "video_face": True, "pose_model": "FULL",
    "video_start": 0.0, "video_end": 0.0, "min_confidence": 0.5, "annotate": True,
    "audio_path": "", "audio_start": 0.0, "audio_end": 0.0, "audio_device": "AUTO",
    }
_PATHS = ("video_path", "audio_path")


def read_settings(scene: bpy.types.Scene) -> dict:
    """The panel values as dict for MocapRunService, with the absolute paths."""
    settings = {}
    for name, default in _DEFAULTS.items():
        value = MOCAP_PROPERTIES.get_value(name, default, entity_reference=scene)
        if name in _PATHS:
            value = str(value or "").strip()
            value = os.path.normpath(bpy.path.abspath(value)) if value else ""
        settings[name] = value
    return settings
