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
from .sourceproperties import VIDEO_INPUTS, face_video_index

MOCAP_PROPERTIES_DIR = os.path.join(os.path.dirname(__file__), "properties")
MOCAP_PROPERTIES = SceneConfigSet.from_definitions_in_json_directory(MOCAP_PROPERTIES_DIR, prefix="MOCAP_")

_DEFAULTS = {
    "video_path": "", "video_pose": True, "video_hands": True, "video_face": True, "pose_model": "FULL",
    "video_start": 0.0, "video_end": 0.0, "min_confidence": 0.5, "annotate": True,
    "audio_path": "", "audio_start": 0.0, "audio_end": 0.0, "audio_device": "AUTO",
    }
_PATHS = ("video_path", "audio_path")


def _value(scene: bpy.types.Scene, name: str):
    value = MOCAP_PROPERTIES.get_value(name, _DEFAULTS[name], entity_reference=scene)
    if name in _PATHS:
        value = str(value or "").strip()
        value = os.path.normpath(bpy.path.abspath(value)) if value else ""
    return value


def picked_video(scene: bpy.types.Scene) -> dict:
    """The video in the picker of Run Models with its settings. Face is not selectable when other video already has it."""
    return {"path": _value(scene, "video_path"), "pose": _value(scene, "video_pose"), "hands": _value(scene, "video_hands"),
            "face": _value(scene, "video_face") and face_video_index(scene) < 0,
            "pose_model": _value(scene, "pose_model"), "min_confidence": _value(scene, "min_confidence"),
            "start": _value(scene, "video_start"), "end": _value(scene, "video_end"), "annotate": _value(scene, "annotate")}


def read_settings(scene: bpy.types.Scene) -> dict:
    """The panel values as dict for MocapRunService, with the absolute paths."""
    videos = [video.as_settings() for video in getattr(scene, VIDEO_INPUTS, [])]
    picked = picked_video(scene)
    if picked["path"]:
        videos.append(picked)
    settings = {"videos": videos}
    for name in ("audio_path", "audio_start", "audio_end", "audio_device"):
        settings[name] = _value(scene, name)
    return settings
