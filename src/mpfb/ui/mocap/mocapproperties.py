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
from ...services.mocapanimationservice import DEFAULT_FACE_SETTINGS
from ...services.mocaplipsyncservice import DEFAULT_LIPSYNC_SETTINGS
from .sourceproperties import VIDEO_INPUTS, AUDIO_SOURCE, face_video_index, main_body_index, absolute

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


_FACE_SETTINGS = {
    "face_smoothing": "smoothing", "face_gain": "gain", "face_use_neutral": "use_neutral",
    "face_neutral_time": "neutral_time", "face_mirror": "mirror", "face_extend_scene": "extend_scene",
    "face_include_children": "include_children", "face_head_rotation": "head_rotation", "face_neck_share": "neck_share",
    }


def read_face_settings(scene: bpy.types.Scene) -> dict:
    """Settings of the face animation for MocapAnimationService."""
    settings = {}
    for name, key in _FACE_SETTINGS.items():
        settings[key] = MOCAP_PROPERTIES.get_value(name, DEFAULT_FACE_SETTINGS[key], entity_reference=scene)
    settings["skip_speech"] = speech_from_audio(scene)
    settings["head_rotation"] = bool(settings["head_rotation"]) and not head_from_body(scene)
    return settings


def head_from_body(scene: bpy.types.Scene) -> bool:
    """True when the head rotation is taken from body animation instead of head (e.g. for face video from helmet camera that turns with head)."""
    if main_body_index(scene) < 0:
        return False
    return bool(MOCAP_PROPERTIES.get_value("face_head_from_body", False, entity_reference=scene))


def speech_from_audio(scene: bpy.types.Scene) -> bool:
    """True when the mouth is left to the lipsync: an audio source is picked and the speech is not kept from video."""
    if not absolute(getattr(scene, AUDIO_SOURCE, "")):
        return False
    return not MOCAP_PROPERTIES.get_value("face_keep_speech", False, entity_reference=scene)


_LIPSYNC_SETTINGS = {
    "lipsync_strength": "strength", "lipsync_lead": "lead", "lipsync_transition": "transition",
    "lipsync_add_sound": "add_sound", "face_extend_scene": "extend_scene", "face_include_children": "include_children",
    }


def read_lipsync_settings(scene: bpy.types.Scene) -> dict:
    """Settings of the lipsync for MocapAnimationService."""
    settings = {}
    for name, key in _LIPSYNC_SETTINGS.items():
        settings[key] = MOCAP_PROPERTIES.get_value(name, DEFAULT_LIPSYNC_SETTINGS[key], entity_reference=scene)
    return settings
