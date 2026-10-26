# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         24.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  scene properties holding the lists of the motion capture panels
# ------------------------------------------------------------------------------
"""Lists of the motion capture panels, stored on the scene:
    Run Models:        the videos added to the list with +, each with its own settings
    Create Animation:  one face output, one audio output, any number of body outputs
"""

import os
import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

from ... import ClassManager
from ...services.mocapsourceservice import MocapSourceService, ROLE_BODY, ERROR

FACE_SOURCE = "MPFB_MOCAP_face_source"
AUDIO_SOURCE = "MPFB_MOCAP_audio_source"
BODY_SOURCES = "MPFB_MOCAP_body_sources"
BODY_INDEX = "MPFB_MOCAP_body_index"
VIDEO_INPUTS = "MPFB_MOCAP_video_inputs"

POSE_MODEL_ITEMS = [("LITE", "Lite", "Fastest, least accurate"), ("FULL", "Full", "Balanced"),
                    ("HEAVY", "Heavy", "Slowest, most accurate")]


class MPFB_PG_Mocap_Video_Input(bpy.types.PropertyGroup):
    """One video / image added to the list of Run Models, with the settings it had when it was added."""
    path: StringProperty(name="File", subtype='FILE_PATH')
    pose: BoolProperty(name="Pose", default=True)
    hands: BoolProperty(name="Hands", default=True)
    face: BoolProperty(name="Face", default=False)
    pose_model: EnumProperty(name="Pose model", items=POSE_MODEL_ITEMS, default="FULL")
    min_confidence: FloatProperty(name="Min. confidence", default=0.5, min=0.0, max=1.0)
    start: FloatProperty(name="Start", default=0.0, min=0.0)
    end: FloatProperty(name="End", default=0.0, min=0.0)
    annotate: BoolProperty(name="Save annotated video", default=True)

    def as_settings(self) -> dict:
        """The video prepared fore MocapRunService."""
        return {"path": absolute(self.path), "pose": self.pose, "hands": self.hands, "face": self.face,
                "pose_model": self.pose_model, "min_confidence": self.min_confidence, "start": self.start,
                "end": self.end, "annotate": self.annotate}


ClassManager.add_class(MPFB_PG_Mocap_Video_Input)


class MPFB_PG_Mocap_Body_Source(bpy.types.PropertyGroup):
    """One video / image output used for the body."""
    path: StringProperty(name="File", description="Saved output of the models (*_landmarks.npz)", subtype='FILE_PATH')
    enabled: BoolProperty(name="Use", description="Use this source for the body animation", default=True)


ClassManager.add_class(MPFB_PG_Mocap_Body_Source)


def register_scene_properties() -> None:
    bpy.types.Scene.MPFB_MOCAP_face_source = StringProperty(
        name="Face source", description="Saved output of the models used for the face (*_landmarks.npz)", default="")
    bpy.types.Scene.MPFB_MOCAP_audio_source = StringProperty(
        name="Audio source", description="Saved output of the phoneme model used for the mouth (*_logits.npz)", default="")
    bpy.types.Scene.MPFB_MOCAP_body_sources = CollectionProperty(type=MPFB_PG_Mocap_Body_Source)
    bpy.types.Scene.MPFB_MOCAP_body_index = IntProperty(default=0)
    bpy.types.Scene.MPFB_MOCAP_video_inputs = CollectionProperty(type=MPFB_PG_Mocap_Video_Input)


def unregister_scene_properties() -> None:
    for name in (FACE_SOURCE, AUDIO_SOURCE, BODY_SOURCES, BODY_INDEX, VIDEO_INPUTS):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)


def absolute(path: str) -> str:
    """A source path as typed in the panel made absolute."""
    path = (path or "").strip()
    return os.path.normpath(bpy.path.abspath(path)) if path else ""


def main_body_index(scene: bpy.types.Scene) -> int:
    """The main body source is the first usable: its timeline and camera are the reference."""
    for index, source in enumerate(getattr(scene, BODY_SOURCES, [])):
        path = absolute(source.path)
        if source.enabled and path:
            problems = MocapSourceService.check_role(MocapSourceService.inspect(path), ROLE_BODY)
            if not any(level == ERROR for level, _message in problems):
                return index
    return -1


def face_video_index(scene: bpy.types.Scene) -> int:
    """Index of the 1 video in the Run Models list which is used for the face."""
    for index, video in enumerate(getattr(scene, VIDEO_INPUTS, [])):
        if video.face:
            return index
    return -1


def fill_sources(scene: bpy.types.Scene, sources: dict) -> list[str]:
    """Put outputs of models into Create Animation slots so user does not have to fill them manually."""
    filled = []
    if sources.get("face"):
        setattr(scene, FACE_SOURCE, sources["face"])
        filled.append("face")
    if sources.get("bodies"):
        bodies = getattr(scene, BODY_SOURCES)
        bodies.clear()
        for path in sources["bodies"]:
            body = bodies.add()
            body.path = path
            body.enabled = True
        setattr(scene, BODY_INDEX, 0)
        filled.append("body")
    if sources.get("audio"):
        setattr(scene, AUDIO_SOURCE, sources["audio"])
        filled.append("audio")
    return filled
