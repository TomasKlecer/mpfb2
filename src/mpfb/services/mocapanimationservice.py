# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         26.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  creating animation from the saved outputs of the motion capture models
# ------------------------------------------------------------------------------
"""Turns the raw outputs of the models into animation of the MPFB2 character.

Face: the 52 blendshapes of the mediapipe face landmarker (or a Live Link Face csv) are ARKit names.
Curves are cleaned up (gaps, neutral face, smoothing, gain), resampled to the scene fps and written as action on the shape keys of the basemesh.
Meshes fitted to the basemesh (eyes, eyelashes, eyebrows, clothes) can follow it through drivers.
"""

import math
import os
import re
from typing import Any

import bpy
import numpy as np

from .logservice import LogService
from .objectservice import ObjectService
from .clothesservice import ClothesService
from .faceservice import FaceService, ARKIT_FACEUNITS
from .mocapsourceservice import MocapSourceService

_LOG = LogService.get_logger("services.mocapanimationservice")

# Unused value Mediapipe adds to ARKit names
MEDIAPIPE_NEUTRAL = "_neutral"

# Gaps in the detection up to this long are interpolated, longer ones fade to the neutral face
MAX_INTERPOLATED_GAP = 0.5
GAP_FADE = 0.2

# One Euro filter: smoothens stornger for slow expression changes to minimize noise and weaker for fast changes so it does not delay them
ONE_EURO_BETA = 0.5
ONE_EURO_BETA_REDUCTION = 0.7
ONE_EURO_D_CUTOFF = 1.0

# Names of the basemesh children the face units were interpolated to (so it is done only once)
INTERPOLATED_CHILDREN_PROP = "mpfb_mocap_faceunits_children"

DEFAULT_FACE_SETTINGS = {
    "smoothing": 0.5,
    "gain": 1.0,
    "use_neutral": False,
    "neutral_time": 0.0,
    "mirror": False,
    "extend_scene": True,
    "include_children": True,
    }


class MocapAnimationService():
    """Creating animation from the outputs of the models."""

    def __init__(self) -> None:
        raise RuntimeError("You should not instance MocapAnimationService. Use its static methods instead.")

    # ------- Face curves ---------

    @staticmethod
    def load_face_source(path: str) -> tuple[np.ndarray, list[str], np.ndarray]:
        """returns times in seconds, ARKit names and their values per frame (NaN where no face was detected)."""
        if path.lower().endswith(".csv"):
            return MocapSourceService.read_livelink_csv(path)
        with np.load(path, allow_pickle=False) as data:
            times = np.asarray(data["times_ms"], dtype=np.float64) / 1000.0
            names = [str(name) for name in data["blendshape_names"]]
            values = np.asarray(data["blendshapes"], dtype=np.float64)
        keep = [index for index, name in enumerate(names) if name != MEDIAPIPE_NEUTRAL]
        return times, [names[index] for index in keep], values[:, keep]

    @staticmethod
    def mirror_name(name: str) -> str:
        """Swaps eyeBlinkLeft and eyeBlinkRight. For videos where the sides of the face come out swapped."""
        if name.endswith("Left"):
            return name[:-4] + "Right"
        if name.endswith("Right"):
            return name[:-5] + "Left"
        return name

    @staticmethod
    def fill_gaps(times: np.ndarray, values: np.ndarray, max_gap: float = MAX_INTERPOLATED_GAP,
                  fade: float = GAP_FADE) -> np.ndarray:
        """Frames without a detection: short gaps are interpolated linearly, longer gaps (and the start/end of the clip) fade to 0 within `fade` seconds."""
        result = values.copy()
        valid = ~np.isnan(values).any(axis=1)
        if not valid.any():
            return np.zeros_like(values)
        index = 0
        count = len(times)
        while index < count:
            if valid[index]:
                index += 1
                continue
            end = index
            while end < count and not valid[end]:
                end += 1
            before = index - 1 if index > 0 else None
            after = end if end < count else None
            if before is not None and after is not None and times[after] - times[before] <= max_gap:
                weight = ((times[index:end] - times[before]) / (times[after] - times[before]))[:, None]
                result[index:end] = values[before] * (1.0 - weight) + values[after] * weight
            else:
                fade_out = np.zeros((end - index, values.shape[1]))
                if before is not None:
                    weight = np.clip(1.0 - (times[index:end] - times[before]) / fade, 0.0, 1.0)[:, None]
                    fade_out = np.maximum(fade_out, values[before] * weight)
                if after is not None:
                    weight = np.clip(1.0 - (times[after] - times[index:end]) / fade, 0.0, 1.0)[:, None]
                    fade_out = np.maximum(fade_out, values[after] * weight)
                result[index:end] = fade_out
            index = end
        return result

    @staticmethod
    def _one_euro(times: np.ndarray, values: np.ndarray, min_cutoff: float, beta: float) -> np.ndarray:
        """One pass of One Euro filter (Casiez, Roussel, Vogel 2012) over every column.
        Low cutoff when the value moves slowly, higher when it moves fast."""
        def alpha(cutoff, step):
            tau = 1.0 / (2.0 * math.pi * cutoff)
            return 1.0 / (1.0 + tau / step)

        result = np.empty_like(values)
        result[0] = values[0]
        derivative = np.zeros(values.shape[1])
        for index in range(1, len(values)):
            step = max(times[index] - times[index - 1], 1e-6)
            raw_derivative = (values[index] - result[index - 1]) / step
            derivative_alpha = alpha(ONE_EURO_D_CUTOFF, step)
            derivative = derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * derivative
            value_alpha = alpha(min_cutoff + beta * np.abs(derivative), step)
            result[index] = value_alpha * values[index] + (1.0 - value_alpha) * result[index - 1]
        return result

    @staticmethod
    def smooth(times: np.ndarray, values: np.ndarray, strength: float) -> np.ndarray:
        """One Euro filter run forward and then backward, so the curves are not shifted in time.
        Strength 0 keeps the values, 1 is the strongest smoothing."""
        if strength <= 0.0 or len(values) < 3:
            return values.copy()
        strength = min(strength, 1.0)
        # 0 -> 10 Hz (almost nothing), 0.5 -> 1.8 Hz, 1 -> 0.3 Hz
        min_cutoff = 10.0 ** (1.0 - 1.5 * strength)
        beta = ONE_EURO_BETA * (1.0 - ONE_EURO_BETA_REDUCTION * strength)
        forward = MocapAnimationService._one_euro(times, values, min_cutoff, beta)
        backward = MocapAnimationService._one_euro(times[-1] - times[::-1], forward[::-1], min_cutoff, beta)
        return backward[::-1]

    @staticmethod
    def remove_neutral(values: np.ndarray, neutral: np.ndarray) -> np.ndarray:
        """Subtract the neutral face of the actor and stretch the rest back to 0 - 1."""
        return np.clip((values - neutral) / np.maximum(1.0 - neutral, 1e-3), 0.0, 1.0)

    @staticmethod
    def prepare_face_curves(path: str, settings: dict[str, Any], fps: float) -> dict[str, Any]:
        """The face source as values per frame of the scene:

            names    ARKit names
            values   (frames, names) array, values 0 - 1
            frames   number of scene frames the curve is in
            start    time in seconds of the first frame in the source (for alignment with other sources)
        """
        settings = dict(DEFAULT_FACE_SETTINGS, **settings)
        times, names, values = MocapAnimationService.load_face_source(path)
        if not len(times) or np.isnan(values).all():
            raise ValueError("No face was detected in " + os.path.basename(path))
        values = MocapAnimationService.fill_gaps(times, values)

        if settings["use_neutral"]:
            row = int(np.argmin(np.abs(times - (times[0] + float(settings["neutral_time"])))))
            values = MocapAnimationService.remove_neutral(values, values[row])
        values = MocapAnimationService.smooth(times, values, float(settings["smoothing"]))
        values = np.clip(values * float(settings["gain"]), 0.0, 1.0)

        if settings["mirror"]:
            names = [MocapAnimationService.mirror_name(name) for name in names]

        if len(times) == 1:
            frames = 1
            resampled = values
        else:
            frames = int(math.floor((times[-1] - times[0]) * fps + 1e-6)) + 1
            frame_times = times[0] + np.arange(frames) / fps
            resampled = np.stack([np.interp(frame_times, times, values[:, column]) for column in range(len(names))], axis=1)
        return {"names": names, "values": resampled, "frames": frames, "start": float(times[0])}

    # ----------- Blender -------------

    @staticmethod
    def ensure_face_units(basemesh: bpy.types.Object) -> list[str]:
        """Finding or loading the ARKit face units on the basemesh."""
        keys = basemesh.data.shape_keys
        present = [name for name in ARKIT_FACEUNITS if keys and name in keys.key_blocks]
        if not present:
            if not FaceService.is_faceunits01_installed():
                raise RuntimeError("The faceunits01 asset pack is not installed")
            _LOG.info("Loading the ARKit face units onto", basemesh.name)
            FaceService.load_targets(basemesh, load_microsoft_visemes=False, load_meta_visemes=False,
                                     load_arkit_faceunits=True)
            keys = basemesh.data.shape_keys
            present = [name for name in ARKIT_FACEUNITS if keys and name in keys.key_blocks]
        return present

    @staticmethod
    def fitted_children(basemesh: bpy.types.Object) -> list[bpy.types.Object]:
        """Meshes fitted to the basemesh (eyes, eyelashes, eyebrows, clothes)."""
        root = basemesh.parent if basemesh.parent else basemesh
        return [child for child in ObjectService.get_list_of_children(root)
                if child != basemesh and child.type == 'MESH' and ClothesService.find_clothes_absolute_path(child)]

    @staticmethod
    def interpolate_to_children(basemesh: bpy.types.Object) -> None:
        """Interpolate the face units to the fitted meshes which did not get them yet."""
        children = sorted(child.name for child in MocapAnimationService.fitted_children(basemesh))
        done = set(str(basemesh.get(INTERPOLATED_CHILDREN_PROP, "")).split("|")) - {""}
        if set(children) <= done:
            return
        _LOG.info("Interpolating the face units to", [name for name in children if name not in done])
        FaceService.interpolate_targets(basemesh)
        basemesh[INTERPOLATED_CHILDREN_PROP] = "|".join(sorted(done | set(children)))

    @staticmethod
    def drive_children(basemesh: bpy.types.Object, names: list[str]) -> int:
        """Makes the fitted meshes (eyes, lashes, clothes) follow the basemesh face animation via drivers."""
        source = basemesh.data.shape_keys
        driven = 0
        for child in MocapAnimationService.fitted_children(basemesh):
            keys = child.data.shape_keys
            if not keys:
                continue
            added = False
            for name in names:
                if name not in keys.key_blocks or name not in source.key_blocks:
                    continue
                data_path = 'key_blocks["%s"].value' % name
                if keys.animation_data and keys.animation_data.drivers.find(data_path):
                    added = True
                    continue
                driver = keys.key_blocks[name].driver_add("value").driver
                driver.type = 'AVERAGE'
                variable = driver.variables.new()
                variable.name = "value"
                variable.type = 'SINGLE_PROP'
                variable.targets[0].id_type = 'KEY'
                variable.targets[0].id = source
                variable.targets[0].data_path = data_path
                added = True
            if added:
                driven += 1
        return driven

    @staticmethod
    def _new_fcurve_collection(key: bpy.types.Key, action: bpy.types.Action) -> Any:
        """Assign the action to the shape keys and return the collection to add its fcurves to. Blender 4.4+ keeps the
        fcurves in a channelbag of an action slot, older versions directly in action.fcurves."""
        animation_data = key.animation_data or key.animation_data_create()
        if not hasattr(action, "slots"):
            animation_data.action = action
            return action.fcurves  # pyright: ignore[reportAttributeAccessIssue]
        slot = action.slots.new(id_type='KEY', name=key.name)
        animation_data.action = action
        animation_data.action_slot = slot
        try:
            from bpy_extras import anim_utils  # pylint: disable=C0415
            channelbag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
        except (ImportError, AttributeError):
            layer = action.layers[0] if len(action.layers) else action.layers.new("Layer")
            strip = layer.strips[0] if len(layer.strips) else layer.strips.new(type='KEYFRAME')
            channelbag = strip.channelbag(slot, ensure=True)
        return channelbag.fcurves

    @staticmethod
    def write_shape_key_action(key: bpy.types.Key, action_name: str, frame_start: int, names: list[str],
                               values: np.ndarray) -> bpy.types.Action:
        """Creates new action with one fcurve per shape key. Action keeps fake user, so it survives assingin of new animation."""
        action = bpy.data.actions.new(action_name)
        action.use_fake_user = True
        fcurves = MocapAnimationService._new_fcurve_collection(key, action)
        frames = frame_start + np.arange(len(values), dtype=np.float64)
        for column, name in enumerate(names):
            if name not in key.key_blocks:
                continue
            fcurve = fcurves.new('key_blocks["%s"].value' % name, index=0)
            points = fcurve.keyframe_points
            points.add(len(values))
            coordinates = np.empty(2 * len(values))
            coordinates[0::2] = frames
            coordinates[1::2] = values[:, column]
            points.foreach_set("co", coordinates)
            for point in points:
                point.interpolation = 'LINEAR'
            fcurve.update()
        return action

    @staticmethod
    def action_name_for(path: str, input_name: str | None) -> str:
        stem = os.path.splitext(input_name or os.path.basename(path))[0]
        stem = re.sub(r"_landmarks$", "", stem)
        return "MPFB_Face_" + stem

    @staticmethod
    def create_face_animation(basemesh: bpy.types.Object, path: str, settings: dict[str, Any], scene: bpy.types.Scene,
                              input_name: str | None = None) -> dict[str, Any]:
        """Create the face animation of the character from a face source, starting at the current frame of the scene."""
        settings = dict(DEFAULT_FACE_SETTINGS, **settings)
        fps = scene.render.fps / scene.render.fps_base
        curves = MocapAnimationService.prepare_face_curves(path, settings, fps)

        present = MocapAnimationService.ensure_face_units(basemesh)
        if settings["include_children"]:
            MocapAnimationService.interpolate_to_children(basemesh)

        key = basemesh.data.shape_keys
        frame_start = scene.frame_current
        names = curves["names"]
        animated = [name for name in names if name in present]
        action = MocapAnimationService.write_shape_key_action(
            key, MocapAnimationService.action_name_for(path, input_name), frame_start, names, curves["values"])

        driven = MocapAnimationService.drive_children(basemesh, animated) if settings["include_children"] else 0
        frame_end = frame_start + curves["frames"] - 1
        if settings["extend_scene"] and scene.frame_end < frame_end:
            scene.frame_end = frame_end

        result = {"action": action.name, "frame_start": frame_start, "frame_end": frame_end, "animated": len(animated),
                  "missing": [name for name in ARKIT_FACEUNITS if name not in animated], "children": driven}
        _LOG.info("Face animation created", result)
        return result
