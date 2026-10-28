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

Head: rotation of the head relative to the camera of the face source (mediapipe facial transformation matrix or
head columns of Live Link Face), shared between the neck and head bones.

Lipsync: visemes02 curves from the phoneme model.

Face and lipsync are separate layers, each action is a strip on its own NLA track of the shape keys ("MPFB Face", "MPFB Lipsync").
The head is a layer on the rig which combines with what is under it (body animation).
They play together in blender and can be exported as separate clips, or baked into one action.
With a lipsync, the speech face units are by default left out of the face layer so the mouth does not move twice.
"""

import math
import os
import re
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix

from .logservice import LogService
from .objectservice import ObjectService
from .clothesservice import ClothesService
from .faceservice import FaceService, ARKIT_FACEUNITS
from .faceservice import META_VISEMES
from .mocapsourceservice import MocapSourceService
from .mocaplipsyncservice import MocapLipsyncService, SPEECH_FACEUNITS, DEFAULT_LIPSYNC_SETTINGS

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

# Names of the basemesh children the face units / visemes were interpolated to (so it is done only once)
INTERPOLATED_CHILDREN_PROP = "mpfb_mocap_faceunits_children"
INTERPOLATED_VISEMES_PROP = "mpfb_mocap_visemes_children"

# NLA tracks of the layers on the shape keys of the basemesh
TRACK_PREFIX = "MPFB "
FACE_TRACK = "MPFB Face"
LIPSYNC_TRACK = "MPFB Lipsync"
BAKED_ACTION = "MPFB_Baked"
# NLA track of the head rotation on the rig
HEAD_TRACK = "MPFB Head"

# Camera of the face source (x to the right of the image, y up, z towards the camera) ->
# rig of the character facing the camera (-Y forward, Z up, X characters left)
_CAMERA_TO_RIG = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# Swapping the sides of the image (yaw and roll change direction)
_MIRROR_CAMERA = np.diag([-1.0, 1.0, 1.0])

DEFAULT_FACE_SETTINGS = {
    "smoothing": 0.5,
    "gain": 1.0,
    "use_neutral": False,
    "neutral_time": 0.0,
    "mirror": False,
    "extend_scene": True,
    "include_children": True,
    "skip_speech": False,
    "head_rotation": True,
    "neck_share": 0.4,
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
                  fade: float = GAP_FADE, signed: bool = False) -> np.ndarray:
        """Frames without a detection: short gaps are interpolated linearly, longer gaps (and the start/end of the clip) fade to 0 within `fade` seconds.
        Signed values (rotations) fade from both sides by adding, shape keys (0 - 1) by taking the larger value."""
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
                combine = np.add if signed else np.maximum
                if before is not None:
                    weight = np.clip(1.0 - (times[index:end] - times[before]) / fade, 0.0, 1.0)[:, None]
                    fade_out = combine(fade_out, values[before] * weight)
                if after is not None:
                    weight = np.clip(1.0 - (times[after] - times[index:end]) / fade, 0.0, 1.0)[:, None]
                    fade_out = combine(fade_out, values[after] * weight)
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

    # ------- Head curves ---------

    @staticmethod
    def euler_to_matrices(angles: np.ndarray) -> np.ndarray:
        """Converts yaw, pitch, roll in radians (frames, 3) to rotations matrices in the camera space (frames, 3, 3).
        Yaw turns around the up axis, pitch around the side axis, roll around the axis towards the camera."""
        yaw, pitch, roll = angles[:, 0], angles[:, 1], angles[:, 2]
        zeros, ones = np.zeros(len(angles)), np.ones(len(angles))

        def stack(rows):
            return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)

        around_y = stack([[np.cos(yaw), zeros, np.sin(yaw)], [zeros, ones, zeros], [-np.sin(yaw), zeros, np.cos(yaw)]])
        around_x = stack([[ones, zeros, zeros], [zeros, np.cos(pitch), -np.sin(pitch)], [zeros, np.sin(pitch), np.cos(pitch)]])
        around_z = stack([[np.cos(roll), -np.sin(roll), zeros], [np.sin(roll), np.cos(roll), zeros], [zeros, zeros, ones]])
        return around_y @ around_x @ around_z

    @staticmethod
    def matrices_to_vectors(matrices: np.ndarray) -> np.ndarray:
        """Converts rotations (frames, 3, 3) to rotation vectors (frames, 3): axis * angle in radians. NaN stays NaN."""
        trace = np.trace(matrices, axis1=1, axis2=2)
        angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        skew = np.stack([matrices[:, 2, 1] - matrices[:, 1, 2], matrices[:, 0, 2] - matrices[:, 2, 0],
                         matrices[:, 1, 0] - matrices[:, 0, 1]], axis=1)
        sine = np.sin(angle)
        with np.errstate(invalid="ignore", divide="ignore"):
            # For small angles angle / sin(angle) -> 1
            factor = np.where(sine > 1e-6, angle / (2.0 * np.where(sine > 1e-6, sine, 1.0)), 0.5)
        return skew * factor[:, None]

    @staticmethod
    def vectors_to_matrices(vectors: np.ndarray) -> np.ndarray:
        """Convert rotation vectors (frames, 3) to rotation matrices (frames, 3, 3). (Rodrigues formula)"""
        angle = np.linalg.norm(vectors, axis=1)
        axis = vectors / np.where(angle > 1e-12, angle, 1.0)[:, None]
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        zeros = np.zeros(len(vectors))
        cross = np.stack([np.stack([zeros, -z, y], axis=-1), np.stack([z, zeros, -x], axis=-1),
                          np.stack([-y, x, zeros], axis=-1)], axis=-2)
        sine = np.sin(angle)[:, None, None]
        cosine = np.cos(angle)[:, None, None]
        return np.eye(3)[None] + sine * cross + (1.0 - cosine) * (cross @ cross)

    @staticmethod
    def load_head_source(path: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Times(sec) and the head rotations (frames, 3, 3) relative to the camera of the source.
        NaN where noface was detected, None when the source has no head rotation."""
        if path.lower().endswith(".csv"):
            head = MocapSourceService.read_livelink_head(path)
            if head is None:
                return None
            times, angles = head
            return times, MocapAnimationService.euler_to_matrices(angles)
        with np.load(path, allow_pickle=False) as data:
            if "face_matrix" not in data.files:
                return None
            times = np.asarray(data["times_ms"], dtype=np.float64) / 1000.0
            matrices = np.asarray(data["face_matrix"], dtype=np.float64)[:, :3, :3]
        # The matrix also scales the canonical face to the size of the actors face
        with np.errstate(invalid="ignore"):
            return times, matrices / np.linalg.norm(matrices, axis=1, keepdims=True)

    @staticmethod
    def prepare_head_curves(path: str, settings: dict[str, Any], fps: float) -> dict[str, Any] | None:
        """The head rotation of the face source per frame of the scene:

            rotations  rotation vectors in the space of the rig (frames, 3), relative to the neutral head
            frames     number of scene frames
            start      time in seconds of the first frame in source
        """
        settings = dict(DEFAULT_FACE_SETTINGS, **settings)
        source = MocapAnimationService.load_head_source(path)
        if source is None:
            return None
        times, matrices = source
        valid = ~np.isnan(matrices).any(axis=(1, 2))
        if not len(times) or not valid.any():
            return None

        reference = np.eye(3)
        if settings["use_neutral"]:
            rows = np.flatnonzero(valid)
            row = rows[int(np.argmin(np.abs(times[rows] - (times[0] + float(settings["neutral_time"])))))]
            reference = matrices[row]
        # Rotation of the head since the neutral pose - first in camera space, then in rig space
        rotations = matrices @ reference.T
        if settings["mirror"]:
            rotations = _MIRROR_CAMERA @ rotations @ _MIRROR_CAMERA
        rotations = _CAMERA_TO_RIG @ rotations @ _CAMERA_TO_RIG.T
        vectors = MocapAnimationService.matrices_to_vectors(rotations)

        vectors = MocapAnimationService.fill_gaps(times, vectors, signed=True)
        vectors = MocapAnimationService.smooth(times, vectors, float(settings["smoothing"]))
        if len(times) == 1:
            frames = 1
        else:
            frames = int(math.floor((times[-1] - times[0]) * fps + 1e-6)) + 1
            frame_times = times[0] + np.arange(frames) / fps
            vectors = np.stack([np.interp(frame_times, times, vectors[:, column]) for column in range(3)], axis=1)
        return {"rotations": vectors, "frames": frames, "start": float(times[0])}

    # --------- Blender ----------

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
    def interpolate_to_children(basemesh: bpy.types.Object, done_prop: str = INTERPOLATED_CHILDREN_PROP) -> None:
        """Interpolate the face units (or visemes) to the fitted meshes which did not get them yet."""
        children = sorted(child.name for child in MocapAnimationService.fitted_children(basemesh))
        done = set(str(basemesh.get(done_prop, "")).split("|")) - {""}
        if set(children) <= done:
            return
        _LOG.info("Interpolating the face shape keys to", [name for name in children if name not in done])
        FaceService.interpolate_targets(basemesh)
        basemesh[done_prop] = "|".join(sorted(done | set(children)))

    @staticmethod
    def ensure_visemes(basemesh: bpy.types.Object) -> list[str]:
        """Find or load the visemes02 shape keys on the basemesh."""
        keys = basemesh.data.shape_keys
        present = [name for name in META_VISEMES if keys and name in keys.key_blocks]
        if not present:
            _LOG.info("Loading the visemes02 shape keys onto", basemesh.name)
            FaceService.load_targets(basemesh, load_microsoft_visemes=False, load_meta_visemes=True,
                                     load_arkit_faceunits=False)
            keys = basemesh.data.shape_keys
            present = [name for name in META_VISEMES if keys and name in keys.key_blocks]
            if not present:
                raise RuntimeError("The visemes02 asset pack is not installed")
        return present

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
    def _new_fcurve_collection(key: bpy.types.ID, action: bpy.types.Action) -> Any:
        """Assign the action to the shape keys (or the rig) and return the collection to add its fcurves to. Blender 4.4+ keeps the
        fcurves in a channelbag of an action slot, older versions directly in action.fcurves."""
        animation_data = key.animation_data or key.animation_data_create()
        if not hasattr(action, "slots"):
            animation_data.action = action
            return action.fcurves  # pyright: ignore[reportAttributeAccessIssue]
        slot = action.slots.new(id_type=key.id_type, name=key.name)
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
            MocapAnimationService._write_fcurve(fcurves, 'key_blocks["%s"].value' % name, 0, frames, values[:, column])
        return action

    @staticmethod
    def _write_fcurve(fcurves: Any, data_path: str, index: int, frames: np.ndarray, values: np.ndarray,
                      group: str | None = None) -> Any:
        """One fcurve with a linear keyframe on every frame."""
        fcurve = None
        if group:
            try:
                fcurve = fcurves.new(data_path, index=index, action_group=group)
            except TypeError:
                # Channelbag of Blender 4.4+ has no groups in new()
                pass
        if fcurve is None:
            fcurve = fcurves.new(data_path, index=index)
        points = fcurve.keyframe_points
        points.add(len(values))
        coordinates = np.empty(2 * len(values))
        coordinates[0::2] = frames
        coordinates[1::2] = values
        points.foreach_set("co", coordinates)
        for point in points:
            point.interpolation = 'LINEAR'
        fcurve.update()
        return fcurve

    # ------- Rig ---------

    @staticmethod
    def find_rig(basemesh: bpy.types.Object) -> bpy.types.Object | None:
        return ObjectService.find_object_of_type_amongst_nearest_relatives(basemesh, "Skeleton")

    @staticmethod
    def head_bones(rig: bpy.types.Object) -> list[str]:
        """Returns head bone and the neck bones from the chest up.
        Names are matched without prefix or case (mixamorig:Head), and number of neck bones doesnt matter (neck01 - neck03 on the default rig, one Neck on Mixamo)."""
        def plain(name):
            return name.split(":")[-1].lower()

        head = next((bone for bone in rig.data.bones if plain(bone.name) == "head"), None)
        if head is None:
            return []
        necks = []
        parent = head.parent
        while parent is not None and plain(parent.name).startswith("neck"):
            necks.insert(0, parent.name)
            parent = parent.parent
        if not necks:
            # Rigify: the neck control is not a parent of the head control
            necks = [bone.name for bone in rig.data.bones if plain(bone.name) == "neck"][:1]
        return necks + [head.name]

    @staticmethod
    def write_bone_rotation_action(rig: bpy.types.Object, action_name: str, frame_start: int,
                                   shares: list[tuple[str, float]], vectors: np.ndarray) -> bpy.types.Action:
        """Creates new action rotating the given bones. `vectors` are rotations in the rig space,
        every bone gets its share of them converted to its own rest orientation, so together they turn the head by the whole rotation."""
        action = bpy.data.actions.new(action_name)
        action.use_fake_user = True
        fcurves = MocapAnimationService._new_fcurve_collection(rig, action)
        frames = frame_start + np.arange(len(vectors), dtype=np.float64)
        for name, share in shares:
            pose_bone = rig.pose.bones[name]
            rest = np.array(pose_bone.bone.matrix_local.to_3x3().normalized())
            local = rest.T @ MocapAnimationService.vectors_to_matrices(vectors * share) @ rest
            mode = pose_bone.rotation_mode
            channels = []
            previous = None
            for matrix in local:
                quaternion = Matrix(matrix.tolist()).to_quaternion()
                if mode == 'QUATERNION':
                    # Keep the quaternions in the same hemisphere so the curves do not flip
                    if previous is not None and quaternion.dot(previous) < 0.0:
                        quaternion.negate()
                    channels.append(tuple(quaternion))
                elif mode == 'AXIS_ANGLE':
                    axis, angle = quaternion.to_axis_angle()
                    channels.append((angle, axis.x, axis.y, axis.z))
                else:
                    euler = quaternion.to_euler(mode, previous) if previous is not None else quaternion.to_euler(mode)
                    channels.append(tuple(euler))
                    quaternion = euler
                previous = quaternion
            data_path = 'pose.bones["%s"].%s' % (name, {"QUATERNION": "rotation_quaternion",
                                                       "AXIS_ANGLE": "rotation_axis_angle"}.get(mode, "rotation_euler"))
            channels = np.asarray(channels, dtype=np.float64)
            for index in range(channels.shape[1]):
                MocapAnimationService._write_fcurve(fcurves, data_path, index, frames, channels[:, index], name)
        return action

    @staticmethod
    def push_down_action(id_data: bpy.types.ID) -> str | None:
        """Action the user has on the rig (pose, mixamo clip) is moved to its own NLA track, so it is not lost when the head action is assigned.
        Returns the name of new track."""
        animation_data = id_data.animation_data
        if not animation_data or not animation_data.action:
            return None
        action = animation_data.action
        slot = getattr(animation_data, "action_slot", None)
        track = animation_data.nla_tracks.new()
        track.name = action.name
        strip = track.strips.new(action.name, int(action.frame_range[0]), action)
        if slot is not None and hasattr(strip, "action_slot") and strip.action_slot != slot:
            strip.action_slot = slot
        animation_data.action = None
        return track.name

    @staticmethod
    def raise_track(id_data: bpy.types.ID, track_name: str) -> None:
        """Move the NLA track to the top (new tracks are added on top)."""
        tracks = id_data.animation_data.nla_tracks
        track = next((track for track in tracks if track.name == track_name), None)
        if track is None or track == tracks[len(tracks) - 1]:
            return
        strips = [(strip.action, strip.frame_start, strip.blend_type, strip.extrapolation, strip.mute,
                   getattr(strip, "action_slot", None)) for strip in track.strips if strip.action]
        tracks.remove(track)
        moved = tracks.new()
        moved.name = track_name
        for action, frame_start, blend_type, extrapolation, mute, slot in strips:
            strip = moved.strips.new(action.name, int(frame_start), action)
            strip.blend_type = blend_type
            strip.extrapolation = extrapolation
            strip.mute = mute
            if slot is not None and hasattr(strip, "action_slot") and strip.action_slot != slot:
                strip.action_slot = slot

    @staticmethod
    def rig_layer_tracks(basemesh: bpy.types.Object) -> list[Any]:
        """The MPFB NLA tracks on the rig of the character."""
        rig = MocapAnimationService.find_rig(basemesh)
        if rig is None or not rig.animation_data:
            return []
        return [track for track in rig.animation_data.nla_tracks if track.name.startswith(TRACK_PREFIX)]

    @staticmethod
    def action_name_for(path: str, input_name: str | None, prefix: str = "MPFB_Face_") -> str:
        stem = os.path.splitext(input_name or os.path.basename(path))[0]
        stem = re.sub(r"_(landmarks|logits)$", "", stem)
        return prefix + stem

    # ------- Layers ---------

    @staticmethod
    def _action_fcurves(action: bpy.types.Action, slot: Any = None) -> list[Any]:
        """Return the fcurves of the action for the given slot.
        Blender 4.4+ keeps them in a channelbag of the slot, older versions directly in action.fcurves."""
        if not hasattr(action, "slots"):
            return list(action.fcurves)  # pyright: ignore[reportAttributeAccessIssue]
        if slot is None and len(action.slots):
            slot = action.slots[0]
        if slot is None:
            return []
        try:
            from bpy_extras import anim_utils  # pylint: disable=C0415
            channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)
            return list(channelbag.fcurves) if channelbag else []
        except (ImportError, AttributeError):
            for layer in action.layers:
                for strip in layer.strips:
                    channelbag = strip.channelbag(slot)
                    if channelbag:
                        return list(channelbag.fcurves)
            return []

    @staticmethod
    def place_on_track(key: bpy.types.ID, action: bpy.types.Action, track_name: str, frame_start: int, frame_end: int) -> Any:
        """Move action from the shape keys (or the rig) onto a strip of NLA track. Strips of that track in the same frames are replaced.
        Also re-enables the layers: the baked action is unassigned and the tracks with MPFB prefix are unmuted."""
        animation_data = key.animation_data or key.animation_data_create()
        slot = getattr(animation_data, "action_slot", None)
        animation_data.action = None
        track = next((track for track in animation_data.nla_tracks if track.name == track_name), None)
        if track is None:
            track = animation_data.nla_tracks.new()
            track.name = track_name
        for other in animation_data.nla_tracks:
            if other.name.startswith(TRACK_PREFIX):
                other.mute = False
        for strip in list(track.strips):
            if strip.frame_start <= frame_end and strip.frame_end >= frame_start:
                track.strips.remove(strip)
        strip = track.strips.new(action.name, int(frame_start), action)
        # Layer only acts in its own frames (so the mouth does not stay open after the voice line)
        strip.extrapolation = 'NOTHING'
        if slot is not None and hasattr(strip, "action_slot") and strip.action_slot != slot:
            strip.action_slot = slot
        return strip

    @staticmethod
    def layer_tracks(basemesh: bpy.types.Object) -> list[Any]:
        """The MPFB NLA tracks on the shape keys of the basemesh."""
        keys = basemesh.data.shape_keys
        if not keys or not keys.animation_data:
            return []
        return [track for track in keys.animation_data.nla_tracks if track.name.startswith(TRACK_PREFIX)]

    @staticmethod
    def bake_layers(basemesh: bpy.types.Object) -> dict[str, Any]:
        """Evaluate the NLA tracks of the shape keys into one action, assign it and mute the original tracks."""
        key = basemesh.data.shape_keys
        tracks = [track for track in (key.animation_data.nla_tracks if key and key.animation_data else []) if not track.mute]
        strips = [strip for track in tracks for strip in track.strips if not strip.mute and strip.action]
        if not strips:
            raise RuntimeError("Nothing to bake, create a face or lipsync animation first")
        frame_start = int(math.floor(min(strip.frame_start for strip in strips)))
        frame_end = int(math.ceil(max(strip.frame_end for strip in strips)))
        frames = np.arange(frame_start, frame_end + 1, dtype=np.float64)

        channels: dict[str, np.ndarray] = {}
        # Tracks are evaluated from the bottom up - strip replaces/adds to what is below it
        for track in tracks:
            for strip in track.strips:
                if strip.mute or not strip.action:
                    continue
                inside = (frames >= strip.frame_start) & (frames <= strip.frame_end)
                local = strip.action_frame_start + (frames[inside] - strip.frame_start) / max(strip.scale, 1e-6)
                for fcurve in MocapAnimationService._action_fcurves(strip.action, getattr(strip, "action_slot", None)):
                    below = channels.setdefault(fcurve.data_path, np.zeros(len(frames)))
                    values = np.array([fcurve.evaluate(frame) for frame in local])
                    if strip.blend_type == 'ADD':
                        below[inside] = below[inside] + values * strip.influence
                    else:
                        below[inside] = below[inside] * (1.0 - strip.influence) + values * strip.influence

        names = []
        columns = []
        for data_path, values in channels.items():
            match = re.match(r'key_blocks\["(.+)"\]\.value$', data_path)
            if match:
                names.append(match.group(1))
                columns.append(values)
        if not names:
            raise RuntimeError("The layers have no shape key curves to bake")
        action = MocapAnimationService.write_shape_key_action(key, BAKED_ACTION, frame_start, names,
                                                              np.stack(columns, axis=1))
        for track in tracks:
            track.mute = True
        result = {"action": action.name, "frame_start": frame_start, "frame_end": frame_end, "channels": len(names)}
        _LOG.info("Layers baked", result)
        return result

    @staticmethod
    def add_sound_strip(scene: bpy.types.Scene, path: str, name: str, frame_start: int, source_start: float,
                        frames: int, fps: float) -> Any:
        """Put the recording into sequencer aligned with lipsync. Strip of same name is replaced."""
        editor = scene.sequence_editor or scene.sequence_editor_create()
        strips = editor.strips if hasattr(editor, "strips") else editor.sequences  # pyright: ignore[reportAttributeAccessIssue]
        for strip in list(strips):
            if strip.name == name:
                strips.remove(strip)
        channel = max((strip.channel for strip in strips), default=0) + 1
        offset = int(round(source_start * fps))
        strip = strips.new_sound(name, path, channel, int(frame_start) - offset)
        try:
            strip.frame_offset_start = offset
            content_end = int(frame_start) - offset + strip.frame_duration
            strip.frame_offset_end = max(0, content_end - (int(frame_start) + frames))
        except (AttributeError, TypeError) as err:
            _LOG.warn("Could not trim the sound strip", err)
        return strip

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
        values = curves["values"]
        if settings["skip_speech"]:
            # The lipsync layer moves the mouth
            keep = [index for index, name in enumerate(names) if name not in SPEECH_FACEUNITS]
            names = [names[index] for index in keep]
            values = values[:, keep]
        animated = [name for name in names if name in present]
        action = MocapAnimationService.write_shape_key_action(
            key, MocapAnimationService.action_name_for(path, input_name), frame_start, names, values)
        frame_end = frame_start + curves["frames"] - 1
        MocapAnimationService.place_on_track(key, action, FACE_TRACK, frame_start, frame_end)

        driven = MocapAnimationService.drive_children(basemesh, animated) if settings["include_children"] else 0

        head = None
        head_skipped = None
        if settings["head_rotation"]:
            head = MocapAnimationService.create_head_rotation(basemesh, path, settings, scene, input_name, frame_start)
            if head.get("skipped"):
                head_skipped = head["skipped"]
                head = None
            else:
                frame_end = max(frame_end, head["frame_end"])
        if settings["extend_scene"] and scene.frame_end < frame_end:
            scene.frame_end = frame_end

        result = {"action": action.name, "track": FACE_TRACK, "frame_start": frame_start, "frame_end": frame_end,
                  "animated": len(animated), "missing": [name for name in ARKIT_FACEUNITS if name not in animated],
                  "children": driven, "skipped_speech": bool(settings["skip_speech"]), "head": head,
                  "head_skipped": head_skipped}
        _LOG.info("Face animation created", result)
        return result

    @staticmethod
    def create_head_rotation(basemesh: bpy.types.Object, path: str, settings: dict[str, Any], scene: bpy.types.Scene,
                             input_name: str | None = None, frame_start: int | None = None) -> dict[str, Any]:
        """Turn the neck and head bones of the rig by the head rotation of the face source."""
        settings = dict(DEFAULT_FACE_SETTINGS, **settings)
        rig = MocapAnimationService.find_rig(basemesh)
        if rig is None:
            return {"skipped": "the character has no rig"}
        bones = MocapAnimationService.head_bones(rig)
        if not bones:
            return {"skipped": "no head bone in the rig " + rig.name}
        fps = scene.render.fps / scene.render.fps_base
        curves = MocapAnimationService.prepare_head_curves(path, settings, fps)
        if curves is None:
            return {"skipped": "the face source has no head rotation"}

        # The neck share is split evenly between neck bones and the head gets the rest
        necks = bones[:-1]
        neck_share = min(max(float(settings["neck_share"]), 0.0), 1.0) if necks else 0.0
        shares = [(name, neck_share / len(necks)) for name in necks] + [(bones[-1], 1.0 - neck_share)]

        if frame_start is None:
            frame_start = scene.frame_current
        pushed = MocapAnimationService.push_down_action(rig)
        if pushed:
            MocapAnimationService.raise_track(rig, HEAD_TRACK)
        action = MocapAnimationService.write_bone_rotation_action(
            rig, MocapAnimationService.action_name_for(path, input_name, "MPFB_Head_"), frame_start, shares,
            curves["rotations"])
        frame_end = frame_start + curves["frames"] - 1
        strip = MocapAnimationService.place_on_track(rig, action, HEAD_TRACK, frame_start, frame_end)
        # Adds rotation to anim/pose under it
        strip.blend_type = 'COMBINE'

        result = {"action": action.name, "track": HEAD_TRACK, "rig": rig.name, "bones": bones,
                  "frame_start": frame_start, "frame_end": frame_end, "pushed_down": pushed}
        _LOG.info("Head rotation created", result)
        return result

    @staticmethod
    def create_lipsync(basemesh: bpy.types.Object, path: str, settings: dict[str, Any], scene: bpy.types.Scene,
                       source: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create lipsync of the character from an audio source, starting at current frame of the scene."""
        settings = dict(DEFAULT_LIPSYNC_SETTINGS, **settings)
        source = source or {}
        fps = scene.render.fps / scene.render.fps_base
        curves = MocapLipsyncService.prepare_lipsync_curves(path, settings, fps)

        present = MocapAnimationService.ensure_visemes(basemesh)
        if settings["include_children"]:
            MocapAnimationService.interpolate_to_children(basemesh, INTERPOLATED_VISEMES_PROP)

        key = basemesh.data.shape_keys
        frame_start = scene.frame_current
        frame_end = frame_start + curves["frames"] - 1
        name = MocapAnimationService.action_name_for(path, source.get("input_name"), "MPFB_Lipsync_")
        action = MocapAnimationService.write_shape_key_action(key, name, frame_start, curves["names"], curves["values"])
        MocapAnimationService.place_on_track(key, action, LIPSYNC_TRACK, frame_start, frame_end)
        animated = [viseme for viseme in curves["names"] if viseme in present]
        driven = MocapAnimationService.drive_children(basemesh, animated) if settings["include_children"] else 0

        sound = None
        recording = source.get("input_path")
        if settings["add_sound"]:
            if recording and os.path.isfile(recording):
                sound = MocapAnimationService.add_sound_strip(scene, recording, action.name, frame_start, curves["start"],
                                                              curves["frames"], fps).name
            else:
                _LOG.warn("The recording of the lipsync was not found, no sound added", recording)
        if settings["extend_scene"] and scene.frame_end < frame_end:
            scene.frame_end = frame_end

        result = {"action": action.name, "track": LIPSYNC_TRACK, "frame_start": frame_start, "frame_end": frame_end,
                  "animated": len(animated), "phonemes": curves["phonemes"], "unknown": curves["unknown"],
                  "children": driven, "sound": sound}
        _LOG.info("Lipsync created", result)
        return result
