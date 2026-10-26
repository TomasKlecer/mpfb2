# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         24.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  reading and checking the saved model outputs before they are used as animation sources
# ------------------------------------------------------------------------------
"""Create Animation panel takes files the worker saved:

    <run folder>/video_<input>_landmarks.npz   vision output: pose, hands, face, blendshapes per frame
    <run folder>/audio_<input>_logits.npz      audio output: phoneme log probabilities per 20 ms
    <any folder>/<take>.csv                    Live Link Face recording (iPhone TrueDepth): ARKit values per frame

This service looks into the file and checks whether it fits the role it was picked for.
"""

import csv
import datetime
import json
import os
import re
from typing import Any

import numpy as np

from .logservice import LogService
from .locationservice import LocationService
from .faceservice import ARKIT_FACEUNITS

_LOG = LogService.get_logger("services.mocapsourceservice")

ROLE_FACE = "FACE"
ROLE_BODY = "BODY"
ROLE_AUDIO = "AUDIO"

KIND_VISION = "vision"
KIND_AUDIO = "audio"
KIND_LIVELINK = "livelink"
KIND_UNKNOWN = "unknown"

ERROR = "ERROR"
WARNING = "WARNING"

# File browser filters for picking a source of each role
ROLE_FILTERS = {ROLE_FACE: "*_landmarks.npz;*.csv", ROLE_BODY: "*_landmarks.npz", ROLE_AUDIO: "*_logits.npz"}

# Live Link Face timecode HH:MM:SS:FF.sss (frame and its fraction)
_TIMECODE = re.compile(r"^(\d+):(\d+):(\d+)[:;](\d+)(?:\.(\d+))?$")
_TIMECODE_RATES = (24, 25, 30, 48, 50, 60)
_DEFAULT_LIVELINK_FPS = 60

# Below this share of frames with a detection, a source is flagged as weak
_LOW_DETECTION = 0.5
_FPS_TOLERANCE = 0.5
_LENGTH_TOLERANCE = 0.5

_CACHE: dict[str, tuple[float, int, dict]] = {}


class MocapSourceService():
    """Inspecting saved model outputs and checking them for their role."""

    def __init__(self) -> None:
        raise RuntimeError("You should not instance MocapSourceService. Use its static methods instead.")

    @staticmethod
    def get_runs_dir() -> str:
        return LocationService.get_user_cache("mocap_runs")

    @staticmethod
    def get_latest_run_dir() -> str | None:
        runs_dir = MocapSourceService.get_runs_dir()
        if not os.path.isdir(runs_dir):
            return None
        runs = sorted(name for name in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, name)))
        return os.path.join(runs_dir, runs[-1]) if runs else None

    @staticmethod
    def sources_from_run(run_dir: str) -> dict[str, Any]:
        """The outputs of a run read from its job.json and the files in its folder:

            face    the landmarks of the video whith face model enabled
            bodies  the landmarks of every video whith pose model enabled
            audio   the logits of the audio task

        Tasks which did not produce their file are left out."""
        sources: dict[str, Any] = {"face": None, "bodies": [], "audio": None}
        try:
            with open(os.path.join(run_dir, "job.json"), "r", encoding="utf-8") as job_file:
                job = json.load(job_file)
        except (OSError, ValueError):
            return sources
        names = sorted(os.listdir(run_dir))

        def output(task_id, suffix):
            for name in names:
                if name.startswith(task_id + "_") and name.endswith(suffix):
                    return os.path.join(run_dir, name)
            return None

        for task in job.get("tasks", []):
            if task.get("type") == "vision":
                path = output(task["id"], "_landmarks.npz")
                if path is None:
                    continue
                if task.get("face", {}).get("enabled") and sources["face"] is None:
                    sources["face"] = path
                if task.get("pose", {}).get("enabled"):
                    sources["bodies"].append(path)
            elif task.get("type") == "audio" and sources["audio"] is None:
                sources["audio"] = output(task["id"], "_logits.npz")
        return sources

    # ------------ Live Link Face --------------

    @staticmethod
    def read_livelink_csv(path: str) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Times in seconds from the first frame, ARKit names used by MPFB2 and their values per frame.
        Columns, which are not ARKit face units (BlendShapeCount, head and eye rotation), are skipped."""
        with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.reader(csv_file))
        header = [column.strip() for column in rows[0]] if rows else []
        known = {name.lower(): name for name in ARKIT_FACEUNITS}
        columns = [(index, known[column.lower()]) for index, column in enumerate(header) if column.lower() in known]
        if not header or header[0].lower() != "timecode" or not columns:
            raise ValueError("Not a Live Link Face csv")

        timecodes = []
        values = []
        for row in rows[1:]:
            if len(row) < len(header):
                continue
            timecodes.append(_TIMECODE.match(row[0].strip()))
            values.append([MocapSourceService._number(row[index]) for index, _name in columns])
        if not values:
            raise ValueError("The csv has no frames")

        if all(timecodes):
            parts = [match.groups() for match in timecodes]
            highest = max(int(part[3]) for part in parts)
            rate = next((rate for rate in _TIMECODE_RATES if rate > highest), _DEFAULT_LIVELINK_FPS)
            times = np.array([int(h) * 3600 + int(m) * 60 + int(sec) + (int(frame) + (int(fraction) / 10 ** len(fraction) if fraction else 0.0)) / rate
                              for h, m, sec, frame, fraction in parts])
        else:
            times = np.arange(len(values)) / _DEFAULT_LIVELINK_FPS
        times = times - times[0]

        # Repeated or out of order timecodes are dropped
        keep = [0]
        for index in range(1, len(times)):
            if times[index] > times[keep[-1]]:
                keep.append(index)
        return times[keep], [name for _index, name in columns], np.asarray(values, dtype=np.float64)[keep]

    @staticmethod
    def _number(text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return float("nan")

    # ----------- Inspecting -------------

    @staticmethod
    def inspect(path: str) -> dict[str, Any]:
        """The file and how good it is. Cached until the file changes, the panel asks on every redraw."""
        try:
            stat = os.stat(path)
        except OSError:
            return {"path": path, "exists": False, "kind": KIND_UNKNOWN, "error": "The file does not exist"}
        cached = _CACHE.get(path)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]
        info = MocapSourceService._read(path)
        _CACHE[path] = (stat.st_mtime, stat.st_size, info)
        return info

    @staticmethod
    def _read(path: str) -> dict[str, Any]:
        info: dict[str, Any] = {"path": path, "exists": True, "kind": KIND_UNKNOWN, "error": None,
                                "frames": 0, "duration": 0.0, "fps": 0.0, "detection": {}, "speech": None,
                                "input_name": None, "run_label": MocapSourceService._run_label(path)}
        if path.lower().endswith(".csv"):
            MocapSourceService._read_livelink(info, path)
            return info
        task = MocapSourceService._job_task(path)
        if task is not None:
            # job.json may come from another system, so split on both kinds of separators
            info["input_name"] = re.split(r"[\\/]", task.get("input", ""))[-1] or None
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
                if {"times_ms", "pose", "blendshapes"} <= keys:
                    MocapSourceService._read_vision(info, data, task)
                elif {"logprobs", "frame_seconds"} <= keys:
                    MocapSourceService._read_audio(info, data)
                else:
                    info["error"] = "Not an output of the motion capture models"
        except Exception as err:  # pylint: disable=W0718
            _LOG.warn("Could not read a motion capture source", (path, err))
            info["error"] = "Could not read the file: %s" % err
        return info

    @staticmethod
    def _read_vision(info: dict, data: Any, task: dict | None) -> None:
        info["kind"] = KIND_VISION
        times = np.asarray(data["times_ms"], dtype=np.float64) / 1000.0
        frames = len(times)
        info["frames"] = frames
        if frames > 1 and times[-1] > times[0]:
            info["fps"] = (frames - 1) / (times[-1] - times[0])
            info["duration"] = frames / info["fps"]
        info["start"] = float(times[0]) if frames else 0.0

        def enabled(name):
            # Without job.json (file copied from elsewhere) assume every model ran
            return task is None or bool(task.get(name, {}).get("enabled", True))

        detection = {}
        if enabled("pose"):
            detection["pose"] = float(np.mean(~np.isnan(data["pose"][:, 0, 0]))) if frames else 0.0
        if enabled("hands") and "hands" in data.files:
            detection["hands"] = float(np.mean(~np.isnan(data["hands"][:, :, 0, 0]).any(axis=1))) if frames else 0.0
        if enabled("face"):
            detection["face"] = float(np.mean(~np.isnan(data["blendshapes"][:, 0]))) if frames else 0.0
        info["detection"] = detection

    @staticmethod
    def _read_livelink(info: dict, path: str) -> None:
        info["input_name"] = os.path.basename(path)
        try:
            times, _names, values = MocapSourceService.read_livelink_csv(path)
        except Exception as err:  # pylint: disable=W0718
            info["error"] = "Could not read the Live Link Face csv: %s" % err
            return
        info["kind"] = KIND_LIVELINK
        info["frames"] = len(times)
        if len(times) > 1:
            info["fps"] = (len(times) - 1) / times[-1]
            info["duration"] = len(times) / info["fps"]
        # LLF writes zeros while it does not track the face
        info["detection"] = {"face": float(np.mean(np.nan_to_num(values).any(axis=1)))}
        info["start"] = 0.0

    @staticmethod
    def _read_audio(info: dict, data: Any) -> None:
        info["kind"] = KIND_AUDIO
        logprobs = data["logprobs"]
        frame_seconds = float(data["frame_seconds"])
        blank = int(data["blank"]) if "blank" in data.files else 0
        info["frames"] = len(logprobs)
        info["duration"] = len(logprobs) * frame_seconds
        info["fps"] = 1.0 / frame_seconds if frame_seconds else 0.0
        info["speech"] = float(np.mean(np.argmax(logprobs, axis=1) != blank)) if len(logprobs) else 0.0
        info["start"] = float(data["frame_times"][0] - frame_seconds / 2) if "frame_times" in data.files and len(logprobs) else 0.0

    @staticmethod
    def _job_task(path: str) -> dict | None:
        """The task of job.json in the same run folder which produced the file (its name starts with the task id)."""
        job_path = os.path.join(os.path.dirname(path), "job.json")
        if not os.path.isfile(job_path):
            return None
        try:
            with open(job_path, "r", encoding="utf-8") as job_file:
                job = json.load(job_file)
        except (OSError, ValueError):
            return None
        name = os.path.basename(path)
        for task in job.get("tasks", []):
            if name.startswith(task.get("id", "") + "_"):
                return task
        return None

    @staticmethod
    def _run_label(path: str) -> str | None:
        """Returns '23.10. 10:14' from a run folder named 20261023_101418 (+ eventual _2 suffix)."""
        folder = os.path.basename(os.path.dirname(path))
        try:
            stamp = datetime.datetime.strptime(folder[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            return None
        return stamp.strftime("%d.%m. %H:%M")

    # ------------ Texts for the UI ---------------

    @staticmethod
    def describe(info: dict) -> str:
        """Short name of the source: input file + time of the run, or the file name."""
        name = info.get("input_name") or os.path.basename(info.get("path", ""))
        if info.get("run_label"):
            return "%s · %s" % (name, info["run_label"])
        return name

    @staticmethod
    def details(info: dict) -> str:
        if not info.get("exists") or info.get("error"):
            return ""
        parts = []
        if info["kind"] == KIND_VISION:
            if info["frames"] == 1:
                parts.append("1 frame (image)")
            else:
                parts.append("%d frames · %.1f s · %.0f fps" % (info["frames"], info["duration"], info["fps"]))
            parts += ["%s %d %%" % (name, round(rate * 100)) for name, rate in info["detection"].items()]
        elif info["kind"] == KIND_LIVELINK:
            parts.append("%d frames · %.1f s · %.0f fps · Live Link Face" % (info["frames"], info["duration"], info["fps"]))
        elif info["kind"] == KIND_AUDIO:
            parts.append("%.1f s" % info["duration"])
            parts.append("speech %d %%" % round((info["speech"] or 0.0) * 100))
        return " · ".join(parts)

    # -------- Checks ---------

    @staticmethod
    def check_role(info: dict, role: str) -> list[tuple[str, str]]:
        """Problems of source in the given role as (ERROR/WARNING, message)."""
        if not info.get("exists"):
            return [(ERROR, "The file does not exist")]
        if info.get("error"):
            return [(ERROR, info["error"])]
        if role == ROLE_AUDIO:
            if info["kind"] != KIND_AUDIO:
                return [(ERROR, "Not an audio output, pick a *_logits.npz")]
            if (info["speech"] or 0.0) < 0.02:
                return [(WARNING, "Almost no speech was recognised")]
            return []

        if info["kind"] == KIND_LIVELINK:
            if role != ROLE_FACE:
                return [(ERROR, "Live Link Face csv has only the face")]
            if info["detection"]["face"] == 0.0:
                return [(ERROR, "The csv has no tracked face, all frames are empty")]
            return []
        if info["kind"] != KIND_VISION:
            if role == ROLE_FACE:
                return [(ERROR, "Not a face output, pick a *_landmarks.npz or a Live Link Face csv")]
            return [(ERROR, "Not a video / image output, pick a *_landmarks.npz")]
        model = "face" if role == ROLE_FACE else "pose"
        rate = info["detection"].get(model)
        if rate is None:
            return [(ERROR, "The %s model was not enabled in this run" % model)]
        if rate == 0.0:
            return [(ERROR, "No %s was detected" % model)]
        if rate < _LOW_DETECTION:
            return [(WARNING, "%s detected only in %d %% of the frames" % (model.capitalize(), round(rate * 100)))]
        return []

    @staticmethod
    def check_together(face: dict | None, bodies: list[dict], audio: dict | None) -> list[tuple[str, str]]:
        """Problems between the sources (different frame rates or lengths)."""
        problems = []
        videos = [info for info in ([face] if face else []) + bodies
                  if info.get("exists") and not info.get("error") and info.get("kind") in (KIND_VISION, KIND_LIVELINK)
                  and info["frames"] > 1]
        rates = sorted({round(info["fps"], 1) for info in videos})
        if len(rates) > 1 and rates[-1] - rates[0] > _FPS_TOLERANCE:
            problems.append((WARNING, "Videos have different fps (%s), they will be resampled"
                             % ", ".join("%g" % rate for rate in rates)))
        timed = videos + ([audio] if audio and audio.get("exists") and audio.get("kind") == KIND_AUDIO else [])
        lengths = [info["duration"] for info in timed]
        if len(lengths) > 1 and max(lengths) - min(lengths) > _LENGTH_TOLERANCE:
            problems.append((WARNING, "Sources have different lengths (%s s), they will need to be synchronised"
                             % ", ".join("%.1f" % length for length in lengths)))
        return problems
