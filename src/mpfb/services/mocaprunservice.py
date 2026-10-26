# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  services for running the motion capture worker from blender
# ------------------------------------------------------------------------------
"""Runs the motion capture worker (mocap_worker/run_worker.py) in the python of the motion capture environment.

The worker gets a job.json describing the inputs, writes its results into a run folder in MPFB2 user cache
(<user home>/cache/mocap_runs/<date_time>/) and reports progress as json lines on stdout.
A background thread reads those lines into a RunJob polled by UI.
"""

import datetime
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any

import bpy

from .logservice import LogService
from .locationservice import LocationService
from .mocapenvservice import MocapEnvService

_LOG = LogService.get_logger("services.mocaprunservice")

PROTOCOL_VERSION = 1
MESSAGE_PREFIX = "@@MPFB "
JOB_FILE = "job.json"
LOG_FILE = "worker.log"
TEXT_PREFIX = "MPFB mocap "

_IS_WINDOWS = sys.platform.startswith("win")
_CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if _IS_WINDOWS else 0

# Model registry ids (data/ai_models/models.json) of models the tasks use
POSE_MODEL_IDS = {"LITE": "mediapipe_pose_lite", "FULL": "mediapipe_pose_full", "HEAVY": "mediapipe_pose_heavy"}
HAND_MODEL_ID = "mediapipe_hand"
FACE_MODEL_ID = "mediapipe_face"
PHONEME_MODEL_ID = "wav2vec2_xlsr53_espeak"


class TaskState():
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"

    def __init__(self, task_id: str, label: str) -> None:
        self.task_id = task_id
        self.label = label
        self.status = TaskState.PENDING
        self.stage = ""
        self.current = 0
        self.total = 0
        self.message = ""


class RunJob():
    """State of one worker run. Written by the reader thread, read by the UI."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def __init__(self, run_dir: str = "") -> None:
        self.run_dir = run_dir
        self.status = RunJob.IDLE
        self.tasks: dict[str, TaskState] = {}
        self.outputs: list[dict[str, str]] = []
        self.error = ""
        self.last_line = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self.cancel_requested = False
        self.results_loaded = False
        self.loaded_texts: list[str] = []
        self.loaded_images: list[str] = []
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self.status == RunJob.RUNNING

    @property
    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at

    @property
    def log_path(self) -> str:
        return os.path.join(self.run_dir, LOG_FILE)

    @property
    def failed_tasks(self) -> list[TaskState]:
        return [task for task in self.tasks.values() if task.status == TaskState.FAILED]


_JOB = RunJob()


class MocapRunService():
    """Building worker jobs, running the worker and loading its results."""

    def __init__(self) -> None:
        raise RuntimeError("You should not instance MocapRunService. Use its static methods instead.")

    # -------------- Locations ---------------

    @staticmethod
    def get_worker_script() -> str:
        return LocationService.get_mpfb_root(os.path.join("mocap_worker", "run_worker.py"))

    @staticmethod
    def new_run_dir() -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LocationService.get_user_cache(os.path.join("mocap_runs", stamp))
        suffix = 1
        unique = path
        while os.path.exists(unique):
            suffix += 1
            unique = "%s_%d" % (path, suffix)
        return unique

    @staticmethod
    def model_path(model_id: str, registry: dict | None = None) -> str:
        """Path of the model relative to data/ai_models: file for single file models, its folder otherwise."""
        if registry is None:
            registry = MocapEnvService.load_models_registry()
        for model in registry.get("models", []):
            if model["id"] == model_id:
                files = [file_info["path"] for file_info in model.get("files", [])]
                if len(files) == 1:
                    return files[0]
                return os.path.dirname(files[0])
        raise KeyError("Model %s is not in the model registry" % model_id)

    # ------------ Job -----------------

    @staticmethod
    def build_job(settings: dict[str, Any], output_dir: str, models_dir: str, registry: dict | None = None) -> dict:
        """The job.json for the worker, settings are the panel values.

        settings["videos"] is a list of videos, each a dict with path, pose, hands, face, pose_model, start, end, min_confidence and annotate.
        Every video becomes one task."""
        def model(enabled, model_id):
            return {"enabled": bool(enabled), "model": MocapRunService.model_path(model_id, registry) if enabled else ""}

        tasks = []
        videos = [video for video in settings.get("videos", []) if (video.get("path") or "").strip()]
        for number, video in enumerate(videos, start=1):
            path = video["path"].strip()
            name = os.path.basename(path)
            tasks.append({
                "id": "video%d" % number, "type": "vision", "input": path,
                "label": ("Video %d: %s" % (number, name)) if len(videos) > 1 else ("Video: " + name),
                "start": float(video.get("start", 0.0)), "end": float(video.get("end", 0.0)),
                "min_confidence": float(video.get("min_confidence", 0.5)),
                "annotate": bool(video.get("annotate", True)),
                "pose": model(video.get("pose", True), POSE_MODEL_IDS[video.get("pose_model", "FULL")]),
                "hands": model(video.get("hands", True), HAND_MODEL_ID),
                "face": model(video.get("face", False), FACE_MODEL_ID),
                })
        audio = (settings.get("audio_path") or "").strip()
        if audio:
            tasks.append({
                "id": "audio", "type": "audio", "input": audio, "label": "Audio: " + os.path.basename(audio),
                "start": float(settings.get("audio_start", 0.0)), "end": float(settings.get("audio_end", 0.0)),
                "device": str(settings.get("audio_device", "AUTO")).lower(),
                "model": MocapRunService.model_path(PHONEME_MODEL_ID, registry),
                })
        return {"protocol": PROTOCOL_VERSION, "output_dir": output_dir, "models_dir": models_dir, "tasks": tasks}

    @staticmethod
    def validate_video(video: dict[str, Any]) -> list[str]:
        """Problems of video input as a messages for the user."""
        problems = []
        path = (video.get("path") or "").strip()
        if not path:
            return ["Select a video or an image"]
        name = os.path.basename(path)
        if not os.path.isfile(path):
            problems.append("The video or image does not exist: " + path)
        if not (video.get("pose", True) or video.get("hands", True) or video.get("face", False)):
            problems.append("Enable at least one of pose/hands/face for " + name)
        if 0 < float(video.get("end", 0.0)) <= float(video.get("start", 0.0)):
            problems.append("The end of %s must be after its start (or 0 for the end of the video)" % name)
        return problems

    @staticmethod
    def validate_settings(settings: dict[str, Any]) -> list[str]:
        """Reports problems which make run pointless as messages for the user. Empty list when the run can start."""
        problems = []
        videos = [video for video in settings.get("videos", []) if (video.get("path") or "").strip()]
        audio = (settings.get("audio_path") or "").strip()
        if not videos and not audio:
            problems.append("Select a video, image or an audio recording")
        for video in videos:
            problems += MocapRunService.validate_video(video)
        if sum(1 for video in videos if video.get("face")) > 1:
            problems.append("Only one video can be used for the face")
        if audio:
            if not os.path.isfile(audio):
                problems.append("The audio recording does not exist: " + audio)
            if 0 < float(settings.get("audio_end", 0.0)) <= float(settings.get("audio_start", 0.0)):
                problems.append("The audio end must be after its start (or 0 for the end of the recording)")
        return problems

    # ---------- Running -------------

    @staticmethod
    def get_job() -> RunJob:
        return _JOB

    @staticmethod
    def build_worker_environment() -> dict[str, str]:
        env = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("GLOG_minloglevel", "2")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        # matplotlib must not try to open a window
        env["MPLBACKEND"] = "Agg"
        return env

    @staticmethod
    def start_run(job_spec: dict, python: str | None = None, worker_script: str | None = None) -> RunJob:
        """Write job.json into job_spec["output_dir"] and start the worker on a background thread."""
        global _JOB
        if _JOB.is_running:
            raise RuntimeError("The motion capture worker is already running")
        if python is None:
            python = MocapEnvService.get_venv_python(MocapEnvService.get_env_dir())
        if worker_script is None:
            worker_script = MocapRunService.get_worker_script()

        run_dir = job_spec["output_dir"]
        os.makedirs(run_dir, exist_ok=True)
        job_path = os.path.join(run_dir, JOB_FILE)
        with open(job_path, "w", encoding="utf-8") as job_file:
            json.dump(job_spec, job_file, indent=2, ensure_ascii=False)

        job = RunJob(run_dir)
        for task in job_spec["tasks"]:
            job.tasks[task["id"]] = TaskState(task["id"], task.get("label") or ("Video / image" if task["type"] == "vision" else "Audio"))
        job.status = RunJob.RUNNING
        job.started_at = time.time()
        _JOB = job

        command = [python, "-u", worker_script, job_path]
        job.thread = threading.Thread(target=MocapRunService._run, args=(job, command, MocapRunService.build_worker_environment()),
                                      name="mpfb-mocap-worker", daemon=True)
        job.thread.start()
        return job

    @staticmethod
    def cancel_run() -> None:
        job = _JOB
        if not job.is_running:
            return
        job.cancel_requested = True
        process = job.process
        if process is not None and process.poll() is None:
            _LOG.info("Terminating the motion capture worker")
            MocapEnvService._kill_process_tree(process)  # pylint: disable=W0212

    @staticmethod
    def handle_line(job: RunJob, line: str) -> None:
        """Apply one line of worker output to the job. Lines without message prefix are only logged."""
        stripped = line.strip()
        if not stripped.startswith(MESSAGE_PREFIX):
            if stripped:
                job.last_line = stripped
            return
        try:
            message = json.loads(stripped[len(MESSAGE_PREFIX):])
        except ValueError:
            _LOG.warn("Unreadable worker message", stripped)
            return
        event = message.get("event")
        task = job.tasks.get(message.get("task", ""))
        if event == "task_start" and task is not None:
            task.status = TaskState.RUNNING
            task.label = message.get("label", task.label)
        elif event == "progress" and task is not None:
            task.stage = message.get("stage", "")
            task.current = int(message.get("current", 0))
            task.total = int(message.get("total", 0))
        elif event == "output":
            job.outputs.append({"task": message.get("task", ""), "kind": message.get("kind", ""),
                                "path": message.get("path", ""), "label": message.get("label", "")})
        elif event == "task_done" and task is not None:
            task.status = TaskState.DONE
            task.stage = "done"
        elif event == "task_failed" and task is not None:
            task.status = TaskState.FAILED
            task.message = message.get("message", "")
        elif event == "log":
            job.last_line = message.get("message", "")

    @staticmethod
    def _run(job: RunJob, command: list[str], env: dict[str, str]) -> None:
        """Body of the reader thread. Must not touch bpy."""
        try:
            with open(job.log_path, "w", encoding="utf-8") as log:
                log.write("MPFB motion capture worker, %s\n$ %s\n\n" % (datetime.datetime.now().isoformat(), " ".join(command)))
                log.flush()
                if job.cancel_requested:
                    raise _RunCancelled()
                try:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                               env=env, cwd=job.run_dir, creationflags=_CREATION_FLAGS,
                                               start_new_session=not _IS_WINDOWS,
                                               text=True, encoding="utf-8", errors="replace", bufsize=1)
                except OSError as err:
                    raise RuntimeError("Could not start the worker: %s" % err) from err
                job.process = process
                if job.cancel_requested:
                    MocapEnvService._kill_process_tree(process)  # pylint: disable=W0212

                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    MocapRunService.handle_line(job, line)
                return_code = process.wait()
                job.process = None
                log.write("\nworker exited with code %d\n" % return_code)

            if job.cancel_requested:
                raise _RunCancelled()
            # Exit code 1 = tasks failed - their messages are in task states
            if return_code not in (0, 1):
                raise RuntimeError("The worker exited with code %d: %s" % (return_code, job.last_line))
            for task in job.tasks.values():
                if task.status in (TaskState.PENDING, TaskState.RUNNING):
                    task.status = TaskState.FAILED
                    task.message = task.message or "The worker ended before this task finished"
            job.status = RunJob.FAILED if job.failed_tasks else RunJob.SUCCEEDED
            if job.failed_tasks:
                job.error = "; ".join("%s: %s" % (task.label, task.message) for task in job.failed_tasks)
            _LOG.info("Motion capture worker finished", (job.status, job.run_dir))
        except _RunCancelled:
            job.status = RunJob.CANCELLED
            job.error = "Cancelled by user"
        except Exception as err:  # pylint: disable=W0718
            job.status = RunJob.FAILED
            job.error = str(err)
            _LOG.error("Motion capture worker failed", err)
        finally:
            job.process = None
            job.finished_at = time.time()

    # ------------ Results --------------

    @staticmethod
    def load_results(job: RunJob) -> None:
        """Load text summaries into text blocks and the pictures into images."""
        if job.results_loaded:
            return
        job.results_loaded = True
        for output in job.outputs:
            path = output["path"]
            if not os.path.isfile(path):
                continue
            try:
                if output["kind"] == "text":
                    name = TEXT_PREFIX + os.path.basename(path)
                    text = bpy.data.texts.get(name) or bpy.data.texts.new(name)
                    with open(path, "r", encoding="utf-8") as text_file:
                        text.from_string(text_file.read())
                    job.loaded_texts.append(text.name)
                elif output["kind"] == "image" and path.lower().endswith(".png"):
                    image = bpy.data.images.load(path, check_existing=True)
                    image.reload()
                    job.loaded_images.append(image.name)
            except Exception as err:  # pylint: disable=W0718
                _LOG.warn("Could not load a worker result into blender", (path, err))


class _RunCancelled(Exception):
    pass
