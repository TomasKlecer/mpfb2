# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         20.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  setup of the external python environment for the motion capture models
# ------------------------------------------------------------------------------
"""The motion capture models (mediapipe, wav2vec2) run in a separate python (not in blender) in a virtual uv environment.
This service manages build of this environment. The build runs on a background thread one subprocess at a time,
so that blender UI stays responsive. The UI only reads the state of the job.

Layout of the environment directory (by default <mpfb user home>/mocap_env):

    uv/                       uv itself, when it was installed through blender pip
    python/                   the python interpreter downloaded by uv
    cache/                    uv download cache
    venv/                     the virtual environment the models run in
    install.log               output of the last install attempt
    mpfb_mocap_env.json       written last, only after a successful install and verification

The models are not part of the environment. They ship with MPFB2 in data/ai_models,
which also contains models.json - registry of the models with their licences and checksums.
"""

import datetime
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import bpy

from .. import MPFB_CONTEXTUAL_INFORMATION, get_preference
from .logservice import LogService
from .locationservice import LocationService

_LOG = LogService.get_logger("services.mocapenvservice")

# An environment built for an older spec is not considered installed, so the install operator shows up again.
ENV_SPEC_VERSION = 1

PYTHON_VERSION = "3.12"

# mediapipe pulls in numpy, opencv-contrib-python, matplotlib and sounddevice by itself.
# opencv-python must not be added on top of it, both packages provide cv2 module and would overwrite each other.
PACKAGES = [
    "mediapipe>=0.10.20",
    "torch>=2.4",
    "transformers>=4.40",
    "scipy",
    "soundfile",
    "imageio-ffmpeg",
    ]

# Run the new environment as the last install step. Fails if any of the packages can not be imported,
# and prints the versions (and if torch sees GPU) as json on its last line.
_VERIFY_SCRIPT = """
import json, sys
import numpy, cv2, mediapipe, torch, transformers, scipy, soundfile, imageio_ffmpeg
print(json.dumps({
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "mediapipe": mediapipe.__version__,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "scipy": scipy.__version__,
    "cuda": bool(torch.cuda.is_available()),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(),
}))
"""

MARKER_FILE = "mpfb_mocap_env.json"
LOG_FILE = "install.log"
MODELS_REGISTRY = "models.json"

_IS_WINDOWS = sys.platform.startswith("win")
_CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if _IS_WINDOWS else 0


class InstallJob():
    """State of one install run. Writen by worker and read by UI."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def __init__(self, env_dir: str = "") -> None:
        self.env_dir = env_dir
        self.status = InstallJob.IDLE
        self.step_index = 0
        self.step_count = 0
        self.step_label = ""
        self.last_line = ""
        self.error = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self.cancel_requested = False
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self.status == InstallJob.RUNNING

    @property
    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at

    @property
    def log_path(self) -> str:
        return os.path.join(self.env_dir, LOG_FILE)


class _InstallCancelled(Exception):
    pass


class _InstallFailed(Exception):
    pass


_JOB = InstallJob()
_MARKER_CACHE: tuple[str | None, float, dict | None] = (None, 0.0, None)
_REGISTRY_CACHE: dict | None = None


class MocapEnvService():
    """Locating, checking and building the python environment of motion capture models."""

    def __init__(self) -> None:
        raise RuntimeError("You should not instance MocapEnvService. Use its static methods instead.")

    # ------------ Locations ---------------

    @staticmethod
    def get_preferences() -> Any:
        """The addon preferences object, needed by the UI to draw the preference properties in the panel."""
        package = MPFB_CONTEXTUAL_INFORMATION["__package__"] if MPFB_CONTEXTUAL_INFORMATION else None
        if package and package in bpy.context.preferences.addons:
            return bpy.context.preferences.addons[package].preferences
        return None

    @staticmethod
    def get_default_env_dir() -> str:
        return LocationService.get_user_home("mocap_env")

    @staticmethod
    def get_env_dir() -> str:
        """Directory picked in the preferences, or default in user home when nothing was picked."""
        chosen = get_preference("mocap_env_dir")
        if chosen and str(chosen).strip():
            return os.path.abspath(bpy.path.abspath(os.path.expanduser(str(chosen).strip())))
        return MocapEnvService.get_default_env_dir()

    @staticmethod
    def get_own_uv_binary() -> str | None:
        """Path to the users own uv binary, or None when the user did not ask to use one."""
        if not get_preference("mocap_use_own_uv"):
            return None
        path = get_preference("mocap_uv_binary")
        if not path or not str(path).strip():
            return None
        return os.path.abspath(bpy.path.abspath(os.path.expanduser(str(path).strip())))

    @staticmethod
    def get_venv_dir(env_dir: str) -> str:
        return os.path.join(env_dir, "venv")

    @staticmethod
    def get_venv_python(env_dir: str) -> str:
        venv = MocapEnvService.get_venv_dir(env_dir)
        if _IS_WINDOWS:
            return os.path.join(venv, "Scripts", "python.exe")
        return os.path.join(venv, "bin", "python")

    @staticmethod
    def get_ai_models_dir(sub_path: str | None = None) -> str:
        if sub_path:
            return LocationService.get_mpfb_data(os.path.join("ai_models", sub_path))
        return LocationService.get_mpfb_data("ai_models")

    # ------------ State checks----------------------

    @staticmethod
    def read_marker(env_dir: str) -> dict | None:
        """Contents of the marker written at the end of a successful install."""
        # The panel asks on every redraw, file should be parsed again only after change
        global _MARKER_CACHE
        path = os.path.join(env_dir, MARKER_FILE)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        if _MARKER_CACHE[0] == path and _MARKER_CACHE[1] == mtime:
            return _MARKER_CACHE[2]
        try:
            with open(path, "r", encoding="utf-8") as json_file:
                marker = json.load(json_file)
        except (OSError, ValueError) as err:
            _LOG.warn("Could not read the mocap environment marker", (path, err))
            marker = None
        _MARKER_CACHE = (path, mtime, marker)
        return marker

    @staticmethod
    def is_env_installed(env_dir: str | None = None) -> bool:
        """True when the environment was completely built."""
        if env_dir is None:
            env_dir = MocapEnvService.get_env_dir()
        marker = MocapEnvService.read_marker(env_dir)
        if not marker or marker.get("spec_version") != ENV_SPEC_VERSION:
            return False
        return os.path.isfile(MocapEnvService.get_venv_python(env_dir))

    @staticmethod
    def load_models_registry() -> dict:
        """Cached contents of the model registry shipped with the addon (data/ai_models/models.json)."""
        global _REGISTRY_CACHE
        if _REGISTRY_CACHE is not None:
            return _REGISTRY_CACHE
        path = MocapEnvService.get_ai_models_dir(MODELS_REGISTRY)
        if not os.path.isfile(path):
            _LOG.warn("The model registry is missing", path)
            return {"models": []}
        with open(path, "r", encoding="utf-8") as json_file:
            _REGISTRY_CACHE = json.load(json_file)
        return _REGISTRY_CACHE

    @staticmethod
    def get_missing_models() -> list[str]:
        """Ids of the models in the registry which have one or more files missing from data/ai_models."""
        missing = []
        for model in MocapEnvService.load_models_registry().get("models", []):
            for file_info in model.get("files", []):
                if not os.path.isfile(MocapEnvService.get_ai_models_dir(file_info["path"])):
                    missing.append(model["id"])
                    break
        return missing

    # --------------- Install commands ----------------

    @staticmethod
    def find_uv_in_dir(uv_dir: str) -> str | None:
        """Where pip --target put the uv binary."""
        name = "uv.exe" if _IS_WINDOWS else "uv"
        preferred = os.path.join(uv_dir, "bin", name)
        if os.path.isfile(preferred):
            return preferred
        for candidate in glob.glob(os.path.join(uv_dir, "**", name), recursive=True):
            if os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def build_uv_environment(env_dir: str) -> dict[str, str]:
        """Process environment for uv. Keep everything inside env_dir and ignore users own uv config."""
        env = dict(os.environ)
        # Neither Blenders nor users python settings can leak into the new interpreter
        for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
            env.pop(key, None)
        env["UV_CACHE_DIR"] = os.path.join(env_dir, "cache")
        env["UV_PYTHON_INSTALL_DIR"] = os.path.join(env_dir, "python")
        env["UV_PYTHON_PREFERENCE"] = "only-managed"
        env["UV_NO_CONFIG"] = "1"
        env["NO_COLOR"] = "1"
        # Trust the certificates in the system store (as pip does) so uv does not fail on machines with antivirus or proxies
        env["UV_SYSTEM_CERTS"] = "true"
        env["UV_NATIVE_TLS"] = "true"
        env["PYTHONNOUSERSITE"] = "1"
        return env

    @staticmethod
    def build_install_steps(own_uv: str | None) -> list[tuple[str, str]]:
        """The steps of an install as (key, label) pairs. The commands themselves are put together when the step runs,
        since the uv binary installed by the first step is only known after it ran."""
        steps = []
        if own_uv:
            steps.append(("check_uv", "Checking uv"))
        else:
            steps.append(("install_uv", "Installing uv into blender python"))
        steps.append(("install_python", "Downloading Python %s" % PYTHON_VERSION))
        steps.append(("create_venv", "Creating the virtual environment"))
        steps.append(("install_packages", "Installing packages (several GB with CUDA)"))
        steps.append(("verify", "Verifying the environment"))
        return steps

    @staticmethod
    def build_command(step: str, env_dir: str, uv: str | None, blender_python: str) -> list[list[str]]:
        """Command lines for one install step. Most steps are one command, installing uv may need to bootstrap pip first."""
        venv_python = MocapEnvService.get_venv_python(env_dir)
        if step == "install_uv":
            target = os.path.join(env_dir, "uv")
            return [[blender_python, "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
                     "--no-warn-script-location", "--target", target, "uv"]]
        if step == "check_uv":
            return [[uv, "--version"]]
        if step == "install_python":
            return [[uv, "python", "install", PYTHON_VERSION]]
        if step == "create_venv":
            return [[uv, "venv", "--clear", "--python", PYTHON_VERSION, MocapEnvService.get_venv_dir(env_dir)]]
        if step == "install_packages":
            # --torch-backend auto picks the CUDA build of torch when an nvidia driver is found and CPU build otherwise
            return [[uv, "pip", "install", "--python", venv_python, "--torch-backend", "auto"] + PACKAGES]
        if step == "verify":
            return [[venv_python, "-c", _VERIFY_SCRIPT]]
        raise ValueError("Unknown install step " + step)

    # ------------ Install job -------------------

    @staticmethod
    def get_job() -> InstallJob:
        return _JOB

    @staticmethod
    def start_install(env_dir: str, own_uv: str | None) -> InstallJob:
        """Start building the environment on background thread and return the job, which the UI can poll."""
        global _JOB
        if _JOB.is_running:
            raise RuntimeError("A motion capture environment install is already running")
        if own_uv and not os.path.isfile(own_uv):
            raise FileNotFoundError("The selected uv binary does not exist: " + own_uv)

        os.makedirs(env_dir, exist_ok=True)
        job = InstallJob(env_dir)
        steps = MocapEnvService.build_install_steps(own_uv)
        job.step_count = len(steps)
        job.status = InstallJob.RUNNING
        job.started_at = time.time()
        _JOB = job

        job.thread = threading.Thread(target=MocapEnvService._run_install, args=(job, steps, own_uv, sys.executable),
                                      name="mpfb-mocap-env-install", daemon=True)
        job.thread.start()
        return job

    @staticmethod
    def cancel_install() -> None:
        """Ask the running install to stop. The current subprocess is terminated, the thread then ends."""
        job = _JOB
        if not job.is_running:
            return
        job.cancel_requested = True
        process = job.process
        if process is not None and process.poll() is None:
            _LOG.info("Terminating the install subprocess")
            MocapEnvService._kill_process_tree(process)

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen) -> None:
        """Kill the process together with its children (uv and pip start own subprocesses)."""
        try:
            if _IS_WINDOWS:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], creationflags=_CREATION_FLAGS,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except OSError as err:
            _LOG.warn("Could not kill the install process tree, terminating only the process", err)
            try:
                process.terminate()
            except OSError:
                pass

    @staticmethod
    def _run_install(job: InstallJob, steps: list[tuple[str, str]], own_uv: str | None, blender_python: str) -> None:
        """Body of the worker thread. Must not touch bpy (not thread safe)."""
        env_dir = job.env_dir
        env = MocapEnvService.build_uv_environment(env_dir)
        uv = own_uv
        verification = None

        try:
            # An outdated marker must not survive rebuild, the environment is only considered installed once it is rewritten
            marker_path = os.path.join(env_dir, MARKER_FILE)
            if os.path.exists(marker_path):
                os.remove(marker_path)

            with open(job.log_path, "w", encoding="utf-8") as log:
                log.write("MPFB motion capture environment install, %s\n" % datetime.datetime.now().isoformat())
                log.write("env_dir: %s\nblender python: %s\n" % (env_dir, blender_python))

                for index, (step, label) in enumerate(steps):
                    job.step_index = index + 1
                    job.step_label = label
                    job.last_line = ""
                    log.write("\n=== Step %d/%d: %s\n" % (index + 1, len(steps), label))

                    if step == "install_uv":
                        MocapEnvService._ensure_pip(job, log, blender_python, env)
                    for command in MocapEnvService.build_command(step, env_dir, uv, blender_python):
                        output = MocapEnvService._run_command(job, log, command, env)
                        if step == "verify":
                            verification = json.loads(output.strip().splitlines()[-1])
                    if step == "install_uv":
                        uv = MocapEnvService.find_uv_in_dir(os.path.join(env_dir, "uv"))
                        if not uv:
                            raise _InstallFailed("pip finished, but the uv binary was not found in " + os.path.join(env_dir, "uv"))
                        log.write("uv binary: %s\n" % uv)

                marker = {
                    "spec_version": ENV_SPEC_VERSION,
                    "python_version": PYTHON_VERSION,
                    "packages": PACKAGES,
                    "uv": uv,
                    "installed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "versions": verification,
                    }
                with open(marker_path, "w", encoding="utf-8") as json_file:
                    json.dump(marker, json_file, indent=2)
                log.write("\nInstall finished: %s\n" % json.dumps(verification))

            job.status = InstallJob.SUCCEEDED
            _LOG.info("Motion capture environment installed", env_dir)
        except _InstallCancelled:
            job.status = InstallJob.CANCELLED
            job.error = "Cancelled by user"
            _LOG.info("Motion capture environment install was cancelled")
        except Exception as err:  # pylint: disable=W0718
            job.status = InstallJob.FAILED
            job.error = str(err)
            _LOG.error("Motion capture environment install failed", err)
            try:
                with open(job.log_path, "a", encoding="utf-8") as log:
                    log.write("\nFAILED: %s\n" % err)
            except OSError:
                pass
        finally:
            job.process = None
            job.finished_at = time.time()

    @staticmethod
    def _ensure_pip(job: InstallJob, log: Any, blender_python: str, env: dict[str, str]) -> None:
        """Blender python may not come with pip. Bootstrap it with ensurepip when it is missing."""
        try:
            MocapEnvService._run_command(job, log, [blender_python, "-m", "pip", "--version"], env)
            return
        except _InstallFailed:
            log.write("pip is not available in blender's python, running ensurepip\n")
        MocapEnvService._run_command(job, log, [blender_python, "-m", "ensurepip", "--upgrade"], env)

    @staticmethod
    def _run_command(job: InstallJob, log: Any, command: list[str], env: dict[str, str]) -> str:
        """Run one command, streaming its output into the log and into job.last_line. Returns the whole output."""
        if job.cancel_requested:
            raise _InstallCancelled()
        _LOG.debug("Running", command)
        # The verify script is long and multi-line
        log.write("$ %s\n" % " ".join(arg if "\n" not in arg else "<script>" for arg in command))
        log.flush()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       env=env, cwd=job.env_dir, creationflags=_CREATION_FLAGS,
                                       # Own process group, so cancel can kill the children too
                                       start_new_session=not _IS_WINDOWS,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as err:
            raise _InstallFailed("Could not start %s: %s" % (command[0], err)) from err
        job.process = process
        # Cancel which came between the check above and now did not see this process
        if job.cancel_requested:
            MocapEnvService._kill_process_tree(process)

        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            log.write(line)
            log.flush()
            stripped = line.strip()
            if stripped:
                job.last_line = stripped
        return_code = process.wait()
        job.process = None

        if job.cancel_requested:
            raise _InstallCancelled()
        if return_code != 0:
            raise _InstallFailed("%s exited with code %d: %s" % (os.path.basename(command[0]), return_code, job.last_line))
        return "".join(lines)
