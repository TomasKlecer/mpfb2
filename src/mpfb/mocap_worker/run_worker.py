# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  entry point of the motion capture worker
# ------------------------------------------------------------------------------
"""Entry point of the motion capture worker.
It runs in the python of the motion capture environment started by services/mocaprunservice.py as:

    <venv python> run_worker.py <job.json>

This folder has no __init__.py: it is not part of MPFB2 package and must not be imported by blender.
Nothing here may import bpy, and the mediapipe or torch are only imported by the task that needs them.

job.json:
    {
      "protocol": 1,
      "output_dir": "...",            every output file is written here
      "models_dir": ".../ai_models",  model paths in the tasks are relative to this
      "tasks": [
        {"id": "video1", "type": "vision", "label": "Video 1: clip.mp4", "input": "...", "start": 0.0, "end": 0.0, "min_confidence": 0.5,
         "annotate": true,
         "pose":  {"enabled": true, "model": "mediapipe/pose_landmarker_full.task"},
         "hands": {"enabled": true, "model": "mediapipe/hand_landmarker.task"},
         "face":  {"enabled": true, "model": "mediapipe/face_landmarker.task"}},
        {"id": "audio", "type": "audio", "label": "Audio: speech.wav", "input": "...", "start": 0.0, "end": 0.0, "device": "auto",
         "model": "wav2vec2-xlsr-53-espeak-cv-ft"}
      ]
    }

end <= start means "until the end of the input".
"""

import json
import os
import sys
import traceback

# Make the modules next to this file importable however the worker was started
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import PROTOCOL_VERSION, emit, setup_stdout  # noqa: E402

_LABELS = {"vision": "Video / image", "audio": "Audio"}


def run_task(task: dict, job: dict) -> None:
    if not os.path.isfile(task["input"]):
        raise FileNotFoundError("The input file does not exist: " + task["input"])
    if task["type"] == "vision":
        import vision_task
        vision_task.run(task, job)
    elif task["type"] == "audio":
        import audio_task
        audio_task.run(task, job)
    else:
        raise ValueError("Unknown task type " + str(task["type"]))


def main(argv: list[str]) -> int:
    setup_stdout()
    if len(argv) != 2:
        print("usage: run_worker.py <job.json>")
        return 2
    with open(argv[1], "r", encoding="utf-8") as job_file:
        job = json.load(job_file)
    if job.get("protocol") != PROTOCOL_VERSION:
        emit("finished", failures=1)
        print("Unsupported job protocol %s, expected %d" % (job.get("protocol"), PROTOCOL_VERSION))
        return 2
    os.makedirs(job["output_dir"], exist_ok=True)

    tasks = job.get("tasks", [])
    failures = 0
    for index, task in enumerate(tasks):
        emit("task_start", task=task["id"], index=index, count=len(tasks), label=task.get("label") or _LABELS.get(task["type"], task["type"]))
        try:
            run_task(task, job)
            emit("task_done", task=task["id"])
        except Exception as err:  # pylint: disable=W0718
            failures += 1
            details = traceback.format_exc()
            print(details)
            emit("task_failed", task=task["id"], message="%s: %s" % (type(err).__name__, err), traceback=details)
    emit("finished", failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
