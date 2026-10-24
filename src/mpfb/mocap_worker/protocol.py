# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  messages the motion capture worker sends to blender
# ------------------------------------------------------------------------------
"""The worker informes blender through its stdout. Every message is one line: the prefix below followed by a json
object with "event" key. Everything else on stdout/stderr (mediapipe and torch logs) is only copied into the log by blender.

Events:
    task_start   task, index, count, label        task (one input) starts
    progress     task, stage, current, total      progress within a task, total=0 when unknown
    output       task, kind, path, label          file was written (kind: text, image, video, csv, data)
    task_done    task                             task finished
    task_failed  task, message, traceback         task failed, the worker goes on with the next one
    finished     failures                         the worker is done
"""

import json
import sys

PROTOCOL_VERSION = 1
MESSAGE_PREFIX = "@@MPFB "


def setup_stdout() -> None:
    """Blender reads the output as utf-8 line by line.."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def emit(event: str, **fields) -> None:
    message = {"event": event}
    message.update(fields)
    sys.stdout.write(MESSAGE_PREFIX + json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class Progress():
    """Sends progress of one stage, at most every few steps, so blender is not flooded with messages."""

    def __init__(self, task: str, stage: str, total: int = 0, every: int = 10) -> None:
        self.task = task
        self.stage = stage
        self.total = total
        self.every = max(1, every)

    def update(self, current: int, force: bool = False) -> None:
        if force or current % self.every == 0 or current == self.total:
            emit("progress", task=self.task, stage=self.stage, current=current, total=self.total)
