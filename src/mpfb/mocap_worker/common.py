# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  helpers shared by tasks of the motion capture worker
# ------------------------------------------------------------------------------
"""Small helpers of the worker. cv2.imread and cv2.imwrite can not open paths with non-ascii characters on windows,
so images are read and written through numpy and cv2 in-memory codecs."""

import csv
import math
import os
import subprocess
import sys

import numpy as np

from protocol import emit

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_IS_WINDOWS = sys.platform.startswith("win")
CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if _IS_WINDOWS else 0


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def read_image_bgr(path: str) -> np.ndarray:
    import cv2
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the image " + path)
    return image


def write_image(path: str, image_bgr: np.ndarray) -> None:
    import cv2
    ok, encoded = cv2.imencode(os.path.splitext(path)[1] or ".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode the image " + path)
    encoded.tofile(path)


def output_path(job: dict, task: dict, suffix: str) -> str:
    """<output dir>/<task id>_<input file name><suffix>, e.g. video_myclip_summary.txt"""
    stem = os.path.splitext(os.path.basename(task["input"]))[0]
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)[:40]
    return os.path.join(job["output_dir"], "%s_%s%s" % (task["id"], safe, suffix))


def announce(task: dict, kind: str, path: str, label: str) -> None:
    emit("output", task=task["id"], kind=kind, path=path, label=label)


def write_csv(path: str, header: list[str], rows) -> None:
    """NaN (no detection in that frame) is written as an empty cell."""
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if isinstance(value, float) and math.isnan(value) else
                             (round(value, 6) if isinstance(value, float) else value) for value in row])


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def format_seconds(seconds: float) -> str:
    return "%d:%06.3f" % (int(seconds // 60), seconds % 60)
