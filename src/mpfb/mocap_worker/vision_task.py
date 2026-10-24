# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  video or image through the mediapipe pose, hand and face landmarkers
# ------------------------------------------------------------------------------
"""Runs mediapipe landmarkers over every frame of a video or over a single image and writes the raw results:

    *_summary.txt        readable summary: detection rates, plausibility checks, blendshape statistics
    *_landmarks.npz      all raw arrays, NaN where nothing was detected (input for the later animation steps)
    *_blendshapes.csv    52 ARKit blendshape scores per frame
    *_pose.csv           33 body landmarks per frame (normalized image coordinates and visibility)
    *_annotated.mp4/png  the input with the landmarks drawn over it
    *_timeline.png       detections and the most active blendshapes in time (bar chart for an image)
"""

import math
import os
import shutil
import tempfile
import time

import numpy as np

import common
import drawing
from protocol import Progress, emit

# Annotated videos are written at most this large, no benefit in bigger visualisation files
_ANNOTATED_MAX_HEIGHT = 720

POSE_COUNT = 33
HAND_COUNT = 21
FACE_COUNT = 478
BLENDSHAPE_COUNT = 52


class _Models():
    """The enabled landmarkers. Models are passed as bytes, mediapipe can not open non-ascii paths on windows."""

    def __init__(self, task: dict, job: dict, video_mode: bool) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        self.vision = vision
        self.mp = mp
        mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        confidence = float(task.get("min_confidence", 0.5))
        self.pose = self.hands = self.face = None
        self.files = {}

        def options(name):
            path = os.path.join(job["models_dir"], task[name]["model"])
            self.files[name] = os.path.basename(path)
            with open(path, "rb") as model_file:
                return mp.tasks.BaseOptions(model_asset_buffer=model_file.read())

        if task["pose"]["enabled"]:
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=options("pose"), running_mode=mode, num_poses=1,
                min_pose_detection_confidence=confidence, min_pose_presence_confidence=confidence,
                min_tracking_confidence=confidence))
        if task["hands"]["enabled"]:
            self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=options("hands"), running_mode=mode, num_hands=2,
                min_hand_detection_confidence=confidence, min_hand_presence_confidence=confidence,
                min_tracking_confidence=confidence))
        if task["face"]["enabled"]:
            self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=options("face"), running_mode=mode, num_faces=1,
                min_face_detection_confidence=confidence, min_face_presence_confidence=confidence,
                min_tracking_confidence=confidence, output_face_blendshapes=True,
                output_facial_transformation_matrixes=True))

        self.pose_pairs = drawing.connection_pairs(vision.PoseLandmarksConnections.POSE_LANDMARKS)
        self.hand_pairs = drawing.connection_pairs(vision.HandLandmarksConnections.HAND_CONNECTIONS)
        self.face_pairs = drawing.connection_pairs(vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)
        self.pose_names = [landmark.name.lower() for landmark in vision.PoseLandmark]

    def close(self) -> None:
        for model in (self.pose, self.hands, self.face):
            if model is not None:
                model.close()


class _Frames():
    """Raw results of every processed frame, NaN where a model did not detect anything."""

    def __init__(self) -> None:
        self.times_ms: list[int] = []
        self.pose: list[np.ndarray] = []          # (33, 5) x, y, z, visibility, presence
        self.pose_world: list[np.ndarray] = []    # (33, 3) metres, origin between the hips
        self.hands: list[np.ndarray] = []         # (2, 21, 3) slot 0 = label "Left", slot 1 = label "Right"
        self.hands_world: list[np.ndarray] = []
        self.hand_scores: list[np.ndarray] = []   # (2,) handedness score
        self.face: list[np.ndarray] = []          # (478, 3)
        self.blendshapes: list[np.ndarray] = []   # (52,)
        self.face_matrix: list[np.ndarray] = []   # (4, 4)
        self.blendshape_names: list[str] = []

    def add(self, time_ms: int, pose_result, hand_result, face_result) -> None:
        self.times_ms.append(time_ms)

        pose = np.full((POSE_COUNT, 5), np.nan, dtype=np.float32)
        pose_world = np.full((POSE_COUNT, 3), np.nan, dtype=np.float32)
        if pose_result is not None and pose_result.pose_landmarks:
            for index, landmark in enumerate(pose_result.pose_landmarks[0][:POSE_COUNT]):
                pose[index] = (landmark.x, landmark.y, landmark.z, _or_nan(landmark.visibility), _or_nan(landmark.presence))
            if pose_result.pose_world_landmarks:
                for index, landmark in enumerate(pose_result.pose_world_landmarks[0][:POSE_COUNT]):
                    pose_world[index] = (landmark.x, landmark.y, landmark.z)
        self.pose.append(pose)
        self.pose_world.append(pose_world)

        hands = np.full((2, HAND_COUNT, 3), np.nan, dtype=np.float32)
        hands_world = np.full((2, HAND_COUNT, 3), np.nan, dtype=np.float32)
        scores = np.full(2, np.nan, dtype=np.float32)
        if hand_result is not None:
            for hand_index, landmarks in enumerate(hand_result.hand_landmarks):
                category = hand_result.handedness[hand_index][0]
                slot = 0 if (category.category_name or "").lower() == "left" else 1
                if not np.isnan(scores[slot]) and scores[slot] >= category.score:
                    continue  # both hands got the same label, keep the more confident one
                scores[slot] = category.score
                hands[slot] = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks[:HAND_COUNT]]
                if hand_result.hand_world_landmarks:
                    hands_world[slot] = [(landmark.x, landmark.y, landmark.z)
                                         for landmark in hand_result.hand_world_landmarks[hand_index][:HAND_COUNT]]
        self.hands.append(hands)
        self.hands_world.append(hands_world)
        self.hand_scores.append(scores)

        face = np.full((FACE_COUNT, 3), np.nan, dtype=np.float32)
        blendshapes = np.full(BLENDSHAPE_COUNT, np.nan, dtype=np.float32)
        matrix = np.full((4, 4), np.nan, dtype=np.float32)
        if face_result is not None and face_result.face_landmarks:
            for index, landmark in enumerate(face_result.face_landmarks[0][:FACE_COUNT]):
                face[index] = (landmark.x, landmark.y, landmark.z)
            if face_result.face_blendshapes:
                categories = face_result.face_blendshapes[0]
                if not self.blendshape_names:
                    self.blendshape_names = [category.category_name for category in categories]
                for index, category in enumerate(categories[:BLENDSHAPE_COUNT]):
                    blendshapes[index] = category.score
            if face_result.facial_transformation_matrixes:
                matrix = np.asarray(face_result.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
        self.face.append(face)
        self.blendshapes.append(blendshapes)
        self.face_matrix.append(matrix)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "times_ms": np.asarray(self.times_ms, dtype=np.int64),
            "pose": np.stack(self.pose), "pose_world": np.stack(self.pose_world),
            "hands": np.stack(self.hands), "hands_world": np.stack(self.hands_world),
            "hand_scores": np.stack(self.hand_scores),
            "face": np.stack(self.face), "blendshapes": np.stack(self.blendshapes),
            "face_matrix": np.stack(self.face_matrix),
            }


def _or_nan(value) -> float:
    return float("nan") if value is None else float(value)


def _detect(models: _Models, rgb: np.ndarray, time_ms: int | None):
    image = models.mp.Image(image_format=models.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    results = []
    for model in (models.pose, models.hands, models.face):
        if model is None:
            results.append(None)
        elif time_ms is None:
            results.append(model.detect(image))
        else:
            results.append(model.detect_for_video(image, time_ms))
    return results


def _annotate(models: _Models, frames: _Frames, image_bgr: np.ndarray, hud: list[str]) -> np.ndarray:
    import cv2
    height, width = image_bgr.shape[:2]
    if height > _ANNOTATED_MAX_HEIGHT:
        scale = _ANNOTATED_MAX_HEIGHT / height
        image_bgr = cv2.resize(image_bgr, (int(round(width * scale / 2) * 2), _ANNOTATED_MAX_HEIGHT),
                               interpolation=cv2.INTER_AREA)
    else:
        image_bgr = image_bgr.copy()
    if models.face is not None:
        drawing.draw_face(image_bgr, frames.face[-1], models.face_pairs)
    if models.pose is not None:
        drawing.draw_pose(image_bgr, frames.pose[-1], models.pose_pairs)
    if models.hands is not None:
        for slot in (0, 1):
            drawing.draw_hand(image_bgr, frames.hands[-1][slot], models.hand_pairs, is_left=slot == 0)
    drawing.draw_hud(image_bgr, hud)
    return image_bgr


def _hud(index: int, time_ms: int, frames: _Frames, models: _Models) -> list[str]:
    lines = ["frame %d  t=%.2fs" % (index, time_ms / 1000.0)]
    if models.pose is not None:
        lines.append("pose: " + ("yes" if not np.isnan(frames.pose[-1][0, 0]) else "-"))
    if models.hands is not None:
        hands = frames.hands[-1]
        labels = [name for slot, name in ((0, "L"), (1, "R")) if not np.isnan(hands[slot, 0, 0])]
        lines.append("hands: " + (" ".join(labels) if labels else "-"))
    if models.face is not None:
        blendshapes = frames.blendshapes[-1]
        if np.isnan(blendshapes[0]):
            lines.append("face: -")
        else:
            names = frames.blendshape_names
            top = np.argsort(np.nan_to_num(blendshapes, nan=0.0))[::-1][:3]
            lines.append("face: " + ", ".join("%s %.2f" % (names[i] if i < len(names) else i, blendshapes[i]) for i in top))
    return lines


class _VideoWriter():
    """H.264 mp4 through imageio-ffmpeg (universal), mp4v through opencv as fallback."""

    def __init__(self, path: str, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._ffmpeg = None
        self._cv = None

    def write(self, image_bgr: np.ndarray) -> None:
        if self._ffmpeg is None and self._cv is None:
            self._open(image_bgr.shape[1], image_bgr.shape[0])
        if self._ffmpeg is not None:
            self._ffmpeg.send(np.ascontiguousarray(image_bgr[:, :, ::-1]))
        else:
            self._cv.write(image_bgr)

    def _open(self, width: int, height: int) -> None:
        try:
            import imageio_ffmpeg
            self._ffmpeg = imageio_ffmpeg.write_frames(self.path, (width, height), fps=self.fps, codec="libx264",
                                                       pix_fmt_out="yuv420p", macro_block_size=2,
                                                       ffmpeg_log_level="error")
            self._ffmpeg.send(None)
        except Exception as err:  # pylint: disable=W0718
            emit("log", message="imageio-ffmpeg writer failed (%s), falling back to opencv" % err)
            import cv2
            self._ffmpeg = None
            self._cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))

    def close(self) -> None:
        if self._ffmpeg is not None:
            self._ffmpeg.close()
        if self._cv is not None:
            self._cv.release()


def _open_video(path: str):
    """cv2.VideoCapture, which also applies the rotation stored in the video metadata. If opencv can not open the path
    (non-ascii characters), the video is copied to a temporary file first."""
    import cv2
    capture = cv2.VideoCapture(path)
    if capture.isOpened():
        return capture, None
    capture.release()
    temp_dir = tempfile.mkdtemp(prefix="mpfb_mocap_")
    temp_path = os.path.join(temp_dir, "input" + os.path.splitext(path)[1])
    shutil.copyfile(path, temp_path)
    capture = cv2.VideoCapture(temp_path)
    if not capture.isOpened():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("Could not open the video " + path)
    return capture, temp_dir


def run(task: dict, job: dict) -> None:
    if common.is_image_file(task["input"]):
        _run_image(task, job)
    else:
        _run_video(task, job)


def _run_video(task: dict, job: dict) -> None:
    import cv2
    capture, temp_dir = _open_video(task["input"])
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if not fps or fps != fps or fps > 1000:
            emit("log", message="The video does not report a usable fps, assuming 30")
            fps = 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_s = max(0.0, float(task.get("start", 0.0)))
        end_s = float(task.get("end", 0.0))
        if end_s <= start_s:
            end_s = frame_count / fps if frame_count else float("inf")
        expected = max(0, int(math.ceil((end_s - start_s) * fps))) if math.isfinite(end_s) else 0

        progress = Progress(task["id"], "loading models")
        progress.update(0, force=True)
        models = _Models(task, job, video_mode=True)
        frames = _Frames()
        annotated_path = common.output_path(job, task, "_annotated.mp4")
        writer = _VideoWriter(annotated_path, fps) if task.get("annotate", True) else None
        progress = Progress(task["id"], "landmarks", expected, every=max(1, int(round(fps / 2))))

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        index = -1
        processed = 0
        last_time_ms = -1
        inference_seconds = 0.0
        try:
            while True:
                ok, image_bgr = capture.read()
                if not ok:
                    break
                index += 1
                time_s = index / fps
                if time_s < start_s - 1e-6:
                    continue
                if time_s > end_s + 1e-6:
                    break
                height, width = image_bgr.shape[:2]
                # VIDEO mode wants strictly increasing timestamps
                time_ms = max(last_time_ms + 1, int(round(time_s * 1000)))
                last_time_ms = time_ms

                began = time.perf_counter()
                results = _detect(models, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), time_ms)
                inference_seconds += time.perf_counter() - began
                frames.add(time_ms, *results)
                if writer is not None:
                    writer.write(_annotate(models, frames, image_bgr, _hud(index, time_ms, frames, models)))
                processed += 1
                progress.update(processed)
        finally:
            if writer is not None:
                writer.close()
            models.close()
        if processed == 0:
            raise ValueError("No frames were read in the range %.2f - %.2f s" % (start_s, end_s))
        progress.update(processed, force=True)

        info = {"kind": "video", "width": width, "height": height, "fps": fps, "frame_count": frame_count,
                "start": start_s, "end": min(end_s, (index + 1) / fps), "processed": processed,
                "inference_seconds": inference_seconds}
        _write_results(task, job, models, frames, info, annotated_path if writer is not None else None)
    finally:
        capture.release()
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _run_image(task: dict, job: dict) -> None:
    import cv2
    Progress(task["id"], "loading models").update(0, force=True)
    image_bgr = common.read_image_bgr(task["input"])
    models = _Models(task, job, video_mode=False)
    frames = _Frames()
    try:
        began = time.perf_counter()
        results = _detect(models, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), None)
        inference_seconds = time.perf_counter() - began
        frames.add(0, *results)
        annotated_path = common.output_path(job, task, "_annotated.png")
        common.write_image(annotated_path, _annotate(models, frames, image_bgr, _hud(0, 0, frames, models)))
    finally:
        models.close()
    height, width = image_bgr.shape[:2]
    info = {"kind": "image", "width": width, "height": height, "processed": 1, "inference_seconds": inference_seconds}
    _write_results(task, job, models, frames, info, annotated_path)


# ----------results-------------


def _head_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Pitch, yaw, roll in degrees from the facial transformation matrix (rotation, scale removed)."""
    rotation = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0, keepdims=True)
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -rotation[2, 0]))))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return pitch, yaw, roll


def _percent(count: int, total: int) -> str:
    return "%d/%d (%.1f %%)" % (count, total, 100.0 * count / total if total else 0.0)


def _summary(task: dict, models: _Models, frames: _Frames, arrays: dict, info: dict, files: list[str]) -> str:
    total = info["processed"]
    lines = ["MPFB motion capture - raw model output check", "=" * 60, "",
             "Input:       %s" % task["input"],
             "Kind:        %s, %d x %d px" % (info["kind"], info["width"], info["height"])]
    if info["kind"] == "video":
        lines.append("FPS:         %.3f (%d frames in the file)" % (info["fps"], info["frame_count"]))
        lines.append("Processed:   %d frames, %.2f - %.2f s" % (total, info["start"], info["end"]))
    lines.append("Models:      " + ", ".join("%s=%s" % item for item in models.files.items()))
    lines.append("Inference:   %.1f s total, %.1f ms per frame (all models, CPU)" % (
        info["inference_seconds"], 1000.0 * info["inference_seconds"] / max(1, total)))
    lines.append("Min. confidence: %.2f" % float(task.get("min_confidence", 0.5)))

    if models.pose is not None:
        pose = arrays["pose"]
        detected = ~np.isnan(pose[:, 0, 0])
        lines += ["", "POSE (33 landmarks)", "-" * 60, "Detected in:  " + _percent(int(detected.sum()), total)]
        if detected.any():
            world = arrays["pose_world"][detected]
            shoulders = np.linalg.norm(world[:, 11] - world[:, 12], axis=1)
            hips = np.linalg.norm(world[:, 23] - world[:, 24], axis=1)
            lines.append("Shoulder width (world, median): %.3f m   (adult approx. 0.30 - 0.45)" % float(np.nanmedian(shoulders)))
            lines.append("Hip width (world, median):      %.3f m   (adult approx. 0.20 - 0.35)" % float(np.nanmedian(hips)))
            visibility = np.nanmean(pose[detected][:, :, 3], axis=0)
            lines.append("Mean visibility per landmark:")
            for index, name in enumerate(models.pose_names[:POSE_COUNT]):
                lines.append("  %2d %-18s %.2f" % (index, name, visibility[index]))
            if info["kind"] == "image":
                lines.append("Landmarks (normalized x, y, z, visibility):")
                for index, name in enumerate(models.pose_names[:POSE_COUNT]):
                    lines.append("  %2d %-18s %7.3f %7.3f %7.3f %5.2f" % ((index, name) + tuple(pose[0, index, :4])))

    if models.hands is not None:
        hands = arrays["hands"]
        present = ~np.isnan(hands[:, :, 0, 0])
        counts = present.sum(axis=1)
        lines += ["", "HANDS (21 landmarks per hand)", "-" * 60,
                  "Frames with 0 / 1 / 2 hands: %d / %d / %d" % tuple(int((counts == n).sum()) for n in (0, 1, 2)),
                  "Label 'Left' detected in:  " + _percent(int(present[:, 0].sum()), total),
                  "Label 'Right' detected in: " + _percent(int(present[:, 1].sum()), total),
                  "Note: mediapipe labels hands as if the image was mirrored (selfie camera)."]
        scores = arrays["hand_scores"]
        if present.any():
            lines.append("Mean handedness score: %.2f" % float(np.nanmean(scores)))

    if models.face is not None:
        blendshapes = arrays["blendshapes"]
        detected = ~np.isnan(blendshapes[:, 0])
        names = frames.blendshape_names or ["blendshape_%d" % i for i in range(BLENDSHAPE_COUNT)]
        lines += ["", "FACE (478 landmarks, 52 ARKit blendshapes)", "-" * 60, "Detected in:  " + _percent(int(detected.sum()), total)]
        if detected.any():
            angles = np.array([_head_angles(matrix) for matrix in arrays["face_matrix"][detected]])
            for column, label in enumerate(("pitch (nod)", "yaw (turn)", "roll (tilt)")):
                lines.append("Head %-12s min %7.1f  max %7.1f  deg" % (label, angles[:, column].min(), angles[:, column].max()))
            values = blendshapes[detected]
            times = arrays["times_ms"][detected] / 1000.0
            if info["kind"] == "image":
                lines.append("Blendshapes (sorted by score):")
                for index in np.argsort(values[0])[::-1]:
                    lines.append("  %-22s %.3f" % (names[index], values[0, index]))
            else:
                lines.append("Blendshapes, sorted by max (a blink should peak eyeBlink*, talking jawOpen):")
                lines.append("  %-22s %6s %6s %6s %9s" % ("name", "mean", "min", "max", "max at s"))
                for index in np.argsort(values.max(axis=0))[::-1]:
                    column = values[:, index]
                    lines.append("  %-22s %6.3f %6.3f %6.3f %9.2f" % (names[index], column.mean(), column.min(),
                                                                      column.max(), times[int(np.argmax(column))]))

    lines += ["", "FILES", "-" * 60] + ["  " + os.path.basename(path) for path in files]
    return "\n".join(lines) + "\n"


def _write_results(task: dict, job: dict, models: _Models, frames: _Frames, info: dict, annotated_path: str | None) -> None:
    Progress(task["id"], "writing results").update(0, force=True)
    arrays = frames.arrays()
    names = frames.blendshape_names or ["blendshape_%d" % i for i in range(BLENDSHAPE_COUNT)]
    times_s = arrays["times_ms"] / 1000.0
    files = []

    npz_path = common.output_path(job, task, "_landmarks.npz")
    np.savez_compressed(npz_path, blendshape_names=np.array(names), pose_names=np.array(models.pose_names),
                        **{key: value for key, value in arrays.items()})
    files.append(npz_path)
    common.announce(task, "data", npz_path, "All landmarks (npz)")

    if models.face is not None:
        path = common.output_path(job, task, "_blendshapes.csv")
        common.write_csv(path, ["frame", "time_s"] + names,
                         ([index, float(times_s[index])] + [float(v) for v in row] for index, row in enumerate(arrays["blendshapes"])))
        files.append(path)
        common.announce(task, "csv", path, "Blendshapes (csv)")

    if models.pose is not None:
        path = common.output_path(job, task, "_pose.csv")
        header = ["frame", "time_s"] + ["%s_%s" % (name, axis) for name in models.pose_names[:POSE_COUNT] for axis in ("x", "y", "z", "vis")]
        common.write_csv(path, header, ([index, float(times_s[index])] + [float(v) for v in row[:, :4].reshape(-1)]
                                        for index, row in enumerate(arrays["pose"])))
        files.append(path)
        common.announce(task, "csv", path, "Pose (csv)")

    if annotated_path:
        files.append(annotated_path)
        common.announce(task, "video" if annotated_path.endswith(".mp4") else "image", annotated_path, "Annotated input")

    try:
        if info["kind"] == "video":
            path = common.output_path(job, task, "_timeline.png")
            detections = {}
            if models.pose is not None:
                detections["pose"] = ~np.isnan(arrays["pose"][:, 0, 0])
            if models.hands is not None:
                detections["hand 'Left'"] = ~np.isnan(arrays["hands"][:, 0, 0, 0])
                detections["hand 'Right'"] = ~np.isnan(arrays["hands"][:, 1, 0, 0])
            if models.face is not None:
                detections["face"] = ~np.isnan(arrays["blendshapes"][:, 0])
            drawing.plot_vision_timeline(path, times_s, detections,
                                         arrays["blendshapes"] if models.face is not None else None, names)
            files.append(path)
            common.announce(task, "image", path, "Timeline plot")
        elif models.face is not None and not np.isnan(arrays["blendshapes"][0, 0]):
            path = common.output_path(job, task, "_blendshapes.png")
            drawing.plot_blendshape_bars(path, arrays["blendshapes"][0], names)
            files.append(path)
            common.announce(task, "image", path, "Blendshapes plot")
    except ImportError as err:
        emit("log", message="No plot, matplotlib is not available: %s" % err)

    summary_path = common.output_path(job, task, "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(_summary(task, models, frames, arrays, info, files + [summary_path]))
    common.announce(task, "text", summary_path, "Summary")
