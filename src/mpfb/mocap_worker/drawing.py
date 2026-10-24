# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  visualisation of raw model outputs of the themotion capture worker
# ------------------------------------------------------------------------------
"""Landmarks drawn over the input frames, and plots of the outputs in time.
Used for user check, whether models produce something sensible, nothing here feeds the animation."""

import numpy as np

# BGR colours
POSE_COLOR = (80, 220, 80)
LEFT_COLOR = (255, 160, 40)    # left side of the body
RIGHT_COLOR = (40, 120, 255)
FACE_COLOR = (230, 230, 230)
TEXT_COLOR = (255, 255, 255)

# Body landmarks with odd index are on the body left side, even on the right
_POSE_FACE_INDICES = set(range(0, 11))


def connection_pairs(connections) -> list[tuple[int, int]]:
    """mediapipe gives connections as Connection(start, end) objects, older versions as tuples."""
    pairs = []
    for connection in connections:
        if hasattr(connection, "start"):
            pairs.append((int(connection.start), int(connection.end)))
        else:
            pairs.append((int(connection[0]), int(connection[1])))
    return pairs


def _to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.stack([points[:, 0] * width, points[:, 1] * height], axis=1).round().astype(np.int32)


def draw_skeleton(image, points: np.ndarray, pairs, color, radius: int, thickness: int,
                  visibility: np.ndarray | None = None, min_visibility: float = 0.3) -> None:
    """points: normalized (N, >=2) landmarks. Landmarks with low visibility are drawn hollow and greyed."""
    import cv2
    height, width = image.shape[:2]
    pixels = _to_pixels(points, width, height)
    visible = np.ones(len(points), dtype=bool) if visibility is None else visibility >= min_visibility
    for start, end in pairs:
        if start < len(points) and end < len(points):
            line_color = color if visible[start] and visible[end] else (110, 110, 110)
            cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), line_color, thickness, cv2.LINE_AA)
    for index, pixel in enumerate(pixels):
        if visible[index]:
            cv2.circle(image, tuple(pixel), radius, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(image, tuple(pixel), radius, (110, 110, 110), 1, cv2.LINE_AA)


def draw_pose(image, pose: np.ndarray, pairs) -> None:
    """pose: (33, 5) x, y, z, visibility, presence"""
    import cv2
    if np.isnan(pose[0, 0]):
        return
    scale = max(1, image.shape[0] // 360)
    draw_skeleton(image, pose, pairs, POSE_COLOR, 2 * scale, scale, visibility=pose[:, 3])
    height, width = image.shape[:2]
    pixels = _to_pixels(pose, width, height)
    # Colour the joints by side, so that a left/right swap of the model is visible at a glance
    for index in range(11, len(pose)):
        if pose[index, 3] >= 0.3:
            side = LEFT_COLOR if index % 2 == 1 else RIGHT_COLOR
            cv2.circle(image, tuple(pixels[index]), 3 * scale, side, -1, cv2.LINE_AA)


def draw_hand(image, hand: np.ndarray, pairs, is_left: bool) -> None:
    if np.isnan(hand[0, 0]):
        return
    scale = max(1, image.shape[0] // 540)
    draw_skeleton(image, hand, pairs, LEFT_COLOR if is_left else RIGHT_COLOR, 2 * scale, scale)


def draw_face(image, face: np.ndarray, pairs) -> None:
    if np.isnan(face[0, 0]):
        return
    draw_skeleton(image, face, pairs, FACE_COLOR, 0, 1)


def draw_hud(image, lines: list[str]) -> None:
    """A few lines of text in the top left corner."""
    import cv2
    scale = max(0.4, image.shape[0] / 1400)
    thickness = max(1, int(round(scale * 2)))
    line_height = int(28 * scale) + 6
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for line in lines) + 16
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, line_height * len(lines) + 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, dst=image)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (8, line_height * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR,
                    thickness, cv2.LINE_AA)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_vision_timeline(path: str, times_s: np.ndarray, detections: dict[str, np.ndarray],
                         blendshapes: np.ndarray | None, blendshape_names: list[str], top: int = 10) -> None:
    """Upper plot: which model detected something in which frame. Lower plot: the blendshapes which move the most."""
    plt = _pyplot()
    rows = 2 if blendshapes is not None else 1
    figure, axes = plt.subplots(rows, 1, figsize=(14, 3 + 4 * (rows - 1)), sharex=True, squeeze=False)
    axis = axes[0][0]
    for offset, (name, detected) in enumerate(detections.items()):
        axis.fill_between(times_s, offset, offset + detected.astype(float) * 0.8, step="mid", alpha=0.7, label=name)
    axis.set_yticks([index + 0.4 for index in range(len(detections))])
    axis.set_yticklabels(list(detections.keys()))
    axis.set_title("Detections per frame")
    if blendshapes is not None:
        axis = axes[1][0]
        spread = np.nan_to_num(np.nanmax(blendshapes, axis=0) - np.nanmin(blendshapes, axis=0), nan=0.0)
        for index in np.argsort(spread)[::-1][:top]:
            axis.plot(times_s, blendshapes[:, index], label=blendshape_names[index], linewidth=1.2)
        axis.set_ylim(0, 1)
        axis.set_ylabel("score")
        axis.set_title("Face blendshapes with the largest range (ARKit names)")
        axis.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1][0].set_xlabel("time [s]")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_blendshape_bars(path: str, scores: np.ndarray, names: list[str]) -> None:
    """All 52 blendshapes of one image, sorted by score."""
    plt = _pyplot()
    order = np.argsort(scores)
    figure, axis = plt.subplots(figsize=(8, 12))
    axis.barh([names[index] for index in order], scores[order], color="#4a90d9")
    axis.set_xlim(0, 1)
    axis.set_title("Face blendshapes")
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_audio(path: str, samples: np.ndarray, sample_rate: int, segments: list[dict], frame_times: np.ndarray,
               confidence: np.ndarray, speech: np.ndarray, start_offset: float = 0.0) -> None:
    """Waveform with the recognised phonemes, and the confidence of the best token per model frame."""
    plt = _pyplot()
    duration = len(samples) / sample_rate
    # Wider figures for longer recordings, so the phoneme labels stay readable (up to a limit)
    width = min(60, max(14, duration * 1.6))
    figure, (wave_axis, conf_axis) = plt.subplots(2, 1, figsize=(width, 6), sharex=True,
                                                   gridspec_kw={"height_ratios": [3, 1]})
    step = max(1, len(samples) // 200000)
    times = start_offset + np.arange(0, len(samples), step) / sample_rate
    wave_axis.plot(times, samples[::step], color="#7a7a7a", linewidth=0.5)
    peak = float(np.max(np.abs(samples))) if len(samples) else 1.0
    for index, segment in enumerate(segments):
        color = "#4a90d9" if index % 2 == 0 else "#d98c4a"
        wave_axis.axvspan(segment["start"], segment["end"], color=color, alpha=0.25)
        wave_axis.text((segment["start"] + segment["end"]) / 2, peak * (0.85 if index % 2 == 0 else 0.65),
                       segment["phoneme"], ha="center", fontsize=9)
    wave_axis.set_title("Waveform and recognised phonemes (wav2vec2, espeak IPA)")
    conf_axis.plot(frame_times, confidence, color="#2a7a2a", linewidth=0.8, label="best token probability")
    conf_axis.fill_between(frame_times, 0, speech.astype(float), color="#4a90d9", alpha=0.15, step="mid",
                           label="non-blank frame")
    conf_axis.set_ylim(0, 1.05)
    conf_axis.set_xlabel("time [s]")
    conf_axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)
