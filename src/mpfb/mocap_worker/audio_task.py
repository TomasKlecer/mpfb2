# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  audio through the wav2vec2 phoneme recognizer
# ------------------------------------------------------------------------------
"""Decodes audio (or the audio track of a video) with ffmpeg to 16 kHz mono, runs wav2vec2-xlsr-53-espeak-cv-ft
over it and decodes the CTC output into timed phonemes (espeak IPA). Writes:

    *_summary.txt     readable summary: recognised phoneme string, timing table, speed, confidence
    *_phonemes.csv    one row per phoneme: start, end, phoneme, mean probability
    *_logits.npz      log probabilities of every token in every model frame (input for the later viseme step)
    *_phonemes.png    waveform with the phonemes, and the confidence of the model in time

The tokenizer shipped with the model wants phonemizer package and espeak, which are only needed to turn text into
phonemes. Decoding the model output only needs vocab.json, so it is read directly.
"""

import json
import os
import subprocess
import time

import numpy as np

import common
import drawing
from protocol import Progress, emit

SAMPLE_RATE = 16000
# The model sees the recording in windows of this length. Attention cost grows with the square of the length
# (20s are ok for 4 GB GPU). The windows overlap, the frames of the overlap are taken from the window
# in which they are further from the edge.
_WINDOW_SECONDS = 20.0
_OVERLAP_SECONDS = 1.0
# The CNN feature encoder of wav2vec2 turns every 320 samples (20 ms) into one frame
_SAMPLES_PER_FRAME = 320

_SPECIAL_TOKENS = {"<pad>", "<s>", "</s>", "<unk>"}


def load_audio(path: str, start: float = 0.0, end: float = 0.0) -> np.ndarray:
    """float32 mono samples at 16 kHz, decoded by ffmpeg (any format ffmpeg knows, including videos)."""
    command = [common.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start > 0:
        command += ["-ss", "%.3f" % start]
    command += ["-i", path]
    if end > start:
        command += ["-t", "%.3f" % (end - start)]
    command += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=common.CREATION_FLAGS,
                            check=False)
    if result.returncode != 0:
        raise ValueError("ffmpeg could not decode the audio: " + result.stderr.decode("utf-8", "replace").strip()[-400:])
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("The file has no audio in the selected range")
    return samples


def load_vocab(model_dir: str) -> list[str]:
    with open(os.path.join(model_dir, "vocab.json"), "r", encoding="utf-8") as vocab_file:
        vocab = json.load(vocab_file)
    tokens = [""] * (max(vocab.values()) + 1)
    for token, index in vocab.items():
        tokens[index] = token
    return tokens


def window_bounds(sample_count: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Start and end sample of each window. The last window is shifted back so that it is full length when possible."""
    if sample_count <= window:
        return [(0, sample_count)]
    bounds = []
    step = window - overlap
    start = 0
    while True:
        end = start + window
        if end >= sample_count:
            bounds.append((max(0, sample_count - window), sample_count))
            break
        bounds.append((start, end))
        start += step
    return bounds


def stitch_windows(window_logprobs: list[np.ndarray], bounds: list[tuple[int, int]], total_frames: int) -> np.ndarray:
    """Merge the window outputs into one array. Every global frame is taken from the window in
    which it lies furthest from the window edge, where the model saw the most context."""
    vocab_size = window_logprobs[0].shape[1]
    merged = np.zeros((total_frames, vocab_size), dtype=np.float32)
    best_margin = np.full(total_frames, -1, dtype=np.int64)
    for logprobs, (start, _end) in zip(window_logprobs, bounds):
        offset = start // _SAMPLES_PER_FRAME
        count = min(len(logprobs), total_frames - offset)
        if count <= 0:
            continue
        local = np.arange(count)
        margin = np.minimum(local, len(logprobs) - 1 - local)
        target = offset + local
        better = margin > best_margin[target]
        merged[target[better]] = logprobs[local[better]]
        best_margin[target[better]] = margin[better]
    return merged


def ctc_segments(logprobs: np.ndarray, tokens: list[str], blank: int, frame_seconds: float, offset: float = 0.0) -> list[dict]:
    """Greedy CTC decoding: the best token in every frame, runs of the same token merged, blank dropped.
    Every segment has the time span of its run of frames and the mean probability of the token over the run."""
    best = logprobs.argmax(axis=1)
    probability = np.exp(logprobs[np.arange(len(best)), best])
    segments = []
    index = 0
    while index < len(best):
        token = int(best[index])
        end = index
        while end + 1 < len(best) and best[end + 1] == token:
            end += 1
        text = tokens[token] if token < len(tokens) else "?"
        if token != blank and text not in _SPECIAL_TOKENS:
            segments.append({
                "phoneme": " " if text == "|" else text,
                "start": offset + index * frame_seconds,
                "end": offset + (end + 1) * frame_seconds,
                "probability": float(probability[index:end + 1].mean()),
                })
        index = end + 1
    return segments


def run(task: dict, job: dict) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC

    model_dir = os.path.join(job["models_dir"], task["model"])
    start = max(0.0, float(task.get("start", 0.0)))
    end = float(task.get("end", 0.0))

    Progress(task["id"], "decoding audio").update(0, force=True)
    samples = load_audio(task["input"], start, end)
    duration = len(samples) / SAMPLE_RATE

    Progress(task["id"], "loading model").update(0, force=True)
    began = time.perf_counter()
    wanted = task.get("device", "auto")
    device = "cuda" if wanted in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    load_seconds = time.perf_counter() - began
    tokens = load_vocab(model_dir)
    blank = tokens.index("<pad>") if "<pad>" in tokens else 0

    bounds = window_bounds(len(samples), int(_WINDOW_SECONDS * SAMPLE_RATE), int(_OVERLAP_SECONDS * SAMPLE_RATE))
    progress = Progress(task["id"], "phonemes", len(bounds), every=1)
    window_logprobs = []
    began = time.perf_counter()
    with torch.inference_mode():
        for index, (window_start, window_end) in enumerate(bounds):
            chunk = samples[window_start:window_end]
            chunk = (chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7)
            inputs = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(inputs).logits[0]
            window_logprobs.append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
            progress.update(index + 1)
    inference_seconds = time.perf_counter() - began
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    total_frames = max(len(window_logprobs[-1]) + bounds[-1][0] // _SAMPLES_PER_FRAME, len(window_logprobs[0]))
    logprobs = stitch_windows(window_logprobs, bounds, total_frames)
    frame_seconds = _SAMPLES_PER_FRAME / SAMPLE_RATE
    segments = ctc_segments(logprobs, tokens, blank, frame_seconds, offset=start)

    Progress(task["id"], "writing results").update(0, force=True)
    files = []
    frame_times = start + (np.arange(len(logprobs)) + 0.5) * frame_seconds
    best = logprobs.argmax(axis=1)
    confidence = np.exp(logprobs.max(axis=1))
    speech = best != blank

    npz_path = common.output_path(job, task, "_logits.npz")
    np.savez_compressed(npz_path, logprobs=logprobs.astype(np.float16), frame_times=frame_times,
                        tokens=np.array(tokens), blank=blank, frame_seconds=frame_seconds)
    files.append(npz_path)
    common.announce(task, "data", npz_path, "Log probabilities (npz)")

    csv_path = common.output_path(job, task, "_phonemes.csv")
    common.write_csv(csv_path, ["start_s", "end_s", "phoneme", "probability"],
                     ([segment["start"], segment["end"], segment["phoneme"], segment["probability"]] for segment in segments))
    files.append(csv_path)
    common.announce(task, "csv", csv_path, "Phonemes (csv)")

    try:
        plot_path = common.output_path(job, task, "_phonemes.png")
        drawing.plot_audio(plot_path, samples, SAMPLE_RATE, [s for s in segments if s["phoneme"].strip()],
                           frame_times, confidence, speech, start_offset=start)
        files.append(plot_path)
        common.announce(task, "image", plot_path, "Phoneme plot")
    except ImportError as err:
        emit("log", message="No plot, matplotlib is not available: %s" % err)

    summary_path = common.output_path(job, task, "_summary.txt")
    files.append(summary_path)
    info = {"duration": duration, "start": start, "device": device, "load_seconds": load_seconds,
            "inference_seconds": inference_seconds, "windows": len(bounds), "frames": len(logprobs),
            "speech_fraction": float(speech.mean()) if len(speech) else 0.0,
            "mean_confidence": float(confidence.mean()) if len(confidence) else 0.0}
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(_summary(task, info, segments, files))
    common.announce(task, "text", summary_path, "Summary")


def _summary(task: dict, info: dict, segments: list[dict], files: list[str]) -> str:
    phonemes = [segment["phoneme"] for segment in segments]
    counts = {}
    for phoneme in phonemes:
        if phoneme.strip():
            counts[phoneme] = counts.get(phoneme, 0) + 1
    lines = ["MPFB motion capture - raw model output check", "=" * 60, "",
             "Input:       %s" % task["input"],
             "Range:       %.2f - %.2f s (%.2f s of audio)" % (info["start"], info["start"] + info["duration"], info["duration"]),
             "Model:       %s on %s" % (task["model"], info["device"].upper()),
             "Load time:   %.1f s" % info["load_seconds"],
             "Inference:   %.2f s for %d window(s), %.0fx faster than real time" % (
                 info["inference_seconds"], info["windows"], info["duration"] / max(1e-6, info["inference_seconds"])),
             "Frames:      %d of 20 ms, %.1f %% not blank" % (info["frames"], 100.0 * info["speech_fraction"]),
             "Mean probability of the best token: %.2f" % info["mean_confidence"],
             "",
             "RECOGNISED PHONEMES (espeak IPA, language independent)", "-" * 60,
             " ".join(phoneme for phoneme in phonemes if phoneme.strip()) or "(nothing recognised)",
             "",
             "Most frequent: " + ", ".join("%s %d" % item for item in sorted(counts.items(), key=lambda item: -item[1])[:15]),
             "",
             "TIMING", "-" * 60,
             "  %8s %8s  %-6s %s" % ("start s", "end s", "phon.", "prob.")]
    for segment in segments:
        if segment["phoneme"].strip():
            lines.append("  %8.2f %8.2f  %-6s %.2f" % (segment["start"], segment["end"], segment["phoneme"], segment["probability"]))
    lines += ["", "FILES", "-" * 60] + ["  " + os.path.basename(path) for path in files]
    return "\n".join(lines) + "\n"
