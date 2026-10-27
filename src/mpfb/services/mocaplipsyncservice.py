# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         26.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  lipsync curves from the phonemes recognised in audio
# ------------------------------------------------------------------------------
"""Turns output of the phoneme model (*_logits.npz) into curves of the visemes02 shape keys.

The model gives IPA tokens of any language. Each token is reduced to its base symbol (tones, length marks and diacritics removed)
and mapped to a viseme, diphthongs give two visemes. Every phoneme opens its viseme a little before its sound,
neighbouring visemes overlap and short pauses inside words are bridged so the mouth does not snap shut between phonemes.
"""

import os
import unicodedata
from typing import Any

import numpy as np

from .logservice import LogService

_LOG = LogService.get_logger("services.mocaplipsyncservice")

PP, FF, TH, DD, KK, CH, SS, NN, RR = "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD", "viseme_kk", "viseme_CH", \
    "viseme_SS", "viseme_nn", "viseme_RR"
AA, E, I, O, U = "viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U"

# viseme_sil is the rest pose (its target is empty) - silence means no viseme
LIPSYNC_VISEMES = [PP, FF, TH, DD, KK, CH, SS, NN, RR, AA, E, I, O, U]

# Mapping of Base IPA symbols to viseme. Symbols mapped to None (h, glottal stop) have no mouth shape so mouth keeps the shape of neighbouring phonemes.
_SYMBOLS = {
    "p": PP, "b": PP, "m": PP, "ɓ": PP,
    "f": FF, "v": FF, "ɸ": FF, "β": FF, "ʋ": FF, "ɱ": FF,
    "θ": TH, "ð": TH,
    "t": DD, "d": DD, "ʈ": DD, "ɖ": DD, "c": DD, "ɟ": DD, "ɗ": DD,
    "k": KK, "ɡ": KK, "g": KK, "q": KK, "ŋ": KK, "x": KK, "ɣ": KK, "χ": KK, "ɴ": KK, "N": KK, "X": KK,
    "ʃ": CH, "ʒ": CH, "ɕ": CH, "ʑ": CH, "ʂ": CH, "ʐ": CH, "ç": CH, "ʝ": CH, "S": CH,
    "s": SS, "z": SS,
    "n": NN, "l": NN, "ɲ": NN, "ɳ": NN, "ɭ": NN, "ʎ": NN, "ɫ": NN, "ɬ": NN, "ɮ": NN,
    "r": RR, "ɾ": RR, "ɹ": RR, "ɻ": RR, "ɽ": RR, "ʁ": RR, "ɚ": RR, "ʀ": RR,
    "a": AA, "ɑ": AA, "æ": AA, "ɐ": AA, "ä": AA, "ɒ": AA, "ʌ": AA,
    "e": E, "ɛ": E, "ə": E, "ɜ": E, "ɘ": E, "ɤ": E,
    "i": I, "ɪ": I, "ɨ": I, "ᵻ": I, "ɯ": I, "j": I,
    "o": O, "ɔ": O, "ɵ": O, "œ": O, "ø": O, "ɞ": O,
    "u": U, "ʊ": U, "y": U, "ʉ": U, "w": U, "ɥ": U, "ʍ": U, "ʏ": U,
    "h": None, "ɦ": None, "ʔ": None, "ħ": None, "ʕ": None,
    }
# Multi symbol consonants checked before the single symbols
_CLUSTERS = {"tʃ": CH, "dʒ": CH, "tɕ": CH, "dʑ": CH, "ʈʂ": CH, "tS": CH, "dZ": CH, "ts": SS, "dz": SS, "pf": FF}
_VOWEL_VISEMES = {AA, E, I, O, U}
# Marks removed from a token before the lookup (length, tones, aspiration, palatalisation, espeak extras)
_REMOVED_MARKS = set("ːˑ:ʰʲˤᵝ^.[]\"'0123456789")

# ARKit face units which belong to speech. With a lipsync these are left out of the face animation from video,
# so the mouth does not move twice. Smile, frown, dimples, cheeks and nose stay.
SPEECH_FACEUNITS = [
    "jawOpen", "jawForward", "jawLeft", "jawRight", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft",
    "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthStretchLeft", "mouthStretchRight", "tongueOut",
    ]

# Pauses inside speech shorter than this keep the mouth shape, longer ones let it close
BRIDGE_GAP = 0.12
# Share of a diphthong spent on its first vowel
DIPHTHONG_FIRST = 0.6

DEFAULT_LIPSYNC_SETTINGS = {
    "strength": 1.0,
    "lead": 0.04,
    "transition": 0.07,
    "add_sound": True,
    "extend_scene": True,
    "include_children": True,
    }

_SPECIAL_TOKENS = {"<pad>", "<s>", "</s>", "<unk>", "|", "??"}


class MocapLipsyncService():
    """Phonemes to viseme curves."""

    def __init__(self) -> None:
        raise RuntimeError("You should not instance MocapLipsyncService. Use its static methods instead.")

    # ----- Phonemes -------

    @staticmethod
    def normalize_token(token: str) -> str:
        """Base symbols of espeak token: tones, length, aspiration, palatalisation and diacritics removed."""
        decomposed = unicodedata.normalize("NFD", token)
        return "".join(char for char in decomposed if char not in _REMOVED_MARKS and not unicodedata.combining(char))

    @staticmethod
    def visemes_for_token(token: str) -> list[str] | None:
        """Visemes of a token (two for a diphthong) in order - [] for a token without a mouth shape (h, glottal stop), None when the token is not known."""
        if token in _SPECIAL_TOKENS:
            return []
        base = MocapLipsyncService.normalize_token(token)
        if not base:
            return []
        if base in _CLUSTERS:
            return [_CLUSTERS[base]]
        for cluster, viseme in _CLUSTERS.items():
            if base.startswith(cluster):
                return [viseme]
        visemes = []
        known = False
        for char in base:
            if char not in _SYMBOLS:
                continue
            known = True
            viseme = _SYMBOLS[char]
            if viseme is not None and (not visemes or visemes[-1] != viseme):
                visemes.append(viseme)
        if not known:
            return None
        # Only vowel sequences are split (diphthongs), consonant cluster keeps only its first consonant
        if len(visemes) > 1 and not all(viseme in _VOWEL_VISEMES for viseme in visemes):
            visemes = visemes[:1]
        return visemes[:2]

    @staticmethod
    def decode_segments(logprobs: np.ndarray, tokens: list[str], blank: int, frame_seconds: float,
                        start: float) -> list[tuple[float, float, str]]:
        """Greedy CTC decoding into (start, end, token) with times in seconds of the recording."""
        best = logprobs.argmax(axis=1)
        segments = []
        index = 0
        while index < len(best):
            token = int(best[index])
            end = index
            while end + 1 < len(best) and best[end + 1] == token:
                end += 1
            if token != blank and token < len(tokens):
                segments.append((start + index * frame_seconds, start + (end + 1) * frame_seconds, tokens[token]))
            index = end + 1
        return segments

    @staticmethod
    def viseme_events(segments: list[tuple[float, float, str]]) -> tuple[list[tuple[str, float, float]], list[str]]:
        """Convert decoded phoneme segments to viseme events (viseme, start, end) in seconds and list the phoneme tokens the table does not know.
        Diphthongs are split in two, short pauses between phonemes are bridged."""
        events = []
        unknown = []
        for start, end, token in segments:
            visemes = MocapLipsyncService.visemes_for_token(token)
            if visemes is None:
                if token not in unknown:
                    unknown.append(token)
                continue
            if len(visemes) == 2:
                middle = start + (end - start) * DIPHTHONG_FIRST
                events.append((visemes[0], start, middle))
                events.append((visemes[1], middle, end))
            elif visemes:
                events.append((visemes[0], start, end))
        bridged = []
        for index, (viseme, start, end) in enumerate(events):
            if index + 1 < len(events):
                following = events[index + 1][1]
                if 0 < following - end <= BRIDGE_GAP:
                    end = following
            bridged.append((viseme, start, end))
        return bridged, unknown

    # --------- Curves ----------

    @staticmethod
    def viseme_curves(events: list[tuple[str, float, float]], times: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
        """Viseme weights as an array of shape (len(times), len(LIPSYNC_VISEMES)). Each viseme starts `lead` seconds before its sound
        and fades in and out over `transition` seconds. Overlapping visemes bkend (weights are scaled down so the sum does not exceed 1)."""
        lead = float(settings["lead"])
        transition = max(float(settings["transition"]), 1e-3)
        weights = np.zeros((len(times), len(LIPSYNC_VISEMES)))
        for viseme, start, end in events:
            start -= lead
            end -= lead
            rise = np.clip((times - (start - transition)) / transition, 0.0, 1.0)
            fall = np.clip(((end + transition) - times) / transition, 0.0, 1.0)
            column = LIPSYNC_VISEMES.index(viseme)
            weights[:, column] = np.maximum(weights[:, column], np.minimum(rise, fall))
        total = weights.sum(axis=1, keepdims=True)
        weights = np.where(total > 1.0, weights / np.maximum(total, 1e-9), weights)
        return np.clip(weights * float(settings["strength"]), 0.0, 1.0)

    @staticmethod
    def load_audio_source(path: str) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            logprobs = np.asarray(data["logprobs"], dtype=np.float32)
            frame_seconds = float(data["frame_seconds"])
            tokens = [str(token) for token in data["tokens"]]
            blank = int(data["blank"]) if "blank" in data.files else 0
            centres = np.asarray(data["frame_times"], dtype=np.float64) if "frame_times" in data.files else None
        start = float(centres[0] - frame_seconds / 2) if centres is not None and len(centres) else 0.0
        return {"logprobs": logprobs, "frame_seconds": frame_seconds, "tokens": tokens, "blank": blank, "start": start,
                "end": start + len(logprobs) * frame_seconds}

    @staticmethod
    def prepare_lipsync_curves(path: str, settings: dict[str, Any], fps: float) -> dict[str, Any]:
        """Returns audio source as viseme values per frame of the scene:

            names     visemes02 shape key names
            values    (frames, names) array, 0 - 1
            frames    number of scene frames
            start     second of the recording where the first frame is
            phonemes  number of recognised phonemes
            unknown   tokens which have no viseme
        """
        settings = dict(DEFAULT_LIPSYNC_SETTINGS, **settings)
        source = MocapLipsyncService.load_audio_source(path)
        segments = MocapLipsyncService.decode_segments(source["logprobs"], source["tokens"], source["blank"],
                                                       source["frame_seconds"], source["start"])
        if not segments:
            raise ValueError("No speech was recognised in " + os.path.basename(path))
        events, unknown = MocapLipsyncService.viseme_events(segments)
        if unknown:
            _LOG.warn("Phonemes without a viseme", unknown)
        frames = int(np.floor((source["end"] - source["start"]) * fps + 1e-6)) + 1
        times = source["start"] + np.arange(frames) / fps
        values = MocapLipsyncService.viseme_curves(events, times, settings)
        return {"names": list(LIPSYNC_VISEMES), "values": values, "frames": frames, "start": source["start"],
                "phonemes": len(segments), "unknown": unknown}
