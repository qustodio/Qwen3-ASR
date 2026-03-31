# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
vLLM ``serve`` entry with forced-aligner support for ``/v1/audio/transcriptions``
(``response_format=verbose_json``). Use console script ``qwen-asr-serve-aligner``.

The forced aligner defaults to **CPU** so it does not compete with vLLM for VRAM.
To load it on a GPU instead (e.g. if you lowered vLLM memory or use a second GPU),
set environment variable ``QWEN_ASR_ALIGNER_DEVICE`` (e.g. ``cuda:0`` or ``cuda:1``).
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import sys
import threading
from typing import Any, Dict, Optional

import librosa
import numpy as np
import soundfile as sf

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from qwen_asr.inference.utils import parse_asr_output
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry

    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
except Exception as e:
    raise ImportError(
        "vLLM is not available, to use qwen-asr-serve-aligner, please install with: pip install qwen-asr[vllm]"
    ) from e

from vllm.entrypoints.cli.main import main as vllm_main
from vllm.logger import init_logger

LOGGER = init_logger(__name__)

DEFAULT_FORCED_ALIGNER_CHECKPOINT = "Qwen/Qwen3-ForcedAligner-0.6B"


def _forced_aligner_from_pretrained_kwargs() -> Dict[str, Any]:
    """
    Kwargs for ``Qwen3ForcedAligner.from_pretrained``.

    Default CPU avoids CUDA OOM alongside a GPU-resident vLLM ASR engine.
    """
    import torch

    raw = (os.environ.get("QWEN_ASR_ALIGNER_DEVICE") or "cpu").strip()
    if not raw or raw.lower() == "cpu":
        return {"dtype": torch.float32, "device_map": "cpu"}
    dev = raw
    if dev.lower() == "cuda":
        dev = "cuda:0"
    return {"dtype": torch.bfloat16, "device_map": dev}

_ORIG_CREATE_SPEECH_TO_TEXT = None
_HOOK_ALIGNER_CKPT: Optional[str] = None
_HOOK_ALIGNER_KWARGS: Optional[Dict[str, Any]] = None
_ALIGNER = None
_ALIGNER_INIT_LOCK = threading.Lock()
_ALIGNER_INFER_LOCK = threading.Lock()
_SENTENCE_END_CHARS = (".", "!", "?", "。", "！", "？", ";", "；")


def _bytes_to_wav_16k_mono(audio_data: bytes) -> np.ndarray:
    with io.BytesIO(audio_data) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    sr = int(sr)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    return wav


def _audio_stats(wav: np.ndarray) -> Dict[str, float]:
    x = np.asarray(wav, dtype=np.float32)
    if x.size == 0:
        return {"rms": 0.0, "peak": 0.0}
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    return {"rms": rms, "peak": peak}


def _is_low_energy_audio(wav: np.ndarray) -> bool:
    """
    Cheap audio gate to suppress obvious silence/background-only inputs.
    Tunable via env vars:
      QWEN_ASR_SILENCE_RMS_TH (default: 0.008)
      QWEN_ASR_SILENCE_PEAK_TH (default: 0.05)
    """
    st = _audio_stats(wav)
    rms_th = float(os.environ.get("QWEN_ASR_SILENCE_RMS_TH", "0.008"))
    peak_th = float(os.environ.get("QWEN_ASR_SILENCE_PEAK_TH", "0.05"))
    return st["rms"] < rms_th and st["peak"] < peak_th


def _should_suppress_hallucinated_text(text: str, wav: np.ndarray) -> bool:
    """
    Suppress very short likely-hallucinated outputs on low-energy audio.
    Tunable via env var:
      QWEN_ASR_SHORT_TEXT_LEN_TH (default: 3)
    """
    s = (text or "").strip()
    if not s:
        return True
    # Only kick in for low-energy inputs to avoid harming real speech.
    if not _is_low_energy_audio(wav):
        return False
    tlen_th = int(os.environ.get("QWEN_ASR_SHORT_TEXT_LEN_TH", "3"))
    # Count letters/digits/CJK only.
    core = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", s)
    return len(core) <= tlen_th


def _looks_suspicious_short_text(text: str, requested_lang: Optional[str]) -> bool:
    """
    Text-only guard (independent of audio energy):
    - suppress ultra-short garbage-like outputs
    - suppress short script-mismatch outputs for forced language requests
    Tunable via env:
      QWEN_ASR_SHORT_TEXT_SCRIPT_MISMATCH_TH (default: 6)
    """
    s = (text or "").strip()
    if not s:
        return True
    latin = len(re.findall(r"[A-Za-z]", s))
    cjk = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", s))
    core = latin + cjk + len(re.findall(r"[0-9]", s))
    if core <= 1:
        return True

    mismatch_th = int(os.environ.get("QWEN_ASR_SHORT_TEXT_SCRIPT_MISMATCH_TH", "6"))
    if requested_lang == "English" and cjk > 0 and latin == 0 and core <= mismatch_th:
        return True
    if requested_lang in {"Chinese", "Cantonese", "Japanese", "Korean"} and latin > 0 and cjk == 0 and core <= mismatch_th:
        return True
    return False


def _get_aligner(aligner_ckpt: str, aligner_kwargs: Dict[str, Any]):
    global _ALIGNER
    with _ALIGNER_INIT_LOCK:
        if _ALIGNER is None:
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

            _ALIGNER = Qwen3ForcedAligner.from_pretrained(aligner_ckpt, **aligner_kwargs)
        return _ALIGNER


def _is_qwen3_asr_handler(handler: Any) -> bool:
    model_cls = getattr(handler, "model_cls", None)
    return getattr(model_cls, "__name__", "") == "Qwen3ASRForConditionalGeneration"


def _is_sentence_boundary(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    return any(ch in s for ch in _SENTENCE_END_CHARS)


def _segment_split_mode(request: Any) -> str:
    """
    Sentence split mode:
      - loose  : split if token contains any sentence end punctuation.
      - strict : split only if token ends with sentence end punctuation.

    Request-level override:
      request.vllm_xargs["segment_split_mode"] in {"loose","strict"}
    Global default override:
      env QWEN_ASR_SEGMENT_SPLIT_MODE in {"loose","strict"}
    """
    mode = (os.environ.get("QWEN_ASR_SEGMENT_SPLIT_MODE") or "loose").strip().lower()
    xargs = getattr(request, "vllm_xargs", None) or {}
    req_mode = str(xargs.get("segment_split_mode", "")).strip().lower()
    if req_mode in {"loose", "strict"}:
        mode = req_mode
    if mode not in {"loose", "strict"}:
        mode = "loose"
    return mode


def _join_token_text(buf: list[str]) -> str:
    # Keep CJK punctuation/characters compact while preserving spaces for non-CJK words.
    out = ""
    for tok in buf:
        t = (tok or "").strip()
        if not t:
            continue
        if not out:
            out = t
            continue
        if t in _SENTENCE_END_CHARS:
            out += t
            continue
        if len(t) == 1 and ("\u4e00" <= t <= "\u9fff"):
            out += t
            continue
        out += " " + t
    return out.strip()


def _build_sentence_segments(
    words: list[Any],
    temperature: float,
    segment_cls: Any,
    *,
    strict_boundary: bool = False,
) -> list[Any]:
    if not words:
        return []
    segments = []
    buf_text: list[str] = []
    buf_start = float(words[0].start)
    buf_end = float(words[0].end)
    seg_id = 0

    for w in words:
        wt = str(getattr(w, "word", "") or "")
        ws = float(getattr(w, "start"))
        we = float(getattr(w, "end"))
        if not buf_text:
            buf_start = ws
        buf_text.append(wt)
        buf_end = we
        if strict_boundary:
            is_boundary = wt.strip().endswith(_SENTENCE_END_CHARS)
        else:
            is_boundary = _is_sentence_boundary(wt)
        if is_boundary:
            sent_text = _join_token_text(buf_text)
            if sent_text:
                segments.append(
                    segment_cls(
                        id=seg_id,
                        seek=0,
                        start=buf_start,
                        end=buf_end,
                        temperature=temperature,
                        text=sent_text,
                        tokens=[],
                    )
                )
                seg_id += 1
            buf_text = []

    if buf_text:
        sent_text = _join_token_text(buf_text)
        if sent_text:
            segments.append(
                segment_cls(
                    id=seg_id,
                    seek=0,
                    start=buf_start,
                    end=buf_end,
                    temperature=temperature,
                    text=sent_text,
                    tokens=[],
                )
            )
    return segments


def _sentence_units_from_text(text: str) -> list[str]:
    """
    Split transcript text into sentence-like units while preserving punctuation.
    """
    s = (text or "").strip()
    if not s:
        return []
    parts = re.findall(r"[^.!?。！？;；]+[.!?。！？;；]*", s)
    return [p.strip() for p in parts if p and p.strip()]


def _sanitize_asr_text(text: str) -> str:
    """
    Remove leaked control markers like:
      - language Chinese<asr_text>
      - language Japanese <asr_text>
    which occasionally appear in model output.
    """
    s = (text or "").strip()
    if not s:
        return ""
    # If a leaked control tail starts with "language ...", drop everything from there.
    # Examples:
    #   "... normal text. language Chinese<asr_text>嗯。"
    #   "... normal text\nlanguage Japanese<asr_text>な。"
    tail_match = re.search(
        r"(?i)(?:^|[\n\r]|[.!?。！？;；]\s*)language\s+[A-Za-z][A-Za-z_-]*(?:\s*<asr_text>)?",
        s,
    )
    if tail_match and tail_match.start() > 0:
        s = s[: tail_match.start()].strip()
    # Remove inline language-prefix markers (case-insensitive).
    s = re.sub(r"language\s+[A-Za-z]+\s*<asr_text>", "", s, flags=re.IGNORECASE)
    # Remove any dangling asr tag if leaked alone.
    s = s.replace("<asr_text>", "")
    # Cleanup extra spaces left by removals.
    s = re.sub(r"\s{2,}", " ", s).strip()
    # Remove spaces before punctuation where possible.
    s = re.sub(r"\s+([.!?。！？;；,，])", r"\1", s)
    return s


def _token_count_for_alignment(text: str) -> int:
    """
    Estimate token count in plain transcript text to map onto aligned `words`.
    """
    if not text:
        return 0
    toks = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]+", text)
    return len(toks)


def _build_sentence_segments_from_text(
    transcript_text: str,
    words: list[Any],
    temperature: float,
    segment_cls: Any,
) -> list[Any]:
    """
    Build sentence segments using punctuation from transcript text and
    timestamps from aligned words.
    """
    if not words:
        return []
    units = _sentence_units_from_text(transcript_text)
    if len(units) <= 1:
        return []

    segments = []
    w_idx = 0
    for i, sent in enumerate(units):
        remaining_sent = len(units) - i
        remaining_words = len(words) - w_idx
        if remaining_words <= 0:
            break

        if i == len(units) - 1:
            take = remaining_words
        else:
            est = max(1, _token_count_for_alignment(sent))
            min_for_rest = remaining_sent - 1  # keep at least 1 word per remaining sentence
            take = min(est, max(1, remaining_words - min_for_rest))

        end_idx = min(len(words), w_idx + take)
        if end_idx <= w_idx:
            continue

        seg_start = float(words[w_idx].start)
        seg_end = float(words[end_idx - 1].end)
        segments.append(
            segment_cls(
                id=len(segments),
                seek=0,
                start=seg_start,
                end=seg_end,
                temperature=temperature,
                text=sent.strip(),
                tokens=[],
            )
        )
        w_idx = end_idx

    if not segments:
        return []
    # If tiny alignment mismatch leaves trailing words, stretch the last segment end.
    if w_idx < len(words):
        segments[-1].end = float(words[-1].end)
    return segments


def _install_transcription_aligner_hook(aligner_ckpt: str, aligner_kwargs: Dict[str, Any]) -> None:
    global _ORIG_CREATE_SPEECH_TO_TEXT, _HOOK_ALIGNER_CKPT, _HOOK_ALIGNER_KWARGS

    _HOOK_ALIGNER_CKPT = aligner_ckpt
    _HOOK_ALIGNER_KWARGS = dict(aligner_kwargs)

    from vllm.entrypoints.openai.protocol import (
        ErrorResponse,
        TranscriptionResponse,
        TranscriptionResponseVerbose,
        TranscriptionSegment,
        TranscriptionWord,
    )
    from vllm.entrypoints.openai.speech_to_text import OpenAISpeechToText

    if _ORIG_CREATE_SPEECH_TO_TEXT is None:
        _ORIG_CREATE_SPEECH_TO_TEXT = OpenAISpeechToText._create_speech_to_text

    orig = _ORIG_CREATE_SPEECH_TO_TEXT

    async def _wrapped(self, audio_data: bytes, request, raw_request, response_class, stream_generator_method):
        want_align = (
            _HOOK_ALIGNER_CKPT is not None
            and self.task_type == "transcribe"
            and _is_qwen3_asr_handler(self)
            and request.response_format == "verbose_json"
        )
        if not want_align:
            return await orig(self, audio_data, request, raw_request, response_class, stream_generator_method)

        request_json = request.model_copy(update={"response_format": "json"})
        base = await orig(
            self,
            audio_data,
            request_json,
            raw_request,
            TranscriptionResponse,
            stream_generator_method,
        )
        if isinstance(base, ErrorResponse):
            return base

        user_lang_name = None
        if request.language:
            user_lang_name = self.model_cls.supported_languages.get(request.language)

        # Parse raw model output without forcing user language, so metadata can be stripped robustly.
        lang, plain_text = parse_asr_output(base.text, user_language=None)
        plain_text = _sanitize_asr_text(plain_text)
        align_lang = user_lang_name or lang or "English"

        wav = _bytes_to_wav_16k_mono(audio_data)
        duration_s = float(len(wav)) / 16000.0
        if _is_low_energy_audio(wav):
            LOGGER.info("Low-energy audio detected; suppressing transcript")
            return TranscriptionResponseVerbose(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                words=[],
            )
        if not plain_text.strip():
            return TranscriptionResponseVerbose(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                words=None,
            )
        if _looks_suspicious_short_text(plain_text, user_lang_name):
            LOGGER.info(
                "Suppressing suspicious short/script-mismatch transcript: %r (requested_lang=%r)",
                plain_text,
                user_lang_name,
            )
            return TranscriptionResponseVerbose(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                words=[],
            )
        if _should_suppress_hallucinated_text(plain_text, wav):
            LOGGER.info("Suppressing likely hallucinated short transcript on low-energy audio: %r", plain_text)
            return TranscriptionResponseVerbose(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                words=[],
            )

        aligner = _get_aligner(_HOOK_ALIGNER_CKPT, _HOOK_ALIGNER_KWARGS or {})

        def _do_align():
            from qwen_asr.inference.utils import SAMPLE_RATE

            with _ALIGNER_INFER_LOCK:
                out = aligner.align(audio=(wav, SAMPLE_RATE), text=plain_text, language=align_lang)
            return out[0] if out else None

        try:
            align_result = await asyncio.to_thread(_do_align)
        except Exception:
            LOGGER.exception("Qwen3-ForcedAligner failed during /v1/audio/transcriptions")
            return self.create_error_response("Forced alignment failed; check server logs.")

        if align_result is None or not align_result.items:
            return TranscriptionResponseVerbose(
                text=plain_text,
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[
                    TranscriptionSegment(
                        id=0,
                        seek=0,
                        start=0.0,
                        end=duration_s,
                        temperature=request.temperature,
                        text=plain_text,
                        tokens=[],
                    )
                ],
                words=None,
            )

        words = [
            TranscriptionWord(
                word=item.text,
                start=float(item.start_time),
                end=float(item.end_time),
            )
            for item in align_result.items
        ]
        split_mode = _segment_split_mode(request)
        # Preferred: split by punctuation from transcript text, map onto word timestamps.
        segments = _build_sentence_segments_from_text(
            plain_text,
            words,
            request.temperature,
            TranscriptionSegment,
        )
        # Fallback: split from word tokens only.
        if not segments:
            segments = _build_sentence_segments(
                words,
                request.temperature,
                TranscriptionSegment,
                strict_boundary=(split_mode == "strict"),
            )
        if not segments:
            segments = [
                TranscriptionSegment(
                    id=0,
                    seek=0,
                    start=float(words[0].start),
                    end=float(words[-1].end),
                    temperature=request.temperature,
                    text=plain_text,
                    tokens=[],
                )
            ]
        return TranscriptionResponseVerbose(
            text=plain_text,
            language=request.language or lang or "",
            duration=str(duration_s),
            segments=segments,
            words=words,
        )

    OpenAISpeechToText._create_speech_to_text = _wrapped


def main():
    kw = _forced_aligner_from_pretrained_kwargs()
    LOGGER.info("Loading Qwen3-ForcedAligner with %s (set QWEN_ASR_ALIGNER_DEVICE to override)", kw)
    _install_transcription_aligner_hook(DEFAULT_FORCED_ALIGNER_CHECKPOINT, kw)
    sys.argv.insert(1, "serve")
    vllm_main()


if __name__ == "__main__":
    main()
