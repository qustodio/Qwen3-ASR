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

The forced aligner is loaded the same way as ``Qwen3ASRModel.LLM(..., forced_aligner=...,
forced_aligner_kwargs=...)`` (see Hugging Face model card); this server still runs ASR inside
vLLM and applies alignment in the OpenAI transcription hook afterward.

The forced aligner defaults to **CPU** so it does not compete with vLLM for VRAM.
To run it on GPU (when you have spare VRAM or a second GPU), either:

- Pass ``--aligner-device cuda:0`` (or ``cuda:1``, etc.) before other ``vllm serve`` args, or
- Set environment variable ``QWEN_ASR_ALIGNER_DEVICE`` (same values; ``cpu`` forces CPU).

Concurrent requests: forced alignment no longer serializes globally by default. To cap how
many aligner forwards run at once (e.g. to limit RAM on CPU or VRAM on GPU), set
``QWEN_ASR_ALIGNER_MAX_CONCURRENT`` to an integer (``1`` restores strict serialization).
Unset or ``0`` means no limit. The limit is read on the **first** ``verbose_json`` transcription
in each process (set the variable before starting the server). With multiple API/engine worker
processes, each process applies its own cap, so total concurrency is roughly
``N * number_of_workers``.

Optional **batched** forced alignment (same process): set ``QWEN_ASR_ALIGN_BATCH_MAX`` to an
integer ``>=2`` to merge up to that many concurrent ``align()`` calls into one HF forward (adds
up to ``QWEN_ASR_ALIGN_BATCH_WAIT_MS`` milliseconds latency per flush when the batch is not full).
``0`` or ``1`` disables batching (legacy one-sample ``align()`` per request).
"""
from __future__ import annotations

import asyncio
import contextvars
import io
import os
import re
import sys
import time
import unicodedata
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from qwen_asr import Qwen3ASRModel
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

# Under ``vllm.*`` so logs use vLLM's console handler (``qwen_asr.*`` would not).
LOGGER = init_logger("vllm.qwen_asr.serve_aligner")

DEFAULT_FORCED_ALIGNER_CHECKPOINT = "Qwen/Qwen3-ForcedAligner-0.6B"


def _pop_aligner_device_cli(argv: List[str]) -> List[str]:
    """
    Remove ``--aligner-device <value>`` from argv and set ``QWEN_ASR_ALIGNER_DEVICE``.
    vLLM does not understand this flag.
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--aligner-device":
            if i + 1 >= len(argv):
                print(
                    "qwen-asr-serve-aligner: --aligner-device requires a value "
                    "(e.g. cuda:0, cuda:1, cpu)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            os.environ["QWEN_ASR_ALIGNER_DEVICE"] = argv[i + 1]
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _default_forced_aligner_kwargs() -> Dict[str, Any]:
    """
    Default ``forced_aligner_kwargs`` (same keys as ``Qwen3ASRModel.LLM`` / model card).

    Default CPU avoids CUDA OOM alongside a GPU-resident vLLM ASR engine.
    Override with ``QWEN_ASR_ALIGNER_DEVICE`` or ``--aligner-device`` (see module docstring).
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
_HOOK_FORCED_ALIGNER: Optional[str] = None
_HOOK_FORCED_ALIGNER_KWARGS: Optional[Dict[str, Any]] = None
_ALIGNER = None
_ALIGNER_INIT_LOCK = threading.Lock()
# None = unlimited concurrent align() calls; Lock(1) or Semaphore(N) when configured.
_ALIGNER_INFER_LIMITER: Optional[Any] = None
_ALIGNER_INFER_LIMITER_CONFIGURED = False
_ALIGNER_LIMITER_CFG_LOCK = threading.Lock()
_SENTENCE_END_CHARS = (".", "!", "?", "。", "！", "？", ";", "；")

_ALIGNED_ENGINE_PATCH_IDS: set[int] = set()
_ALIGNED_ENGINE_PATCH_LOCK = threading.Lock()
_ALIGNED_VERBOSE_MODELS: tuple[type, type] | None = None
_ALIGNED_CAPTURE_LOCK = threading.Lock()


class _AlignedTokenTotals:
    __slots__ = ("prompt_tokens", "completion_tokens", "encoder_prompt_tokens")

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.encoder_prompt_tokens = 0


# Fallback when request_id must be used (ContextVar not visible in the yielding task).
_ALIGNED_CAPTURE_BY_PREFIX: Dict[str, _AlignedTokenTotals] = {}
# Primary: active aligned-transcription token bucket for this asyncio Task.
_ALIGNED_TOKEN_TOTALS_CTX: contextvars.ContextVar[Optional[_AlignedTokenTotals]] = contextvars.ContextVar(
    "qwen_asr_aligned_token_totals", default=None
)


def _accum_aligned_tokens(output: Any, totals: _AlignedTokenTotals) -> None:
    """
    Merge counts from one engine yield. Do not require ``finished`` — vLLM may omit it on
    partial yields.

    Decoder prompt lengths are treated as the running maximum (typically stable). Completion
    counts may be cumulative across yields (monotonic) or reset per chunk (sum increments).
    """
    pt_ids = getattr(output, "prompt_token_ids", None) or []
    if pt_ids:
        totals.prompt_tokens = max(totals.prompt_tokens, len(pt_ids))
    enc_ids = getattr(output, "encoder_prompt_token_ids", None) or []
    if enc_ids:
        totals.encoder_prompt_tokens = max(totals.encoder_prompt_tokens, len(enc_ids))
    outs = getattr(output, "outputs", None) or []
    if outs:
        try:
            ct = len(outs[0].token_ids)
        except TypeError:
            ct = 0
        if ct >= totals.completion_tokens:
            totals.completion_tokens = ct
        else:
            totals.completion_tokens += ct


def _aligner_token_capture_begin(engine_client: Any, request_id_prefix: str) -> None:
    _ensure_engine_generate_aligned_capture(engine_client)
    bucket = _AlignedTokenTotals()
    _ALIGNED_TOKEN_TOTALS_CTX.set(bucket)
    with _ALIGNED_CAPTURE_LOCK:
        _ALIGNED_CAPTURE_BY_PREFIX[request_id_prefix] = bucket


def _aligner_token_capture_end(request_id_prefix: str) -> _AlignedTokenTotals:
    prev = _ALIGNED_TOKEN_TOTALS_CTX.get(None)
    _ALIGNED_TOKEN_TOTALS_CTX.set(None)
    with _ALIGNED_CAPTURE_LOCK:
        _ALIGNED_CAPTURE_BY_PREFIX.pop(request_id_prefix, None)
    return prev or _AlignedTokenTotals()


def _ensure_engine_generate_aligned_capture(engine_client: Any) -> None:
    cid = id(engine_client)
    with _ALIGNED_ENGINE_PATCH_LOCK:
        if cid in _ALIGNED_ENGINE_PATCH_IDS:
            return
        original = engine_client.generate

        async def _generate_with_optional_token_capture(prompt: Any, sampling_params: Any, request_id: str, **kwargs: Any):
            async for output in original(prompt, sampling_params, request_id, **kwargs):
                totals = _ALIGNED_TOKEN_TOTALS_CTX.get(None)
                if totals is None:
                    with _ALIGNED_CAPTURE_LOCK:
                        for cap_prefix, bucket in _ALIGNED_CAPTURE_BY_PREFIX.items():
                            if request_id == cap_prefix or request_id.startswith(cap_prefix + "_"):
                                totals = bucket
                                break
                if totals is not None:
                    _accum_aligned_tokens(output, totals)
                yield output

        setattr(engine_client, "generate", _generate_with_optional_token_capture)
        _ALIGNED_ENGINE_PATCH_IDS.add(cid)


def _ensure_aligned_verbose_models(transcription_verbose_cls: type[Any]) -> tuple[type, type]:
    global _ALIGNED_VERBOSE_MODELS
    if _ALIGNED_VERBOSE_MODELS is not None:
        return _ALIGNED_VERBOSE_MODELS
    from pydantic import BaseModel

    class AlignedVerboseTranscriptionTokens(BaseModel):

        total: int
        by_type: Dict[str, int]

    class AlignedTranscriptionResponseVerbose(transcription_verbose_cls):  # type: ignore[misc]
        tokens: AlignedVerboseTranscriptionTokens

        def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            data = super().model_dump(*args, **kwargs)
            data.pop("words", None)
            segs = data.get("segments")
            if isinstance(segs, list):
                data["segments"] = [
                    {"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in segs
                ]
            return data

    _ALIGNED_VERBOSE_MODELS = (
        AlignedVerboseTranscriptionTokens,
        AlignedTranscriptionResponseVerbose,
    )
    return _ALIGNED_VERBOSE_MODELS


def _configure_aligner_infer_limiter() -> None:
    """
    Read ``QWEN_ASR_ALIGNER_MAX_CONCURRENT`` and set ``_ALIGNER_INFER_LIMITER``.

    - unset / empty / ``0``: no limiter (parallel aligner inference).
    - ``1``: global lock (legacy serialized behavior).
    - ``N>1``: at most N aligner forwards at a time.
    """
    global _ALIGNER_INFER_LIMITER
    raw_env = os.environ.get("QWEN_ASR_ALIGNER_MAX_CONCURRENT")
    raw = (raw_env or "0").strip().lower()
    if raw in ("", "0", "unlimited", "none"):
        _ALIGNER_INFER_LIMITER = None
        LOGGER.info(
            "Forced-aligner inference concurrency: unlimited (QWEN_ASR_ALIGNER_MAX_CONCURRENT=%r)",
            raw_env,
        )
        return
    try:
        n = int(raw)
    except ValueError:
        LOGGER.warning(
            "Ignoring invalid QWEN_ASR_ALIGNER_MAX_CONCURRENT=%r; using unlimited aligner concurrency",
            raw_env,
        )
        _ALIGNER_INFER_LIMITER = None
        LOGGER.info(
            "Forced-aligner inference concurrency: unlimited (QWEN_ASR_ALIGNER_MAX_CONCURRENT=%r)",
            raw_env,
        )
        return
    if n <= 0:
        _ALIGNER_INFER_LIMITER = None
        LOGGER.info(
            "Forced-aligner inference concurrency: unlimited (QWEN_ASR_ALIGNER_MAX_CONCURRENT=%r)",
            raw_env,
        )
        return
    if n == 1:
        _ALIGNER_INFER_LIMITER = threading.Lock()
        LOGGER.info(
            "Forced-aligner inference concurrency: serialized (1) (QWEN_ASR_ALIGNER_MAX_CONCURRENT=%r)",
            raw_env,
        )
        return
    _ALIGNER_INFER_LIMITER = threading.Semaphore(n)
    LOGGER.info(
        "Forced-aligner inference concurrency: max %d concurrent (QWEN_ASR_ALIGNER_MAX_CONCURRENT=%r)",
        n,
        raw_env,
    )


def _ensure_aligner_infer_limiter() -> None:
    """Apply ``QWEN_ASR_ALIGNER_MAX_CONCURRENT`` once per process, before the first aligned transcribe."""
    global _ALIGNER_INFER_LIMITER_CONFIGURED
    if _ALIGNER_INFER_LIMITER_CONFIGURED:
        return
    with _ALIGNER_LIMITER_CFG_LOCK:
        if _ALIGNER_INFER_LIMITER_CONFIGURED:
            return
        _configure_aligner_infer_limiter()
        _ALIGNER_INFER_LIMITER_CONFIGURED = True


# Micro-batch concurrent align() forwards (see module docstring: QWEN_ASR_ALIGN_BATCH_*).
_ALIGN_BATCH_COORD: Optional["_AlignBatchCoordinator"] = None
_ALIGN_BATCH_COORD_LOCK: Optional[asyncio.Lock] = None


def _align_batch_max_from_env() -> int:
    try:
        return int((os.environ.get("QWEN_ASR_ALIGN_BATCH_MAX") or "0").strip())
    except ValueError:
        return 0


def _align_batch_wait_s_from_env() -> float:
    try:
        ms = float((os.environ.get("QWEN_ASR_ALIGN_BATCH_WAIT_MS") or "8").strip())
    except ValueError:
        ms = 8.0
    return max(0.0, ms) / 1000.0


class _AlignBatchCoordinator:
    """
    Queue concurrent alignment requests and run ``Qwen3ForcedAligner.align`` on lists
    (one GPU/CPU forward per batch). Preserves per-request result order.
    """

    __slots__ = ("_max_batch", "_wait_s", "_lock", "_pending", "_drain_task")

    def __init__(self, max_batch: int, wait_s: float) -> None:
        self._max_batch = max(2, int(max_batch))
        self._wait_s = float(wait_s)
        self._lock = asyncio.Lock()
        self._pending: list[tuple[Any, str, str, asyncio.Future]] = []
        self._drain_task: Optional[asyncio.Task] = None

    async def _delayed_drain(self) -> None:
        try:
            if self._wait_s > 0:
                await asyncio.sleep(self._wait_s)
        except asyncio.CancelledError:
            return
        batch: Optional[list[tuple[Any, str, str, asyncio.Future]]] = None
        async with self._lock:
            self._drain_task = None
            if self._pending:
                batch = self._pending
                self._pending = []
        if batch:
            await self._execute_batch(batch)

    async def _execute_batch(self, batch: list[tuple[Any, str, str, asyncio.Future]]) -> None:
        from qwen_asr.inference.utils import SAMPLE_RATE

        def _thread_fn() -> list[Any]:
            aligner = _get_aligner(_HOOK_FORCED_ALIGNER, _HOOK_FORCED_ALIGNER_KWARGS or {})
            audios = [(wav, SAMPLE_RATE) for wav, _, _, _ in batch]
            texts = [t for _, t, _, _ in batch]
            langs = [lang for _, _, lang, _ in batch]
            lim = _ALIGNER_INFER_LIMITER
            if lim is not None:
                with lim:
                    return list(aligner.align(audio=audios, text=texts, language=langs))
            return list(aligner.align(audio=audios, text=texts, language=langs))

        try:
            results = await asyncio.to_thread(_thread_fn)
        except Exception as e:
            for *_, fut in batch:
                if not fut.done():
                    fut.set_exception(e)
            return
        if len(results) != len(batch):
            err = RuntimeError(
                f"aligner batch size mismatch: got {len(results)} results for {len(batch)} inputs"
            )
            for *_, fut in batch:
                if not fut.done():
                    fut.set_exception(err)
            return
        for (_, _, _, fut), res in zip(batch, results):
            if not fut.done():
                fut.set_result(res)

    async def submit(self, wav: Any, plain_text: str, align_lang: str) -> Any:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        batch_to_run: Optional[list[tuple[Any, str, str, asyncio.Future]]] = None
        async with self._lock:
            self._pending.append((wav, plain_text, align_lang, fut))
            if len(self._pending) >= self._max_batch:
                batch_to_run = self._pending
                self._pending = []
                if self._drain_task is not None:
                    if not self._drain_task.done():
                        self._drain_task.cancel()
                    self._drain_task = None
            elif self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._delayed_drain())
        if batch_to_run is not None:
            await self._execute_batch(batch_to_run)
        return await fut


async def _get_align_batch_coordinator() -> Optional[_AlignBatchCoordinator]:
    global _ALIGN_BATCH_COORD, _ALIGN_BATCH_COORD_LOCK
    if _align_batch_max_from_env() <= 1:
        return None
    if _ALIGN_BATCH_COORD is not None:
        return _ALIGN_BATCH_COORD
    if _ALIGN_BATCH_COORD_LOCK is None:
        _ALIGN_BATCH_COORD_LOCK = asyncio.Lock()
    async with _ALIGN_BATCH_COORD_LOCK:
        if _ALIGN_BATCH_COORD is None:
            mx = _align_batch_max_from_env()
            if mx <= 1:
                return None
            _ALIGN_BATCH_COORD = _AlignBatchCoordinator(mx, _align_batch_wait_s_from_env())
            LOGGER.info(
                "Forced-aligner batching enabled: QWEN_ASR_ALIGN_BATCH_MAX=%d QWEN_ASR_ALIGN_BATCH_WAIT_MS=%s",
                mx,
                os.environ.get("QWEN_ASR_ALIGN_BATCH_WAIT_MS", "8"),
            )
        return _ALIGN_BATCH_COORD


def _bytes_to_wav_16k_mono(audio_data: bytes) -> np.ndarray:
    """
    Decode request-body audio to a mono float32 waveform at 16 kHz.

    For **mono PCM at 16 kHz** (typical WAV), ``soundfile`` decode is essentially the only
    cost: stereo down-mix and ``librosa`` resampling are skipped automatically.
    """
    with io.BytesIO(audio_data) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    sr = int(sr)
    if sr != 16000:
        import librosa

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


def _letter_or_digit_char_count(s: str) -> int:
    """Unicode letters + digits (Cyrillic, Arabic, CJK, Latin, etc.)."""
    return sum(1 for ch in s if ch.isdigit() or unicodedata.category(ch).startswith("L"))


def _all_aligned_word_times_near_zero(words: list[Any], duration_s: float) -> bool:
    """
    Detect degenerate forced alignment: word items exist but every start/end is ~0.
    Common when the ASR hallucinates over music/noise and the aligner cannot place text.

    Opt out with env ``QWEN_ASR_KEEP_ZERO_ALIGNED_TRANSCRIPTS=1``.
    Epsilon override: ``QWEN_ASR_ALIGN_ZERO_EPS`` (default 1e-3).
    """
    if not words:
        return False
    if (os.environ.get("QWEN_ASR_KEEP_ZERO_ALIGNED_TRANSCRIPTS") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    eps = float(os.environ.get("QWEN_ASR_ALIGN_ZERO_EPS", "1e-3"))
    # Ignore pathological zero-duration requests.
    if duration_s <= 0.0:
        return False
    for w in words:
        if abs(float(w.start)) > eps or abs(float(w.end)) > eps:
            return False
    return True


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
    return _letter_or_digit_char_count(s) <= tlen_th


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
    cyrillic = len(re.findall(r"[\u0400-\u04FF\u0500-\u052F]", s))
    core = _letter_or_digit_char_count(s)
    if core <= 1:
        return True

    mismatch_th = int(os.environ.get("QWEN_ASR_SHORT_TEXT_SCRIPT_MISMATCH_TH", "6"))
    if requested_lang == "English" and latin == 0 and (cjk > 0 or cyrillic > 0) and core <= mismatch_th:
        return True
    if requested_lang in {"Chinese", "Cantonese", "Japanese", "Korean"} and latin > 0 and cjk == 0 and core <= mismatch_th:
        return True
    return False


def _get_aligner(forced_aligner: str, forced_aligner_kwargs: Dict[str, Any]):
    global _ALIGNER
    with _ALIGNER_INIT_LOCK:
        if _ALIGNER is None:
            _ALIGNER = Qwen3ASRModel.load_forced_aligner(
                forced_aligner, forced_aligner_kwargs
            )
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


# Periods inside these spans are not sentence boundaries (e.g. "Mr. Smith").
_ABBREV_DOT_PROTECT = re.compile(
    r"(?:"
    r"\bPh\.D\."
    r"|\be\.g\."
    r"|\bi\.e\."
    r"|\bU\.S\."
    r"|\bU\.K\."
    r"|\b(?:Mrs|Ms|Mr|Dr|Prof|Sr|Jr|St|vs|etc|al|ed|vol|no|fig|approx)\."
    r"|\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\."
    r")",
    re.IGNORECASE,
)
_DOT_SENTINEL_START = "\ue000"
_DOT_SENTINEL_END = "\ue001"


def _sentence_units_from_text(text: str) -> list[str]:
    """
    Split transcript text into sentence-like units while preserving punctuation.

    Does not split on periods that belong to common abbreviations (e.g. ``Mr.``)
    or decimal numbers (e.g. ``3.14``).
    """
    s = (text or "").strip()
    if not s:
        return []
    vault: list[str] = []

    def stash(match: re.Match[str]) -> str:
        vault.append(match.group(0))
        return f"{_DOT_SENTINEL_START}{len(vault) - 1}{_DOT_SENTINEL_END}"

    s = re.sub(r"\d+\.\d+", stash, s)
    s = _ABBREV_DOT_PROTECT.sub(stash, s)
    parts = re.findall(r"[^.!?。！？;；]+[.!?。！？;；]*", s)

    def unstash(m: re.Match[str]) -> str:
        return vault[int(m.group(1))]

    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(rf"{re.escape(_DOT_SENTINEL_START)}(\d+){re.escape(_DOT_SENTINEL_END)}", unstash, p)
        if p.strip():
            out.append(p.strip())
    return out


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
    # Unicode word runs (Cyrillic, Latin, CJK, etc.); matches default str semantics in Python 3.
    return len(re.findall(r"\w+", text))


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


def _install_transcription_aligner_hook(
    forced_aligner: str, forced_aligner_kwargs: Dict[str, Any]
) -> None:
    global _ORIG_CREATE_SPEECH_TO_TEXT, _HOOK_FORCED_ALIGNER, _HOOK_FORCED_ALIGNER_KWARGS

    _HOOK_FORCED_ALIGNER = forced_aligner
    _HOOK_FORCED_ALIGNER_KWARGS = dict(forced_aligner_kwargs)

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

    TokensCls, VerboseCls = _ensure_aligned_verbose_models(TranscriptionResponseVerbose)

    def _tokens_from_totals(totals: _AlignedTokenTotals) -> Any:
        by_type = {
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "encoder_prompt_tokens": totals.encoder_prompt_tokens,
        }
        total = totals.prompt_tokens + totals.completion_tokens + totals.encoder_prompt_tokens
        return TokensCls(total=total, by_type=dict(by_type))

    async def _wrapped(self, audio_data: bytes, request, raw_request, response_class, stream_generator_method):
        def _log_transcription_timing(
            asr_s: float, aligner_s: Optional[float] = None, *, detail: str = ""
        ) -> None:
            suffix = f" {detail}" if detail else ""
            if aligner_s is None:
                LOGGER.info(
                    "Transcription timing: ASR model=%.3fs, aligner=skipped%s",
                    asr_s,
                    suffix,
                )
            else:
                LOGGER.info(
                    "Transcription timing: ASR model=%.3fs, aligner=%.3fs%s",
                    asr_s,
                    aligner_s,
                    suffix,
                )

        want_align = (
            _HOOK_FORCED_ALIGNER is not None
            and self.task_type == "transcribe"
            and _is_qwen3_asr_handler(self)
            and request.response_format == "verbose_json"
        )
        if not want_align:
            return await orig(self, audio_data, request, raw_request, response_class, stream_generator_method)

        _ensure_aligner_infer_limiter()

        request_json = request.model_copy(update={"response_format": "json"})
        t_asr_start = time.perf_counter()
        asr_rid_prefix = f"{self.task_type}-{self._base_request_id(raw_request)}"
        _aligner_token_capture_begin(self.engine_client, asr_rid_prefix)
        try:
            base = await orig(
                self,
                audio_data,
                request_json,
                raw_request,
                TranscriptionResponse,
                stream_generator_method,
            )
        finally:
            asr_tokens_totals = _aligner_token_capture_end(asr_rid_prefix)

        tokens_usage = _tokens_from_totals(asr_tokens_totals)
        asr_s = time.perf_counter() - t_asr_start
        if isinstance(base, ErrorResponse):
            _log_transcription_timing(asr_s, detail="(ASR error response)")
            return base

        # Decode/resample audio in a worker thread while we parse text and (on first use) load the
        # aligner, so those CPU/IO-bound steps overlap instead of running strictly sequentially.
        wav_task = asyncio.create_task(asyncio.to_thread(_bytes_to_wav_16k_mono, audio_data))
        try:
            user_lang_name = None
            if request.language:
                user_lang_name = self.model_cls.supported_languages.get(request.language)

            # Parse raw model output without forcing user language, so metadata can be stripped robustly.
            lang, plain_text = parse_asr_output(base.text, user_language=None)
            plain_text = _sanitize_asr_text(plain_text)
            align_lang = user_lang_name or lang or "English"

            aligner = _get_aligner(_HOOK_FORCED_ALIGNER, _HOOK_FORCED_ALIGNER_KWARGS or {})

            wav = await wav_task
        finally:
            if not wav_task.done():
                await wav_task

        duration_s = float(len(wav)) / 16000.0
        if _is_low_energy_audio(wav):
            LOGGER.info("Low-energy audio detected; suppressing transcript")
            _log_transcription_timing(asr_s, detail="(low-energy audio)")
            return VerboseCls(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                tokens=tokens_usage,
                words=None,
            )
        if not plain_text.strip():
            _log_transcription_timing(asr_s, detail="(empty transcript)")
            return VerboseCls(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                tokens=tokens_usage,
                words=None,
            )
        if _looks_suspicious_short_text(plain_text, user_lang_name):
            LOGGER.info(
                "Suppressing suspicious short/script-mismatch transcript: %r (requested_lang=%r)",
                plain_text,
                user_lang_name,
            )
            _log_transcription_timing(asr_s, detail="(suspicious short/script-mismatch)")
            return VerboseCls(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                tokens=tokens_usage,
                words=None,
            )
        if _should_suppress_hallucinated_text(plain_text, wav):
            LOGGER.info("Suppressing likely hallucinated short transcript on low-energy audio: %r", plain_text)
            _log_transcription_timing(asr_s, detail="(hallucination guard)")
            return VerboseCls(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                tokens=tokens_usage,
                words=None,
            )

        def _do_align_single() -> Any:
            from qwen_asr.inference.utils import SAMPLE_RATE

            lim = _ALIGNER_INFER_LIMITER
            if lim is not None:
                with lim:
                    out = aligner.align(audio=(wav, SAMPLE_RATE), text=plain_text, language=align_lang)
            else:
                out = aligner.align(audio=(wav, SAMPLE_RATE), text=plain_text, language=align_lang)
            return out[0] if out else None

        t_align_start = time.perf_counter()
        try:
            batch_coord = await _get_align_batch_coordinator()
            if batch_coord is not None:
                align_result = await batch_coord.submit(wav, plain_text, align_lang)
            else:
                align_result = await asyncio.to_thread(_do_align_single)
        except Exception:
            aligner_s = time.perf_counter() - t_align_start
            _log_transcription_timing(asr_s, aligner_s, detail="(aligner raised)")
            LOGGER.exception("Qwen3-ForcedAligner failed during /v1/audio/transcriptions")
            return self.create_error_response("Forced alignment failed; check server logs.")
        aligner_s = time.perf_counter() - t_align_start

        if align_result is None or not align_result.items:
            _log_transcription_timing(asr_s, aligner_s, detail="(no alignment items)")
            return VerboseCls(
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
                tokens=tokens_usage,
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
        if _all_aligned_word_times_near_zero(words, duration_s):
            LOGGER.info(
                "Suppressing transcript: %d aligned words but all timestamps ~0 (duration=%.3fs)",
                len(words),
                duration_s,
            )
            _log_transcription_timing(asr_s, aligner_s, detail="(degenerate timestamps)")
            return VerboseCls(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                tokens=tokens_usage,
                words=None,
            )
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
        _log_transcription_timing(asr_s, aligner_s)
        return VerboseCls(
            text=plain_text,
            language=request.language or lang or "",
            duration=str(duration_s),
            segments=segments,
            tokens=tokens_usage,
            words=None,
        )

    OpenAISpeechToText._create_speech_to_text = _wrapped


def main():
    sys.argv[1:] = _pop_aligner_device_cli(sys.argv[1:])
    forced_aligner_kwargs = _default_forced_aligner_kwargs()
    LOGGER.info(
        "Loading forced aligner %s with forced_aligner_kwargs=%s "
        "(same as Qwen3ASRModel.LLM; override device via --aligner-device or "
        "QWEN_ASR_ALIGNER_DEVICE)",
        DEFAULT_FORCED_ALIGNER_CHECKPOINT,
        forced_aligner_kwargs,
    )
    _install_transcription_aligner_hook(
        DEFAULT_FORCED_ALIGNER_CHECKPOINT, forced_aligner_kwargs
    )
    sys.argv.insert(1, "serve")
    vllm_main()


if __name__ == "__main__":
    main()
