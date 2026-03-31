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

        lang, plain_text = parse_asr_output(base.text, user_language=user_lang_name)
        align_lang = lang or user_lang_name or "English"

        wav = _bytes_to_wav_16k_mono(audio_data)
        duration_s = float(len(wav)) / 16000.0
        if not plain_text.strip():
            return TranscriptionResponseVerbose(
                text="",
                language=request.language or lang or "",
                duration=str(duration_s),
                segments=[],
                words=None,
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
        segment = TranscriptionSegment(
            id=0,
            seek=0,
            start=float(words[0].start),
            end=float(words[-1].end),
            temperature=request.temperature,
            text=plain_text,
            tokens=[],
        )
        return TranscriptionResponseVerbose(
            text=plain_text,
            language=request.language or lang or "",
            duration=str(duration_s),
            segments=[segment],
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
