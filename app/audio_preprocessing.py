"""Pré-processamento e validação de áudio para datasets de treino de voz.

Reaproveita os utilitários de áudio já usados pelo `omnivoice` (remoção de
silêncio, corte de trechos longos, carregamento com resample) em vez de
reimplementar essa lógica.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from omnivoice.utils.audio import (
    audiosegment_to_numpy,
    load_audio,
    numpy_to_audiosegment,
    remove_silence,
)
from pydub.silence import detect_nonsilent

from app import config

logger = logging.getLogger(__name__)


@dataclass
class ClipResult:
    audio: Optional[np.ndarray] = None  # (1, T) float32 @ target sr
    duration_sec: float = 0.0
    ok: bool = True
    reason: Optional[str] = None
    text: Optional[str] = None  # preenchido no modo longform (transcrição automática)
    warnings: list[str] = field(default_factory=list)


def validate_transcript(text: str) -> tuple[bool, Optional[str]]:
    text = (text or "").strip()
    if not text:
        return False, "transcrição vazia"
    if len(text) < 2:
        return False, "transcrição muito curta"
    if len(text) > 2000:
        return False, "transcrição excede 2000 caracteres (divida o áudio em clipes menores)"
    return True, None


def _peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-9:
        return -120.0
    return 20.0 * np.log10(peak)


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms <= 1e-9:
        return -120.0
    return 20.0 * np.log10(rms)


def normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """Normaliza o áudio para um RMS alvo, respeitando o teto de pico (anti-clip)."""
    if audio.size == 0:
        return audio
    rms_db = _rms_dbfs(audio)
    gain_db = config.TARGET_RMS_DBFS - rms_db
    gain = 10.0 ** (gain_db / 20.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    max_gain_for_peak = (10.0 ** (config.TARGET_PEAK_DBFS / 20.0)) / peak if peak > 1e-9 else gain
    gain = min(gain, max_gain_for_peak)

    return (audio * gain).astype(np.float32)


def validate_audio_array(audio: np.ndarray, sampling_rate: int) -> tuple[bool, Optional[str]]:
    if audio.size == 0:
        return False, "áudio vazio"
    if not np.isfinite(audio).all():
        return False, "áudio contém valores inválidos (NaN/Inf)"
    duration = audio.shape[-1] / sampling_rate
    if duration < config.MIN_CLIP_DURATION_SEC:
        return False, f"áudio muito curto ({duration:.2f}s < {config.MIN_CLIP_DURATION_SEC}s)"
    if duration > config.MAX_CLIP_DURATION_SEC:
        return False, (
            f"áudio muito longo para um clipe único ({duration:.1f}s > "
            f"{config.MAX_CLIP_DURATION_SEC}s) — use o modo 'longform' para "
            "segmentação automática"
        )
    rms_db = _rms_dbfs(audio)
    if rms_db < -55.0:
        return False, "áudio parece ser silêncio (nível muito baixo)"
    clipped_ratio = float(np.mean(np.abs(audio) >= 0.999))
    if clipped_ratio > 0.01:
        return False, f"áudio contém clipping excessivo ({clipped_ratio * 100:.1f}% das amostras)"
    return True, None


def preprocess_clip(path: str, sampling_rate: int) -> ClipResult:
    """Carrega, resample, remove silêncio de borda e normaliza um clipe curto."""
    try:
        audio = load_audio(path, sampling_rate)
    except Exception as exc:  # arquivo corrompido/formato inválido
        return ClipResult(ok=False, reason=f"falha ao ler áudio: {exc}")

    ok, reason = validate_audio_array(audio, sampling_rate)
    if not ok:
        return ClipResult(ok=False, reason=reason)

    audio = remove_silence(audio, sampling_rate, mid_sil=250, lead_sil=80, trail_sil=150)
    if audio.shape[-1] == 0:
        return ClipResult(ok=False, reason="áudio ficou vazio após remoção de silêncio")

    audio = normalize_loudness(audio)

    duration = audio.shape[-1] / sampling_rate
    if duration < config.MIN_CLIP_DURATION_SEC:
        return ClipResult(ok=False, reason="áudio muito curto após remoção de silêncio")

    return ClipResult(audio=audio.astype(np.float32), duration_sec=duration)


def segment_longform(
    path: str,
    sampling_rate: int,
    transcribe_fn,
) -> list[ClipResult]:
    """Segmenta um áudio longo (30min-várias horas) em clipes curtos e os
    transcreve automaticamente com `transcribe_fn(np.ndarray, sr) -> str`.

    Usa detecção de trechos não-silenciosos e agrupa-os em blocos de
    ``LONGFORM_CHUNK_MIN_SEC``–``LONGFORM_CHUNK_MAX_SEC`` segundos.
    """
    audio = load_audio(path, sampling_rate)
    ok, reason = validate_audio_array_longform(audio, sampling_rate)
    if not ok:
        return [ClipResult(ok=False, reason=reason)]

    segment = numpy_to_audiosegment(audio, sampling_rate)
    spans = detect_nonsilent(segment, min_silence_len=350, silence_thresh=-40, seek_step=20)
    if not spans:
        return [ClipResult(ok=False, reason="nenhum trecho de fala detectado (áudio parece silencioso)")]

    max_ms = int(config.LONGFORM_CHUNK_MAX_SEC * 1000)
    min_ms = int(config.LONGFORM_CHUNK_MIN_SEC * 1000)

    chunks: list[tuple[int, int]] = []  # (start_ms, end_ms)
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if end - cur_start <= max_ms:
            cur_end = end
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    chunks.append((cur_start, cur_end))

    # Divide à força qualquer bloco que ainda exceda o máximo (fala contínua
    # sem pausas detectáveis).
    final_chunks: list[tuple[int, int]] = []
    for start, end in chunks:
        span = end - start
        if span <= max_ms:
            final_chunks.append((start, end))
            continue
        cursor = start
        while cursor < end:
            nxt = min(cursor + max_ms, end)
            final_chunks.append((cursor, nxt))
            cursor = nxt

    results: list[ClipResult] = []
    for start, end in final_chunks:
        if end - start < min_ms and (end - start) < int(config.MIN_CLIP_DURATION_SEC * 1000):
            continue
        clip_segment = segment[start:end]
        clip_audio = audiosegment_to_numpy(clip_segment)
        ok, reason = validate_audio_array(clip_audio, sampling_rate)
        if not ok:
            results.append(ClipResult(ok=False, reason=reason))
            continue
        clip_audio = normalize_loudness(clip_audio.astype(np.float32))
        try:
            text = transcribe_fn(clip_audio, sampling_rate)
        except Exception as exc:
            results.append(ClipResult(ok=False, reason=f"falha na transcrição automática: {exc}"))
            continue
        text_ok, text_reason = validate_transcript(text)
        if not text_ok:
            results.append(ClipResult(ok=False, reason=f"transcrição automática inválida: {text_reason}"))
            continue
        duration = clip_audio.shape[-1] / sampling_rate
        result = ClipResult(audio=clip_audio, duration_sec=duration, text=text)
        results.append(result)
    return results


def validate_audio_array_longform(audio: np.ndarray, sampling_rate: int) -> tuple[bool, Optional[str]]:
    if audio.size == 0:
        return False, "áudio vazio"
    if not np.isfinite(audio).all():
        return False, "áudio contém valores inválidos (NaN/Inf)"
    duration = audio.shape[-1] / sampling_rate
    if duration < config.MIN_CLIP_DURATION_SEC:
        return False, f"áudio muito curto ({duration:.2f}s)"
    return True, None
