"""Métricas de avaliação para vozes treinadas: similaridade de locutor
(timbre/identidade) e qualidade/inteligibilidade do áudio gerado.

- Similaridade de locutor: embeddings do `resemblyzer` (leve, roda em CPU),
  comparados por similaridade de cosseno contra um clipe de referência do
  próprio dataset da voz.
- Inteligibilidade: word/character error rate comparando o texto de entrada
  com a transcrição do áudio gerado, usando o Whisper que já está carregado
  no processo (`app.engine.whisper_model`) — nenhuma dependência nova.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_voice_encoder = None


def _get_voice_encoder():
    global _voice_encoder
    if _voice_encoder is None:
        from resemblyzer import VoiceEncoder

        # CPU proposital: modelo pequeno (~17MB), evita competir por VRAM
        # com o modelo principal durante checkpoints de avaliação no treino.
        _voice_encoder = VoiceEncoder(device="cpu")
    return _voice_encoder


def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return audio


def speaker_similarity(gen_audio: np.ndarray, gen_sr: int, ref_audio: np.ndarray, ref_sr: int) -> float:
    """Similaridade de cosseno entre os embeddings de locutor de dois áudios.
    Retorna um valor tipicamente entre -1 e 1 (quanto maior, mais parecido).
    """
    from resemblyzer import preprocess_wav

    encoder = _get_voice_encoder()
    gen_wav = preprocess_wav(_as_mono_float32(gen_audio), source_sr=gen_sr)
    ref_wav = preprocess_wav(_as_mono_float32(ref_audio), source_sr=ref_sr)

    if gen_wav.size == 0 or ref_wav.size == 0:
        raise ValueError("áudio vazio após pré-processamento do resemblyzer")

    emb_gen = encoder.embed_utterance(gen_wav)
    emb_ref = encoder.embed_utterance(ref_wav)

    denom = float(np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(emb_gen, emb_ref) / denom)


def audio_quality_stats(audio: np.ndarray, sr: int) -> dict:
    audio = _as_mono_float32(audio)
    duration = audio.shape[-1] / sr if sr else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak_dbfs = 20.0 * np.log10(peak) if peak > 1e-9 else -120.0
    rms_dbfs = 20.0 * np.log10(rms) if rms > 1e-9 else -120.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0
    silence_ratio = float(np.mean(np.abs(audio) < 10 ** (-45 / 20))) if audio.size else 1.0
    return {
        "duration_sec": round(duration, 3),
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_dbfs": round(rms_dbfs, 2),
        "clipping_ratio": round(clipping_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
    }


def transcription_error_rates(reference_text: str, hypothesis_text: str) -> dict:
    import jiwer

    reference_text = (reference_text or "").strip()
    hypothesis_text = (hypothesis_text or "").strip()
    if not reference_text:
        return {"wer": None, "cer": None}
    return {
        "wer": round(float(jiwer.wer(reference_text, hypothesis_text)), 4),
        "cer": round(float(jiwer.cer(reference_text, hypothesis_text)), 4),
    }
