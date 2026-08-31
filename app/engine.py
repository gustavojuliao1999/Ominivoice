"""Carregamento compartilhado dos modelos (OmniVoice + Whisper).

Centralizado aqui para que `server.py` (FastAPI) e `handler.py` (RunPod
serverless) usem exatamente a mesma instância de modelo e o mesmo gerente de
adapters de voz, em vez de duplicar a lógica de carregamento.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from omnivoice import OmniVoice

from app import config

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_device = config.WHISPER_DEVICE or device
compute_type = config.WHISPER_COMPUTE_TYPE or ("float16" if whisper_device == "cuda" else "int8")

print("⏳ Carregando OmniVoice...")
print(device)
tts_model = OmniVoice.from_pretrained(
    config.BASE_MODEL_ID,
    device_map=f"{device}:0" if device == "cuda" else device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)

print("⏳ Carregando Whisper", config.WHISPER_MODEL_SIZE, "...")
print(whisper_device, compute_type)
whisper_model = WhisperModel(
    config.WHISPER_MODEL_SIZE,
    device=whisper_device,
    compute_type=compute_type,
    download_root=str(config.MODELS_DIR / "whisper"),
)
print("✅ Modelos prontos!")

# O semáforo é criado sob demanda (lazy) porque `asyncio.Semaphore()` deve
# ser instanciado dentro do event loop que efetivamente vai usá-lo — em
# alguns runtimes (ex.: RunPod serverless) o loop só existe depois do import
# deste módulo.
_gpu_semaphore: asyncio.Semaphore | None = None


def get_gpu_semaphore() -> asyncio.Semaphore:
    global _gpu_semaphore
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(1)
    return _gpu_semaphore


from app.adapters import VoiceAdapterManager  # noqa: E402

adapter_manager = VoiceAdapterManager(tts_model)


def run_whisper_sync(tmp_path: str, language: Optional[str], task: str):
    """Transcreve um arquivo de áudio com o Whisper já carregado. Bloqueante
    — chame via `asyncio.to_thread`/`run_in_threadpool` a partir de código
    assíncrono."""
    segments, info = whisper_model.transcribe(
        tmp_path,
        language=language or None,
        task=task,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segmentos_list = []
    texto_completo = []
    for seg in segments:
        texto_completo.append(seg.text.strip())
        segmentos_list.append({
            "inicio": round(seg.start, 2),
            "fim": round(seg.end, 2),
            "texto": seg.text.strip(),
        })

    return texto_completo, segmentos_list, info


def transcribe_array(audio: np.ndarray, sr: int, language: Optional[str] = None, task: str = "transcribe") -> str:
    """Transcreve um array numpy em memória, via um arquivo temporário
    (mais confiável que passar arrays direto ao faster-whisper). Bloqueante."""
    tmp_path = config.CACHE_DIR / f"_eval_{uuid.uuid4().hex}.wav"
    try:
        sf.write(str(tmp_path), audio, sr, format="WAV")
        texto_completo, _, _ = run_whisper_sync(str(tmp_path), language, task)
        return " ".join(texto_completo)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
