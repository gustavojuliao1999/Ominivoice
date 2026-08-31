"""Constrói o dataset de treino (tokens de áudio + texto) a partir do
manifest de uma voz, reaproveitando o mesmo audio_tokenizer (codec) já
carregado no modelo de inferência.

Os tokens são cacheados em disco (`dataset/tokens_cache.pt`) e invalidados
automaticamente quando o manifest muda, para não recodificar tudo a cada
treino.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch

from app import voice_store

logger = logging.getLogger(__name__)


def _manifest_hash(manifest: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for entry in manifest:
        h.update(entry["audio"].encode("utf-8"))
        h.update(b"\x00")
        h.update(entry["text"].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


@torch.inference_mode()
def _encode_clip(model, wav_path: Path) -> torch.Tensor:
    """Codifica um clipe wav (já no sample rate do modelo) em tokens de
    áudio multi-codebook, shape (C, T). Espelha `OmniVoice.create_voice_clone_prompt`.
    """
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    wav = data.T  # (T, C) -> (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(axis=0, keepdims=True)

    chunk_size = model.audio_tokenizer.config.hop_length
    clip_size = int(wav.shape[-1] % chunk_size)
    if clip_size > 0:
        wav = wav[:, :-clip_size]
    if wav.shape[-1] < chunk_size:
        raise ValueError(f"clipe curto demais para o codec de áudio: {wav_path}")

    wav_tensor = torch.from_numpy(wav).to(model.audio_tokenizer.device)
    tokens = model.audio_tokenizer.encode(wav_tensor.unsqueeze(0)).audio_codes.squeeze(0)
    return tokens.cpu()


def build_or_load_samples(voice_id: str, model) -> list[dict[str, Any]]:
    """Retorna uma lista de samples crus prontos para o
    `OmniVoiceSampleProcessor`: ``{"audio_tokens": Tensor[C,T], "label": {"text": str}}``.
    """
    manifest = voice_store.read_manifest(voice_id)
    if not manifest:
        raise ValueError(f"Voz '{voice_id}' não tem clipes de áudio no dataset")

    current_hash = _manifest_hash(manifest)
    cache_path = voice_store.dataset_dir(voice_id) / "tokens_cache.pt"

    if cache_path.exists():
        try:
            cached = torch.load(cache_path, weights_only=False)
            if cached.get("manifest_hash") == current_hash:
                logger.info("Usando cache de tokens de áudio para voz '%s'", voice_id)
                return cached["samples"]
        except Exception:
            logger.warning("Falha ao ler cache de tokens de '%s', recodificando", voice_id)

    logger.info("Codificando %d clipes de áudio para a voz '%s'...", len(manifest), voice_id)
    samples = []
    base_dir = voice_store.dataset_dir(voice_id)
    for entry in manifest:
        wav_path = base_dir / entry["audio"]
        try:
            tokens = _encode_clip(model, wav_path)
        except Exception as exc:
            logger.warning("Pulando clipe '%s': %s", wav_path, exc)
            continue
        samples.append({"audio_tokens": tokens, "label": {"text": entry["text"]}})

    if not samples:
        raise ValueError(f"Nenhum clipe válido pôde ser codificado para a voz '{voice_id}'")

    torch.save({"manifest_hash": current_hash, "samples": samples}, cache_path)
    return samples
