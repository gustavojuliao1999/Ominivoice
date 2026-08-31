"""Lógica de negócio de gerenciamento de vozes, independente de transporte.

Usada tanto por `app/voices_api.py` (FastAPI, multipart/HTTP) quanto por
`handler.py` (RunPod serverless, base64/JSON), para não duplicar as regras
de ingestão de dataset e avaliação entre os dois entrypoints.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf

from app import audio_preprocessing, config, engine, evaluation, trainer, voice_store


def save_bytes_to_tmp(raw: bytes, suffix: str) -> Path:
    tmp_path = config.CACHE_DIR / f"_upload_{uuid.uuid4().hex}{suffix or '.wav'}"
    tmp_path.write_bytes(raw)
    return tmp_path


def append_clip(voice_id: str, audio, sr: int, text: str) -> dict:
    filename = f"{uuid.uuid4().hex}.wav"
    dest = voice_store.clips_dir(voice_id) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), audio.T if audio.ndim > 1 else audio, sr, format="WAV")
    duration = audio.shape[-1] / sr
    return {"audio": f"clips/{filename}", "text": text.strip(), "duration": round(duration, 3)}


def ingest_paired_clip(voice_id: str, tmp_path: Path, text: str) -> tuple[Optional[dict], Optional[str]]:
    """Processa um único clipe curto (áudio + transcrição já fornecida).
    Retorna (entrada_do_manifest, None) em sucesso, ou (None, motivo) em rejeição.
    Bloqueante — chame via thread pool.
    """
    text_ok, text_reason = audio_preprocessing.validate_transcript(text)
    if not text_ok:
        return None, text_reason

    sampling_rate = engine.tts_model.sampling_rate
    result = audio_preprocessing.preprocess_clip(str(tmp_path), sampling_rate)
    if not result.ok:
        return None, result.reason
    return append_clip(voice_id, result.audio, sampling_rate, text), None


def ingest_longform_file(voice_id: str, tmp_path: Path) -> tuple[list[dict], list[str]]:
    """Segmenta e transcreve automaticamente um áudio longo. Bloqueante — o
    chamador deve garantir exclusividade de GPU (semáforo) e rodar em thread
    pool, pois usa o Whisper internamente.
    """
    sampling_rate = engine.tts_model.sampling_rate

    def transcribe_fn(clip_audio, sr):
        return engine.transcribe_array(clip_audio, sr)

    clip_results = audio_preprocessing.segment_longform(str(tmp_path), sampling_rate, transcribe_fn)
    accepted, rejected = [], []
    for result in clip_results:
        if not result.ok:
            rejected.append(result.reason)
            continue
        accepted.append(append_clip(voice_id, result.audio, sampling_rate, result.text))
    return accepted, rejected


async def finalize_upload(voice_id: str, accepted_entries: list[dict]) -> dict:
    if accepted_entries:
        voice_store.append_manifest_entries(voice_id, accepted_entries)

    manifest = voice_store.read_manifest(voice_id)
    total_duration = sum(e["duration"] for e in manifest)

    def mutate(meta):
        meta["dataset"] = {"clips": len(manifest), "total_duration_sec": round(total_duration, 1)}
        if len(manifest) > 0 and meta["status"] == voice_store.STATUS_CREATED:
            meta["status"] = voice_store.STATUS_READY

    updated = await voice_store.update_meta(voice_id, mutate)

    warnings = []
    if total_duration < config.RECOMMENDED_TOTAL_DATASET_SEC:
        warnings.append(
            f"Dataset tem {total_duration / 60:.1f} min; o recomendado é ao menos "
            f"{config.RECOMMENDED_TOTAL_DATASET_SEC / 60:.0f} min para melhores resultados."
        )
    return {"dataset": updated["dataset"], "warnings": warnings}


async def delete_voice_full(voice_id: str) -> None:
    semaphore = engine.get_gpu_semaphore()
    async with semaphore:
        engine.adapter_manager.unload(voice_id)
    await voice_store.delete_voice(voice_id)


async def evaluate_voice(voice_id: str, texts: list[str]) -> dict:
    meta = voice_store.read_meta(voice_id)
    if meta["status"] != voice_store.STATUS_TRAINED:
        raise ValueError(f"Voz '{voice_id}' ainda não foi treinada com sucesso")
    if trainer.is_training_busy():
        raise RuntimeError("Aguarde o treinamento em andamento terminar antes de avaliar")
    if not texts:
        raise ValueError("Nenhuma frase disponível para avaliação")

    ref_audio, ref_sr = trainer.load_reference_clip(voice_id)
    active_checkpoint = (meta.get("training") or {}).get("active_checkpoint", "last")
    adapter_path = voice_store.adapter_dir(voice_id, active_checkpoint)

    semaphore = engine.get_gpu_semaphore()
    results = []
    for text in texts:
        def gen_and_eval(text=text):
            engine.adapter_manager.activate(voice_id, adapter_path)
            audio = trainer.generate_eval_sample(text)
            sr = engine.tts_model.sampling_rate
            sim = evaluation.speaker_similarity(audio, sr, ref_audio, ref_sr)
            quality = evaluation.audio_quality_stats(audio, sr)
            transcribed = engine.transcribe_array(audio, sr)
            return sim, quality, transcribed

        async with semaphore:
            sim, quality, transcribed = await asyncio.to_thread(gen_and_eval)

        error_rates = evaluation.transcription_error_rates(text, transcribed)
        results.append({
            "text": text,
            "transcribed": transcribed,
            "speaker_similarity": round(sim, 4),
            **quality,
            **error_rates,
        })

    async with semaphore:
        engine.adapter_manager.deactivate()

    summary = {
        "num_samples": len(results),
        "avg_speaker_similarity": round(sum(r["speaker_similarity"] for r in results) / len(results), 4),
        "avg_wer": (
            round(sum(r["wer"] for r in results if r["wer"] is not None) / len(results), 4)
            if results else None
        ),
        "samples": results,
    }

    def persist(meta):
        meta["evaluation"] = summary

    await voice_store.update_meta(voice_id, persist)
    return summary
