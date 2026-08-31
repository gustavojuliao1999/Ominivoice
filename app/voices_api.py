"""Rotas de gerenciamento de vozes: criação, upload de dataset, treino,
status, listagem, remoção e avaliação.

Rotas HTTP/multipart finas — a lógica de negócio vive em
`app/voice_service.py` (compartilhada com `handler.py`).
Montado em `server.py` via `app.include_router(voices_router)`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app import config, engine, trainer, voice_service, voice_store
from app.schemas import EvaluateRequest, TrainRequest, VoiceCreateRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voices", tags=["voices"])

# Mantém referências fortes às tasks de treino em background (asyncio não
# garante que uma Task sem referência sobreviva até o fim) e permite
# cancelamento best-effort via DELETE ?force=true.
_training_tasks: dict[str, asyncio.Task] = {}


def _get_meta_or_404(voice_id: str) -> dict:
    try:
        return voice_store.read_meta(voice_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Voz '{voice_id}' não encontrada")


@router.post("")
async def create_voice(req: VoiceCreateRequest):
    try:
        meta = await voice_store.create_voice(req.voice_id, req.name, req.language, req.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return meta


@router.get("")
async def list_voices():
    return {"voices": voice_store.list_voices()}


@router.get("/{voice_id}")
async def get_voice(voice_id: str):
    return _get_meta_or_404(voice_id)


@router.delete("/{voice_id}")
async def remove_voice(
    voice_id: str,
    force: bool = Query(False, description="Cancela o treino em andamento, se houver, e apaga mesmo assim"),
):
    meta = _get_meta_or_404(voice_id)
    if meta["status"] == voice_store.STATUS_TRAINING and not force:
        raise HTTPException(
            status_code=409,
            detail="Voz está em treinamento. Use ?force=true para cancelar e remover.",
        )

    task = _training_tasks.get(voice_id)
    if task and not task.done():
        task.cancel()

    await voice_service.delete_voice_full(voice_id)
    return {"status": "deleted", "voice_id": voice_id}


@router.post("/{voice_id}/audio")
async def upload_audio(
    voice_id: str,
    mode: str = Form("paired", description="'paired' (áudio+transcrição por arquivo) ou 'longform' (segmentação e transcrição automáticas)"),
    files: list[UploadFile] = File(...),
    texts: Optional[str] = Form(None, description="JSON com lista de transcrições, na mesma ordem de 'files' (obrigatório no modo 'paired')"),
):
    meta = _get_meta_or_404(voice_id)
    if meta["status"] == voice_store.STATUS_TRAINING:
        raise HTTPException(status_code=409, detail="Não é possível enviar áudio enquanto a voz está treinando")
    if mode not in ("paired", "longform"):
        raise HTTPException(status_code=400, detail="mode deve ser 'paired' ou 'longform'")

    text_list: list[str] = []
    if mode == "paired":
        if not texts:
            raise HTTPException(status_code=400, detail="Campo 'texts' (JSON com a transcrição de cada arquivo) é obrigatório no modo 'paired'")
        try:
            text_list = json.loads(texts)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="'texts' deve ser um JSON válido (lista de strings)")
        if not isinstance(text_list, list) or len(text_list) != len(files):
            raise HTTPException(status_code=400, detail="'texts' deve ter o mesmo número de itens que 'files'")

    accepted: list[dict] = []
    rejected: list[dict] = []
    semaphore = engine.get_gpu_semaphore()

    for idx, upload in enumerate(files):
        raw = await upload.read()
        suffix = Path(upload.filename or "audio").suffix or ".wav"
        tmp_path = await run_in_threadpool(voice_service.save_bytes_to_tmp, raw, suffix)
        try:
            if mode == "paired":
                entry, reason = await run_in_threadpool(
                    voice_service.ingest_paired_clip, voice_id, tmp_path, text_list[idx]
                )
                if entry is None:
                    rejected.append({"file": upload.filename, "reason": reason})
                else:
                    accepted.append(entry)
            else:  # longform
                async with semaphore:
                    new_entries, reasons = await run_in_threadpool(
                        voice_service.ingest_longform_file, voice_id, tmp_path
                    )
                accepted.extend(new_entries)
                rejected.extend({"file": upload.filename, "reason": r} for r in reasons)
        finally:
            tmp_path.unlink(missing_ok=True)

    result = await voice_service.finalize_upload(voice_id, accepted)
    return {
        "voice_id": voice_id,
        "accepted": len(accepted),
        "rejected": rejected,
        **result,
    }


@router.post("/{voice_id}/train")
async def start_training(voice_id: str, req: TrainRequest):
    meta = _get_meta_or_404(voice_id)
    if meta["status"] == voice_store.STATUS_TRAINING or trainer.is_training_busy():
        raise HTTPException(status_code=409, detail="Já existe um treinamento em andamento neste servidor")
    if meta["dataset"]["clips"] == 0:
        raise HTTPException(status_code=400, detail="Envie áudios com POST /voices/{voice_id}/audio antes de treinar")

    total_duration = meta["dataset"]["total_duration_sec"]
    if total_duration < config.MIN_TOTAL_DATASET_SEC:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset muito pequeno ({total_duration:.1f}s); mínimo de {config.MIN_TOTAL_DATASET_SEC:.0f}s para treinar",
        )

    warnings = []
    if total_duration < config.RECOMMENDED_TOTAL_DATASET_SEC:
        warnings.append(
            f"Dataset abaixo do recomendado (~{config.RECOMMENDED_TOTAL_DATASET_SEC / 60:.0f} min); "
            "a qualidade/identidade da voz pode ficar instável."
        )

    train_cfg = req.model_dump(exclude_none=True)

    task = asyncio.create_task(trainer.run_training_job(voice_id, train_cfg))
    _training_tasks[voice_id] = task
    task.add_done_callback(lambda t, vid=voice_id: _training_tasks.pop(vid, None))

    return {"status": "training_started", "voice_id": voice_id, "warnings": warnings}


@router.post("/{voice_id}/evaluate")
async def evaluate_voice(voice_id: str, req: EvaluateRequest):
    meta = _get_meta_or_404(voice_id)
    manifest = voice_store.read_manifest(voice_id)
    texts = req.sample_texts or [e["text"] for e in manifest[: req.num_samples]]
    try:
        return await voice_service.evaluate_voice(voice_id, texts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
