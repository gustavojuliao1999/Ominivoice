import asyncio
import base64
import io
import uuid

import runpod
import soundfile as sf

from app import config, voice_service, voice_store
from app.engine import adapter_manager, get_gpu_semaphore, run_whisper_sync, tts_model

TRADUCOES_PT_EN = config.TRADUCOES_PT_EN
VALID_INSTRUCTS = config.VALID_INSTRUCTS


async def get_audio_from_url(url: str):
    import httpx

    file_name = url.split("/")[-1]
    file_path = config.CACHE_DIR / file_name
    if file_path.exists():
        return file_path
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        await asyncio.to_thread(file_path.write_bytes, response.content)
    return file_path


async def _handle_tts(event_input: dict) -> dict:
    text = event_input.get("text", "")
    voice_id = event_input.get("voice_id")
    preset = event_input.get("preset", config.DEFAULT_PRESET)
    if preset not in config.TTS_PRESETS:
        return {"error": "preset inválido", "validos": list(config.TTS_PRESETS.keys())}
    preset_cfg = config.TTS_PRESETS[preset]

    speed = float(event_input.get("speed", 1.0))
    guidance = event_input.get("guidance")
    guidance_scale = float(guidance) if guidance is not None else preset_cfg["guidance_scale"]
    steps = event_input.get("steps")
    num_step = int(steps) if steps is not None else preset_cfg["num_step"]
    duration = event_input.get("duration", None)
    ref_audio_url = event_input.get("ref_audio_url", None)
    ref_text = event_input.get("ref_text", "")
    instruct = event_input.get("instruct", "female, portuguese accent")

    if voice_id and ref_audio_url:
        return {"error": "Use 'voice_id' OU 'ref_audio_url'/'ref_text', não os dois."}

    kwargs = {
        "text": text,
        "speed": speed,
        "duration": duration,
        "denoise": True,
        "preprocess_prompt": True,
        "postprocess_output": True,
        "num_step": num_step,
        "guidance_scale": guidance_scale,
    }

    adapter_path = None
    if voice_id:
        try:
            meta = voice_store.read_meta(voice_id)
        except FileNotFoundError:
            return {"error": f"Voz '{voice_id}' não encontrada"}
        if meta["status"] != voice_store.STATUS_TRAINED:
            return {"error": f"Voz '{voice_id}' ainda não está treinada (status: {meta['status']})"}
        checkpoint = (meta.get("training") or {}).get("active_checkpoint", "last")
        adapter_path = voice_store.adapter_dir(voice_id, checkpoint)
        kwargs["instruct"] = None
    else:
        tags_enviadas = [t.strip().lower() for t in instruct.split(",")] if instruct else []
        tags_invalidas = [t for t in tags_enviadas if t not in VALID_INSTRUCTS]
        if tags_invalidas:
            return {"error": "Tags não suportadas", "invalidas": tags_invalidas}
        kwargs["instruct"] = instruct
        if ref_audio_url:
            ref_path = await get_audio_from_url(ref_audio_url)
            kwargs["ref_audio"] = str(ref_path)
            if ref_text:
                kwargs["ref_text"] = ref_text

    semaphore = get_gpu_semaphore()
    async with semaphore:
        if voice_id:
            await asyncio.to_thread(adapter_manager.activate, voice_id, adapter_path)
        else:
            await asyncio.to_thread(adapter_manager.deactivate)
        output_results = await asyncio.to_thread(tts_model.generate, **kwargs)

    output_audio = output_results[0] if isinstance(output_results, list) else output_results
    audio_data = (
        output_audio.detach().cpu().numpy().squeeze()
        if hasattr(output_audio, "cpu")
        else output_audio.squeeze()
    )

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, 24000, format="WAV")
    buffer.seek(0)
    return {"audio_base64": base64.b64encode(buffer.read()).decode("utf-8")}


async def _handle_transcribe(event_input: dict) -> dict:
    language = event_input.get("language", None)
    task = event_input.get("task", "transcribe")
    audio_url = event_input.get("audio_url", None)
    audio_base64 = event_input.get("audio_base64", None)
    ext = event_input.get("ext", ".wav")

    if not audio_url and not audio_base64:
        return {"error": "Forneça 'audio_url' ou 'audio_base64' no input."}

    tmp_path = config.CACHE_DIR / f"{uuid.uuid4()}{ext}"
    try:
        if audio_url:
            tmp_path = await get_audio_from_url(audio_url)
        else:
            audio_bytes = base64.b64decode(audio_base64)
            await asyncio.to_thread(tmp_path.write_bytes, audio_bytes)

        semaphore = get_gpu_semaphore()
        async with semaphore:
            texto_completo, segmentos_list, info = await asyncio.to_thread(
                run_whisper_sync, str(tmp_path), language, task
            )

        return {
            "texto": " ".join(texto_completo),
            "idioma_detectado": info.language,
            "probabilidade_idioma": round(info.language_probability, 4),
            "duracao_segundos": round(info.duration, 2),
            "segmentos": segmentos_list,
        }
    finally:
        if audio_base64 and tmp_path.exists():
            await asyncio.to_thread(tmp_path.unlink, missing_ok=True)


async def _handle_voice_create(event_input: dict) -> dict:
    try:
        return await voice_store.create_voice(
            event_input.get("voice_id"),
            event_input.get("name"),
            event_input.get("language"),
            event_input.get("description"),
        )
    except (ValueError, FileExistsError) as exc:
        return {"error": str(exc)}


async def _handle_voice_audio(event_input: dict) -> dict:
    voice_id = event_input.get("voice_id")
    if not voice_id:
        return {"error": "voice_id é obrigatório"}
    try:
        voice_store.read_meta(voice_id)
    except FileNotFoundError:
        return {"error": f"Voz '{voice_id}' não encontrada"}

    mode = event_input.get("mode", "paired")
    files = event_input.get("files", [])
    if not files:
        return {"error": "'files' deve ser uma lista de {audio_base64, text?, ext?}"}

    semaphore = get_gpu_semaphore()
    accepted: list[dict] = []
    rejected: list[dict] = []

    for i, item in enumerate(files):
        audio_b64 = item.get("audio_base64")
        if not audio_b64:
            rejected.append({"index": i, "reason": "audio_base64 ausente"})
            continue
        raw = base64.b64decode(audio_b64)
        suffix = item.get("ext", ".wav")
        tmp_path = await asyncio.to_thread(voice_service.save_bytes_to_tmp, raw, suffix)
        try:
            if mode == "paired":
                text = item.get("text", "")
                entry, reason = await asyncio.to_thread(
                    voice_service.ingest_paired_clip, voice_id, tmp_path, text
                )
                if entry is None:
                    rejected.append({"index": i, "reason": reason})
                else:
                    accepted.append(entry)
            else:
                async with semaphore:
                    new_entries, reasons = await asyncio.to_thread(
                        voice_service.ingest_longform_file, voice_id, tmp_path
                    )
                accepted.extend(new_entries)
                rejected.extend({"index": i, "reason": r} for r in reasons)
        finally:
            tmp_path.unlink(missing_ok=True)

    result = await voice_service.finalize_upload(voice_id, accepted)
    return {"voice_id": voice_id, "accepted": len(accepted), "rejected": rejected, **result}


async def _handle_voice_train(event_input: dict) -> dict:
    from app import trainer

    voice_id = event_input.get("voice_id")
    if not voice_id:
        return {"error": "voice_id é obrigatório"}
    try:
        meta = voice_store.read_meta(voice_id)
    except FileNotFoundError:
        return {"error": f"Voz '{voice_id}' não encontrada"}
    if meta["status"] == voice_store.STATUS_TRAINING or trainer.is_training_busy():
        return {"error": "Já existe um treinamento em andamento neste worker"}
    if meta["dataset"]["clips"] == 0:
        return {"error": "Envie áudios (endpoint 'voice_audio') antes de treinar"}

    train_cfg = {
        k: event_input.get(k)
        for k in ("steps", "learning_rate", "batch_size", "gradient_accumulation_steps", "eval_every", "r", "lora_alpha", "lora_dropout")
        if event_input.get(k) is not None
    }

    # RunPod Serverless não expõe polling de progresso entre invocações de
    # forma confiável — por padrão o treino roda de forma síncrona e a
    # resposta só volta quando termina. Para datasets grandes isso pode
    # exceder o timeout do endpoint; ajuste o timeout do template do RunPod
    # ou use o servidor FastAPI (server.py) para treinos longos com polling
    # assíncrono via GET /voices/{voice_id}.
    wait = event_input.get("wait", True)
    if wait:
        try:
            await trainer.run_training_job(voice_id, train_cfg)
        except Exception as exc:
            return {"error": str(exc)}
        return voice_store.read_meta(voice_id)
    else:
        asyncio.create_task(trainer.run_training_job(voice_id, train_cfg))
        return {"status": "training_started", "voice_id": voice_id}


async def _handle_voice_status(event_input: dict) -> dict:
    voice_id = event_input.get("voice_id")
    if not voice_id:
        return {"error": "voice_id é obrigatório"}
    try:
        return voice_store.read_meta(voice_id)
    except FileNotFoundError:
        return {"error": f"Voz '{voice_id}' não encontrada"}


async def _handle_voice_list(event_input: dict) -> dict:
    return {"voices": voice_store.list_voices()}


async def _handle_voice_delete(event_input: dict) -> dict:
    voice_id = event_input.get("voice_id")
    if not voice_id:
        return {"error": "voice_id é obrigatório"}
    try:
        voice_store.read_meta(voice_id)
    except FileNotFoundError:
        return {"error": f"Voz '{voice_id}' não encontrada"}
    await voice_service.delete_voice_full(voice_id)
    return {"status": "deleted", "voice_id": voice_id}


async def _handle_voice_evaluate(event_input: dict) -> dict:
    voice_id = event_input.get("voice_id")
    if not voice_id:
        return {"error": "voice_id é obrigatório"}
    try:
        manifest = voice_store.read_manifest(voice_id)
    except FileNotFoundError:
        return {"error": f"Voz '{voice_id}' não encontrada"}
    num_samples = int(event_input.get("num_samples", 3))
    texts = event_input.get("sample_texts") or [e["text"] for e in manifest[:num_samples]]
    try:
        return await voice_service.evaluate_voice(voice_id, texts)
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}


_ENDPOINTS = {
    "tts": _handle_tts,
    "transcribe": _handle_transcribe,
    "voice_create": _handle_voice_create,
    "voice_audio": _handle_voice_audio,
    "voice_train": _handle_voice_train,
    "voice_status": _handle_voice_status,
    "voice_list": _handle_voice_list,
    "voice_delete": _handle_voice_delete,
    "voice_evaluate": _handle_voice_evaluate,
}


async def handler(event):
    event_input = event["input"]
    endpoint = event_input.get("endpoint", "tts")

    handler_fn = _ENDPOINTS.get(endpoint)
    if handler_fn is None:
        return {"error": f"endpoint '{endpoint}' não reconhecido. Use um de: {list(_ENDPOINTS)}"}
    return await handler_fn(event_input)


runpod.serverless.start({"handler": handler})
