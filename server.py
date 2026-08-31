import io
import uuid
from pathlib import Path
from typing import Optional

import httpx
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app import config, voice_store
from app.engine import adapter_manager, get_gpu_semaphore, run_whisper_sync, tts_model
from app.schemas import TTSRequest
from app.voices_api import router as voices_router

app = FastAPI(title="OmniVoice TTS + STT API")
app.include_router(voices_router)


# Helpers
async def get_audio_from_url(url: str) -> Path:
    file_name = url.split("/")[-1]
    file_path = config.CACHE_DIR / file_name
    if file_path.exists():
        return file_path
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        await run_in_threadpool(file_path.write_bytes, response.content)
    return file_path


# Rotas
@app.get("/")
async def get_docs():
    return JSONResponse(content={
        "rotas": {
            "POST /tts": "Gera áudio a partir de texto (zero-shot via ref_audio/ref_text, ou fine-tuned via voice_id)",
            "POST /transcribe": "Transcreve áudio para texto (Whisper)",
            "POST /voices": "Cria uma nova voz (para fine-tuning)",
            "POST /voices/{voice_id}/audio": "Envia áudios + transcrições para o dataset de uma voz",
            "POST /voices/{voice_id}/train": "Inicia o treinamento LoRA de uma voz",
            "GET /voices/{voice_id}": "Consulta status/progresso de uma voz",
            "GET /voices": "Lista todas as vozes",
            "DELETE /voices/{voice_id}": "Remove uma voz",
            "POST /voices/{voice_id}/evaluate": "Avalia a qualidade/similaridade de uma voz treinada",
        },
        "exemplo_tts_zero_shot": {
            "text": "Olá, esta é uma voz gerada via API.",
            "instruct": "male, portuguese accent, low pitch",
            "preset": "balanced",
            "speed": 1.0,
        },
        "exemplo_tts_voz_treinada": {
            "text": "Olá, esta é a voz do João.",
            "voice_id": "joao",
            "preset": "fast",
        },
        "presets_disponiveis": list(config.TTS_PRESETS.keys()),
        "exemplo_transcribe": "Envie um arquivo de áudio via multipart/form-data no campo 'audio'.",
        "comandos_disponiveis_tradução": [
            {"portugues": pt, "tag_obrigatoria_ingles": en}
            for pt, en in config.TRADUCOES_PT_EN.items()
        ],
    })


@app.post("/tts")
async def generate_tts(request: TTSRequest):
    try:
        if request.voice_id and request.ref_audio_url:
            raise HTTPException(
                status_code=400,
                detail="Use 'voice_id' (voz treinada) OU 'ref_audio_url'/'ref_text' (zero-shot), não os dois.",
            )

        if request.preset not in config.TTS_PRESETS:
            raise HTTPException(
                status_code=400,
                detail={"erro": "preset inválido", "validos": list(config.TTS_PRESETS.keys())},
            )
        preset_cfg = config.TTS_PRESETS[request.preset]
        num_step = request.steps if request.steps is not None else preset_cfg["num_step"]
        guidance_scale = request.guidance if request.guidance is not None else preset_cfg["guidance_scale"]

        if not request.voice_id:
            tags_enviadas = [t.strip().lower() for t in request.instruct.split(",")] if request.instruct else []
            tags_invalidas = [t for t in tags_enviadas if t not in config.VALID_INSTRUCTS]
            if tags_invalidas:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "erro": "Tags não suportadas encontradas",
                        "invalidas": tags_invalidas,
                        "ajuda": "Consulte GET / para ver as tags válidas em inglês.",
                    },
                )

        kwargs = {
            "text": request.text,
            "speed": request.speed,
            "duration": request.duration,
            "denoise": True,
            "preprocess_prompt": True,
            "postprocess_output": True,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
        }

        voice_id = request.voice_id
        if voice_id:
            meta = _get_trained_voice_or_404(voice_id)
            adapter_checkpoint = (meta.get("training") or {}).get("active_checkpoint", "last")
            adapter_path = voice_store.adapter_dir(voice_id, adapter_checkpoint)
            kwargs["instruct"] = None
        else:
            kwargs["instruct"] = request.instruct
            if request.ref_audio_url:
                ref_path = await get_audio_from_url(request.ref_audio_url)
                kwargs["ref_audio"] = str(ref_path)
                if request.ref_text:
                    kwargs["ref_text"] = request.ref_text

        print(kwargs)

        semaphore = get_gpu_semaphore()
        async with semaphore:
            if voice_id:
                await run_in_threadpool(adapter_manager.activate, voice_id, adapter_path)
            else:
                await run_in_threadpool(adapter_manager.deactivate)
            output_results = await run_in_threadpool(tts_model.generate, **kwargs)

        output_audio = output_results[0] if isinstance(output_results, list) else output_results

        audio_data = (
            output_audio.detach().cpu().numpy().squeeze()
            if hasattr(output_audio, "cpu")
            else output_audio.squeeze()
        )

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format="WAV")
        buffer.seek(0)
        headers = {"X-Preset": request.preset}
        if voice_id:
            headers["X-Voice-Id"] = voice_id
        return Response(content=buffer.read(), media_type="audio/wav", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro Crítico TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_trained_voice_or_404(voice_id: str) -> dict:
    try:
        meta = voice_store.read_meta(voice_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Voz '{voice_id}' não encontrada")
    if meta["status"] != voice_store.STATUS_TRAINED:
        raise HTTPException(
            status_code=409,
            detail=f"Voz '{voice_id}' ainda não está treinada (status atual: {meta['status']})",
        )
    return meta


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(..., description="Arquivo de áudio (wav, mp3, ogg, flac, m4a…)"),
    language: Optional[str] = Form(default=None, description="Idioma do áudio (ex: 'pt', 'en'). Deixe vazio para detecção automática."),
    task: str = Form(default="transcribe", description="'transcribe' ou 'translate'"),
):
    tmp_path = None
    try:
        audio_bytes = await audio.read()
        ext = Path(audio.filename).suffix if audio.filename else ".tmp"
        tmp_path = config.CACHE_DIR / f"{uuid.uuid4()}{ext}"

        await run_in_threadpool(tmp_path.write_bytes, audio_bytes)

        semaphore = get_gpu_semaphore()
        async with semaphore:
            texto_completo, segmentos_list, info = await run_in_threadpool(
                run_whisper_sync, str(tmp_path), language, task
            )

        return JSONResponse(content={
            "texto": " ".join(texto_completo),
            "idioma_detectado": info.language,
            "probabilidade_idioma": round(info.language_probability, 4),
            "duracao_segundos": round(info.duration, 2),
            "segmentos": segmentos_list,
        })

    except Exception as e:
        print(f"Erro Crítico STT: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and tmp_path.exists():
            await run_in_threadpool(tmp_path.unlink, missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
