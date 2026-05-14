
import os
from pathlib import Path

# Pastas
MODELS_DIR = Path(__file__).parent / "modelos"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache_audios")
CACHE_DIR.mkdir(exist_ok=True)
HF_CACHE_DIR = MODELS_DIR / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Força o Hugging Face a usar esta pasta
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

import io
import torch
import uvicorn
import httpx
import soundfile as sf
import uuid
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import Response, JSONResponse
from starlette.concurrency import run_in_threadpool
from omnivoice import OmniVoice
from faster_whisper import WhisperModel

app = FastAPI(title="OmniVoice TTS + STT API")





# Mapeamento PT -> EN
TRADUCOES_PT_EN = {
    "sotaque americano": "american accent",
    "sotaque australiano": "australian accent",
    "sotaque britânico": "british accent",
    "sotaque canadense": "canadian accent",
    "sotaque português": "portuguese accent",
    "sotaque chinês": "chinese accent",
    "sotaque indiano": "indian accent",
    "sotaque japonês": "japanese accent",
    "sotaque coreano": "korean accent",
    "sotaque russo": "russian accent",
    "criança": "child",
    "adolescente": "teenager",
    "jovem adulto": "young adult",
    "meia-idade": "middle-aged",
    "idoso": "elderly",
    "feminino": "female",
    "masculino": "male",
    "tom agudo": "high pitch",
    "tom muito agudo": "very high pitch",
    "tom grave": "low pitch",
    "tom muito grave": "very low pitch",
    "tom moderado": "moderate pitch",
    "sussurro": "whisper",
}

VALID_INSTRUCTS = list(TRADUCOES_PT_EN.values())

# Dispositivo e Semáforo
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
gpu_semaphore = asyncio.Semaphore(1)

# Modelos
print("⏳ Carregando OmniVoice...")
print(device)
tts_model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=f"{device}:0" if device == "cuda" else device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    # cache_dir=str(MODELS_DIR / "huggingface"),
)

print("⏳ Carregando Whisper small...")
print(device, compute_type)
whisper_model = WhisperModel(
    "small",
    device=device,
    compute_type=compute_type,
    download_root=str(MODELS_DIR / "whisper"),
)
print("✅ Modelos prontos!")

# Schemas
class TTSRequest(BaseModel):
    text: str
    ref_audio_url: Optional[str] = None
    ref_text: Optional[str] = ""
    instruct: str = "female, portuguese accent"
    guidance: float = 3.0
    steps: int = 32
    speed: float = 1.0
    duration: Optional[float] = None

# Helpers
async def get_audio_from_url(url: str) -> Path:
    file_name = url.split("/")[-1]
    file_path = CACHE_DIR / file_name
    if file_path.exists():
        return file_path
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        await run_in_threadpool(file_path.write_bytes, response.content)
    return file_path

def run_whisper_sync(tmp_path: str, language: Optional[str], task: str):
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

# Rotas
@app.get("/")
async def get_docs():
    return JSONResponse(content={
        "rotas": {
            "POST /tts": "Gera áudio a partir de texto",
            "POST /transcribe": "Transcreve áudio para texto (Whisper small)",
        },
        "exemplo_tts": {
            "text": "Olá, esta é uma voz gerada via API.",
            "instruct": "male, portuguese accent, low pitch",
            "speed": 1.0,
        },
        "exemplo_transcribe": "Envie um arquivo de áudio via multipart/form-data no campo 'audio'.",
        "comandos_disponiveis_tradução": [
            {"portugues": pt, "tag_obrigatoria_ingles": en}
            for pt, en in TRADUCOES_PT_EN.items()
        ],
    })

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    try:
        tags_enviadas = [t.strip().lower() for t in request.instruct.split(",")]
        tags_invalidas = [t for t in tags_enviadas if t not in VALID_INSTRUCTS]

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
            "inference_steps": request.steps,
            "guidance_scale": request.guidance,
            "instruct": request.instruct,
        }
        print(kwargs)

        if request.ref_audio_url:
            ref_path = await get_audio_from_url(request.ref_audio_url)
            kwargs["ref_audio"] = str(ref_path)
            if request.ref_text:
                kwargs["ref_text"] = request.ref_text

        async with gpu_semaphore:
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
        return Response(content=buffer.read(), media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro Crítico TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        tmp_path = CACHE_DIR / f"{uuid.uuid4()}{ext}"
        
        await run_in_threadpool(tmp_path.write_bytes, audio_bytes)

        async with gpu_semaphore:
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