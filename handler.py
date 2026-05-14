import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "modelos"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache_audios")
CACHE_DIR.mkdir(exist_ok=True)
HF_CACHE_DIR = MODELS_DIR / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

import io
import base64
import torch
import httpx
import soundfile as sf
import uuid
import asyncio
from typing import Optional
from omnivoice import OmniVoice
from faster_whisper import WhisperModel
import runpod

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

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
gpu_semaphore = None

print("⏳ Carregando OmniVoice...")
print(device)
tts_model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=f"{device}:0" if device == "cuda" else device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
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


async def get_audio_from_url(url: str) -> Path:
    file_name = url.split("/")[-1]
    file_path = CACHE_DIR / file_name
    if file_path.exists():
        return file_path
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        await asyncio.to_thread(file_path.write_bytes, response.content)
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


def handler(job):
    global gpu_semaphore
    if gpu_semaphore is None:
        gpu_semaphore = asyncio.Semaphore(1)

    job_input = job["input"]
    endpoint = job_input.get("endpoint", "tts")

    if endpoint == "tts":
        text = job_input.get("text", "")
        instruct = job_input.get("instruct", "female, portuguese accent")
        speed = float(job_input.get("speed", 1.0))
        guidance = float(job_input.get("guidance", 3.0))
        steps = int(job_input.get("steps", 32))
        duration = job_input.get("duration", None)
        ref_audio_url = job_input.get("ref_audio_url", None)
        ref_text = job_input.get("ref_text", "")

        tags_enviadas = [t.strip().lower() for t in instruct.split(",")]
        tags_invalidas = [t for t in tags_enviadas if t not in VALID_INSTRUCTS]
        if tags_invalidas:
            return {"error": "Tags não suportadas", "invalidas": tags_invalidas}

        kwargs = {
            "text": text,
            "speed": speed,
            "duration": duration,
            "denoise": True,
            "preprocess_prompt": True,
            "postprocess_output": True,
            "inference_steps": steps,
            "guidance_scale": guidance,
            "instruct": instruct,
        }
        print(kwargs)

        if ref_audio_url:
            ref_path = await get_audio_from_url(ref_audio_url)
            kwargs["ref_audio"] = str(ref_path)
            if ref_text:
                kwargs["ref_text"] = ref_text

        async with gpu_semaphore:
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

    elif endpoint == "transcribe":
        language = job_input.get("language", None)
        task = job_input.get("task", "transcribe")
        audio_url = job_input.get("audio_url", None)
        audio_base64 = job_input.get("audio_base64", None)

        if not audio_url and not audio_base64:
            return {"error": "Forneça 'audio_url' ou 'audio_base64' no input."}

        tmp_path = CACHE_DIR / f"{uuid.uuid4()}.wav"
        try:
            if audio_url:
                audio_file = await get_audio_from_url(audio_url)
                tmp_path = audio_file
            else:
                audio_bytes = base64.b64decode(audio_base64)
                await asyncio.to_thread(tmp_path.write_bytes, audio_bytes)

            async with gpu_semaphore:
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

    else:
        return {"error": f"endpoint '{endpoint}' não reconhecido. Use 'tts' ou 'transcribe'."}


runpod.serverless.start({"handler": handler})
