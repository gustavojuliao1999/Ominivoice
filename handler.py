import io
import base64
import uuid
import torch
import soundfile as sf
import runpod
from pathlib import Path
from typing import Optional

from omnivoice import OmniVoice
from faster_whisper import WhisperModel

MODELS_DIR = Path(__file__).parent / "modelos"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache_audios")
CACHE_DIR.mkdir(exist_ok=True)

VALID_INSTRUCTS = [
    "american accent", "australian accent", "british accent", "canadian accent",
    "portuguese accent", "chinese accent", "indian accent", "japanese accent",
    "korean accent", "russian accent",
    "child", "teenager", "young adult", "middle-aged", "elderly",
    "female", "male",
    "high pitch", "very high pitch", "low pitch", "very low pitch", "moderate pitch",
    "whisper",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print("⏳ Carregando OmniVoice...")
tts_model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=f"{device}:0" if device == "cuda" else device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    cache_dir=str(MODELS_DIR / "huggingface"),
)

print("⏳ Carregando Whisper large-v3...")
whisper_model = WhisperModel(
    "large-v3",
    device=device,
    compute_type=compute_type,
    download_root=str(MODELS_DIR / "whisper"),
)
print("✅ Modelos prontos!")


def handle_tts(job_input: dict) -> dict:
    text = job_input.get("text")
    if not text:
        return {"error": "Campo 'text' é obrigatório."}

    instruct = job_input.get("instruct", "female, portuguese accent")
    speed = float(job_input.get("speed", 1.0))
    guidance = float(job_input.get("guidance", 3.0))
    steps = int(job_input.get("steps", 32))
    duration = job_input.get("duration")
    ref_audio_b64 = job_input.get("ref_audio_base64")
    ref_text = job_input.get("ref_text", "")

    tags = [t.strip().lower() for t in instruct.split(",")]
    invalid = [t for t in tags if t not in VALID_INSTRUCTS]
    if invalid:
        return {"error": f"Tags inválidas: {invalid}", "validas": VALID_INSTRUCTS}

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

    if ref_audio_b64:
        ref_path = CACHE_DIR / f"{uuid.uuid4()}.wav"
        ref_path.write_bytes(base64.b64decode(ref_audio_b64))
        kwargs["ref_audio"] = str(ref_path)
        if ref_text:
            kwargs["ref_text"] = ref_text

    try:
        output = tts_model.generate(**kwargs)
        output_audio = output[0] if isinstance(output, list) else output
        audio_data = (
            output_audio.detach().cpu().numpy().squeeze()
            if hasattr(output_audio, "cpu")
            else output_audio.squeeze()
        )

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format="WAV")
        buffer.seek(0)

        return {
            "audio_base64": base64.b64encode(buffer.read()).decode("utf-8"),
            "format": "wav",
            "sample_rate": 24000,
        }
    finally:
        if ref_audio_b64 and "ref_audio" in kwargs:
            Path(kwargs["ref_audio"]).unlink(missing_ok=True)


def handle_transcribe(job_input: dict) -> dict:
    audio_b64 = job_input.get("audio_base64")
    if not audio_b64:
        return {"error": "Campo 'audio_base64' é obrigatório."}

    language: Optional[str] = job_input.get("language")
    task = job_input.get("task", "transcribe")
    ext = job_input.get("ext", ".wav")

    tmp_path = CACHE_DIR / f"{uuid.uuid4()}{ext}"
    try:
        tmp_path.write_bytes(base64.b64decode(audio_b64))

        segments, info = whisper_model.transcribe(
            str(tmp_path),
            language=language or None,
            task=task,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        texto, segs = [], []
        for seg in segments:
            texto.append(seg.text.strip())
            segs.append({
                "inicio": round(seg.start, 2),
                "fim": round(seg.end, 2),
                "texto": seg.text.strip(),
            })

        return {
            "texto": " ".join(texto),
            "idioma_detectado": info.language,
            "probabilidade_idioma": round(info.language_probability, 4),
            "duracao_segundos": round(info.duration, 2),
            "segmentos": segs,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


def handler(job):
    job_input = job.get("input", {})
    endpoint = job_input.get("endpoint", "tts")

    try:
        if endpoint == "tts":
            return handle_tts(job_input)
        elif endpoint == "transcribe":
            return handle_transcribe(job_input)
        else:
            return {"error": f"Endpoint desconhecido: '{endpoint}'. Use 'tts' ou 'transcribe'."}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
