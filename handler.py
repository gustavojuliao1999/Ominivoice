import io
import base64
import uuid
import soundfile as sf
import runpod
from pathlib import Path
from typing import Optional

from models import tts_model, whisper_model, VALID_INSTRUCTS


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

    ref_path = None
    try:
        if ref_audio_b64:
            ref_path = Path(f"/tmp/{uuid.uuid4()}.wav")
            ref_path.write_bytes(base64.b64decode(ref_audio_b64))
            kwargs["ref_audio"] = str(ref_path)
            if ref_text:
                kwargs["ref_text"] = ref_text

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
        if ref_path:
            ref_path.unlink(missing_ok=True)


def handle_transcribe(job_input: dict) -> dict:
    if whisper_model is None:
        return {"error": "Whisper desativado. Defina WHISPER_ENABLED=true para usar transcrição."}

    audio_b64 = job_input.get("audio_base64")
    if not audio_b64:
        return {"error": "Campo 'audio_base64' é obrigatório."}

    language: Optional[str] = job_input.get("language")
    task = job_input.get("task", "transcribe")
    ext = job_input.get("ext", ".wav")

    tmp_path = Path(f"/tmp/{uuid.uuid4()}{ext}")
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

    if endpoint == "tts":
        return handle_tts(job_input)
    elif endpoint == "transcribe":
        return handle_transcribe(job_input)
    else:
        return {"error": f"Endpoint desconhecido: '{endpoint}'. Use 'tts' ou 'transcribe'."}


runpod.serverless.start({"handler": handler})
