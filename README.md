# OmniVoice TTS & STT

[![Runpod](https://api.runpod.io/badge/gustavojuliao1999/Ominivoice)](https://console.runpod.io/hub/gustavojuliao1999/Ominivoice)

Text-to-speech and speech-to-text API powered by [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) and [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3).

## Endpoints

### `POST /tts` — Text to Speech

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | string | `"tts"` | Must be `"tts"` |
| `text` | string | required | Text to synthesize |
| `instruct` | string | `"female, portuguese accent"` | Comma-separated voice tags (see below) |
| `speed` | float | `1.0` | Playback speed |
| `guidance` | float | `3.0` | Classifier-free guidance scale |
| `steps` | int | `32` | Diffusion inference steps |
| `duration` | float | `null` | Target audio duration in seconds |
| `ref_audio_base64` | string | `null` | Base64-encoded reference audio for voice cloning |
| `ref_text` | string | `""` | Transcript of the reference audio |

**Response:** `{ "audio_base64": "...", "format": "wav", "sample_rate": 24000 }`

---

### `POST /transcribe` — Speech to Text

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | string | required | Must be `"transcribe"` |
| `audio_base64` | string | required | Base64-encoded audio file |
| `language` | string | `null` | Language code (e.g. `"pt"`, `"en"`). Auto-detect if omitted |
| `task` | string | `"transcribe"` | `"transcribe"` or `"translate"` |
| `ext` | string | `".wav"` | File extension hint (e.g. `".mp3"`, `".ogg"`) |

**Response:** `{ "texto": "...", "idioma_detectado": "pt", "probabilidade_idioma": 0.99, "duracao_segundos": 5.2, "segmentos": [...] }`

---

## Valid voice tags for `instruct`

**Accent:** `american accent`, `australian accent`, `british accent`, `canadian accent`, `portuguese accent`, `chinese accent`, `indian accent`, `japanese accent`, `korean accent`, `russian accent`

**Age:** `child`, `teenager`, `young adult`, `middle-aged`, `elderly`

**Gender:** `female`, `male`

**Pitch:** `high pitch`, `very high pitch`, `low pitch`, `very low pitch`, `moderate pitch`, `whisper`

Example: `"female, british accent, low pitch"`

---

## Running locally with Docker

```bash
docker build -t ominivoice .

# FastAPI server (override CMD)
docker run --gpus all -p 8000:8000 ominivoice \
  uvicorn server:app --host 0.0.0.0 --port 8000
```

## RunPod Serverless

Set `HF_HOME=/runpod-volume/huggingface` in the template environment variables and mount a Network Volume at `/runpod-volume` (≥ 30 GB) to cache models between cold starts.
