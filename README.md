# OmniVoice TTS & STT

[![Runpod](https://api.runpod.io/badge/gustavojuliao1999/Ominivoice)](https://console.runpod.io/hub/gustavojuliao1999/Ominivoice)

Text-to-speech and speech-to-text API powered by [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) and [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3).

## Endpoints

### `POST /tts` — Text to Speech

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | string | `"tts"` | Must be `"tts"` (RunPod handler only) |
| `text` | string | required | Text to synthesize |
| `voice_id` | string | `null` | Use a fine-tuned voice (see **Voice training** below). Mutually exclusive with `ref_audio_url`/`ref_text`. |
| `instruct` | string | `"female, portuguese accent"` | Comma-separated voice tags (see below). Ignored when `voice_id` is set. |
| `preset` | string | `"balanced"` | `"fast"` \| `"balanced"` \| `"quality"` — trades off speed vs. quality (see below) |
| `guidance` | float | preset default | Classifier-free guidance scale. Overrides the preset. |
| `steps` | int | preset default | Diffusion decoding steps. Overrides the preset. |
| `speed` | float | `1.0` | Playback speed |
| `duration` | float | `null` | Target audio duration in seconds |
| `ref_audio_url` (server) / `ref_audio_base64` (RunPod) | string | `null` | Reference audio for zero-shot voice cloning |
| `ref_text` | string | `""` | Transcript of the reference audio |

**Response:** WAV bytes (FastAPI) or `{ "audio_base64": "...", "format": "wav", "sample_rate": 24000 }` (RunPod).

#### Quality/speed presets

| Preset | Steps | Guidance | Use case |
|---|---|---|---|
| `fast` | 16 | 2.0 | Low latency, draft quality |
| `balanced` | 32 | 3.0 | Default, matches the original API behavior |
| `quality` | 64 | 4.0 | Slower, higher fidelity |

`steps`/`guidance` always override the chosen preset when explicitly provided.

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

## Voice training (LoRA fine-tuning)

Beyond zero-shot cloning (`ref_audio` + `ref_text`), you can fine-tune a persistent voice with your
own dataset. Training uses [LoRA](https://arxiv.org/abs/2106.09685) adapters on the internal LLM
(via [`peft`](https://github.com/huggingface/peft)) instead of full fine-tuning, so it fits on a
single **RTX 3060** (or similar 8–12 GB consumer GPU) and each trained voice is only a few MB —
the base model stays shared and resident in VRAM, and voices are swapped in/out instantly by
activating their LoRA adapter.

Each voice lives under `voices/<voice_id>/` (`dataset/`, `adapter/`, `meta.json`). Mount this
directory as a volume (`VOICES_DIR` env var) to persist trained voices across container restarts.

### Dataset requirements

- Clean audio + correct transcripts. ~30 minutes to a few hours of speech from a single speaker.
- Two upload modes (see `POST /voices/{voice_id}/audio` below):
  - **`paired`** (recommended): many short clips, each with its own transcript — most accurate.
  - **`longform`**: one or more long recordings; the server automatically segments on silence and
    transcribes each chunk with the already-loaded Whisper model.
- On upload, every clip is validated and preprocessed automatically: resampled to the model's
  native sample rate, loudness-normalized, silence-trimmed, checked for clipping/invalid/silent
  audio, and (for `paired` mode) checked for empty/implausible transcripts. Rejected clips are
  reported back instead of failing the whole batch.

### Endpoints

| Method & path | Description |
|---|---|
| `POST /voices` | Create a voice: `{"voice_id": "joao", "name": "João", "language": "pt"}` |
| `POST /voices/{voice_id}/audio` | Upload dataset audio (multipart: `mode`, `files[]`, `texts` JSON array) |
| `POST /voices/{voice_id}/train` | Start LoRA training in the background (returns immediately) |
| `GET /voices/{voice_id}` | Status, dataset stats, and live training progress (poll this) |
| `GET /voices` | List all voices |
| `DELETE /voices/{voice_id}?force=true` | Delete a voice (force cancels an in-progress training) |
| `POST /voices/{voice_id}/evaluate` | Generate sample sentences and score speaker similarity + WER |

Example workflow:

```bash
curl -X POST localhost:8000/voices -H 'content-type: application/json' \
  -d '{"voice_id": "joao", "name": "João"}'

curl -X POST localhost:8000/voices/joao/audio \
  -F mode=paired \
  -F files=@clip001.wav -F files=@clip002.wav \
  -F 'texts=["Primeira frase do João.", "Segunda frase do João."]'

curl -X POST localhost:8000/voices/joao/train -H 'content-type: application/json' -d '{}'

curl localhost:8000/voices/joao   # poll training.progress until status == "trained"

curl -X POST localhost:8000/tts -H 'content-type: application/json' \
  -d '{"text": "Olá, aqui é o João falando.", "voice_id": "joao", "preset": "fast"}' \
  --output out.wav
```

Training defaults scale with dataset size (steps ≈ clips × 40, clamped to [200, 3000]) and can be
overridden in the `/train` body (`steps`, `learning_rate`, `batch_size`,
`gradient_accumulation_steps`, `eval_every`, `r`, `lora_alpha`, `lora_dropout`). Every `eval_every`
steps, a checkpoint is scored by speaker-similarity against the dataset and the best-scoring
checkpoint (not just the last) is kept as the voice's active adapter — this protects against
overfitting drifting away from the original speaker's identity.

GPU access is shared: the RTX 3060 runs one operation at a time (`/tts`, `/transcribe`, and each
training step all queue on the same semaphore), so `/tts` requests continue to work — just
interleaved — while a voice trains in the background.

RunPod serverless (`handler.py`) exposes the same operations via `endpoint`:
`voice_create`, `voice_audio` (base64 files), `voice_train`, `voice_status`, `voice_list`,
`voice_delete`, `voice_evaluate`. Note: serverless invocations don't support out-of-band progress
polling, so `voice_train` runs synchronously by default (`"wait": true`) and only returns once
training finishes — for long training runs, prefer the FastAPI server.

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

# FastAPI server (override CMD), persisting trained voices on the host
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/voices:/workspace/voices \
  ominivoice uvicorn server:app --host 0.0.0.0 --port 8000
```

## RunPod Serverless

Set `HF_HOME=/runpod-volume/huggingface` in the template environment variables and mount a Network Volume at `/runpod-volume` (≥ 30 GB) to cache models between cold starts. To persist trained voices across workers/restarts, also set `VOICES_DIR=/runpod-volume/voices`.
