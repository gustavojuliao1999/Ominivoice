import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = Path(__file__).parent / "modelos"
MODELS_DIR.mkdir(exist_ok=True)

# HF_HOME deve apontar para o mesmo dir antes de qualquer import do HuggingFace.
# snapshot_download usa HF_HOME — sem isso salva em ~/.cache e perde no restart.
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")

# Ativa offline se o snapshot já estiver completo — evita 30s de verificação de rede.
_snapshots = MODELS_DIR / "huggingface" / "hub" / "models--k2-fsa--OmniVoice" / "snapshots"
if _snapshots.exists() and any(_snapshots.iterdir()):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print("Cache encontrado — modo offline ativado.")

WHISPER_ENABLED = os.environ.get("WHISPER_ENABLED", "true").lower() == "true"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")

from omnivoice import OmniVoice
from faster_whisper import WhisperModel

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

print(f"⏳ Carregando OmniVoice (device={device})...")
tts_model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=f"{device}:0" if device == "cuda" else device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    cache_dir=str(MODELS_DIR / "huggingface"),
)

if WHISPER_ENABLED:
    print(f"⏳ Carregando Whisper {WHISPER_MODEL_SIZE}...")
    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        download_root=str(MODELS_DIR / "whisper"),
    )
    print("✅ Modelos prontos!")
else:
    whisper_model = None
    print("✅ OmniVoice pronto! (Whisper desativado)")
