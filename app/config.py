import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_DIR / "modelos"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache_audios"))
VOICES_DIR = Path(os.environ.get("VOICES_DIR", str(PROJECT_DIR / "voices")))
HF_CACHE_DIR = MODELS_DIR / "huggingface"

MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)
VOICES_DIR.mkdir(exist_ok=True, parents=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

BASE_MODEL_ID = os.environ.get("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")

# Mapeamento PT -> EN das tags de instruct suportadas pelo modelo base.
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

# --- Presets de qualidade/velocidade para /tts -----------------------------
# `num_step` = passos de decodificação iterativa do diffusion LM.
# `guidance_scale` = escala de classifier-free guidance.
TTS_PRESETS = {
    "fast": {"num_step": 16, "guidance_scale": 2.0},
    "balanced": {"num_step": 32, "guidance_scale": 3.0},
    "quality": {"num_step": 64, "guidance_scale": 4.0},
}
DEFAULT_PRESET = "balanced"

# --- Defaults de LoRA / treino ---------------------------------------------
# Alvo padrão: projeções de atenção + MLP dos blocos Qwen3 usados como LLM
# interno do OmniVoice. Cobrem o suficiente de capacidade para adaptar o
# timbre/identidade do falante sem retreinar o modelo inteiro.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_DEFAULTS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": LORA_TARGET_MODULES,
    "bias": "none",
}
# Nome do adapter "neutro" (nunca treinado) usado para servir o modelo base
# (zero-shot / voice design) depois que o wrapper PEFT é instanciado.
BASE_ADAPTER_NAME = "__base__"

TRAIN_DEFAULTS = {
    "learning_rate": 2e-4,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "eval_every": 50,
    "save_every": 50,
    "seed": 42,
    # Faixa de passos default quando o usuário não especifica: escalado pelo
    # número de clipes do dataset e limitado a [min_steps, max_steps].
    "steps_per_clip": 40,
    "min_steps": 200,
    "max_steps": 3000,
}

# --- Pré-processamento de áudio --------------------------------------------
MIN_CLIP_DURATION_SEC = 0.3
MAX_CLIP_DURATION_SEC = 20.0
MIN_TOTAL_DATASET_SEC = 20.0  # piso absoluto para permitir iniciar um treino
RECOMMENDED_TOTAL_DATASET_SEC = 30 * 60  # 30 minutos, recomendado pelo produto
LONGFORM_CHUNK_MAX_SEC = 15.0
LONGFORM_CHUNK_MIN_SEC = 3.0
TARGET_PEAK_DBFS = -1.0
TARGET_RMS_DBFS = -20.0
