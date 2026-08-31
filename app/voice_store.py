"""Registry de vozes: metadados persistidos em voices/<voice_id>/meta.json.

Cada voz vive em seu próprio diretório, com o dataset bruto/processado, o
adapter LoRA treinado e um meta.json com status/progresso. O acesso é
serializado por um asyncio.Lock por processo (o servidor roda como um único
worker, então isso é suficiente para evitar corrupção de escrita
concorrente).
"""

import asyncio
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import config

VOICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")

STATUS_CREATED = "created"
STATUS_READY = "ready_for_training"
STATUS_TRAINING = "training"
STATUS_TRAINED = "trained"
STATUS_FAILED = "failed"

_locks: dict[str, asyncio.Lock] = {}


def _lock_for(voice_id: str) -> asyncio.Lock:
    if voice_id not in _locks:
        _locks[voice_id] = asyncio.Lock()
    return _locks[voice_id]


def validate_voice_id(voice_id: str) -> None:
    if not VOICE_ID_RE.match(voice_id):
        raise ValueError(
            "voice_id inválido: use 2-64 caracteres [a-z0-9_-], começando "
            "com letra/número (ex.: 'joao', 'maria_2')."
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def voice_dir(voice_id: str) -> Path:
    return config.VOICES_DIR / voice_id


def meta_path(voice_id: str) -> Path:
    return voice_dir(voice_id) / "meta.json"


def dataset_dir(voice_id: str) -> Path:
    return voice_dir(voice_id) / "dataset"


def clips_dir(voice_id: str) -> Path:
    return dataset_dir(voice_id) / "clips"


def manifest_path(voice_id: str) -> Path:
    return dataset_dir(voice_id) / "manifest.jsonl"


def adapter_dir(voice_id: str, checkpoint: str = "last") -> Path:
    return voice_dir(voice_id) / "adapter" / checkpoint


def exists(voice_id: str) -> bool:
    return meta_path(voice_id).exists()


def _default_meta(voice_id: str, name: Optional[str], language: Optional[str], description: Optional[str]) -> dict:
    return {
        "voice_id": voice_id,
        "name": name or voice_id,
        "language": language,
        "description": description or "",
        "status": STATUS_CREATED,
        "created_at": _now(),
        "updated_at": _now(),
        "dataset": {"clips": 0, "total_duration_sec": 0.0},
        "training": None,
        "evaluation": None,
        "error": None,
    }


def read_meta(voice_id: str) -> dict:
    path = meta_path(voice_id)
    if not path.exists():
        raise FileNotFoundError(f"Voz '{voice_id}' não encontrada")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_meta(voice_id: str, meta: dict) -> None:
    meta["updated_at"] = _now()
    path = meta_path(voice_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


async def create_voice(voice_id: str, name: Optional[str], language: Optional[str], description: Optional[str]) -> dict:
    validate_voice_id(voice_id)
    async with _lock_for(voice_id):
        if exists(voice_id):
            raise FileExistsError(f"Voz '{voice_id}' já existe")
        clips_dir(voice_id).mkdir(parents=True, exist_ok=True)
        (voice_dir(voice_id) / "dataset" / "raw").mkdir(parents=True, exist_ok=True)
        meta = _default_meta(voice_id, name, language, description)
        _write_meta(voice_id, meta)
        return meta


async def update_meta(voice_id: str, mutate) -> dict:
    """Aplica `mutate(meta) -> None` de forma atômica e persiste o resultado."""
    async with _lock_for(voice_id):
        meta = read_meta(voice_id)
        mutate(meta)
        _write_meta(voice_id, meta)
        return meta


def list_voices() -> list[dict]:
    if not config.VOICES_DIR.exists():
        return []
    voices = []
    for entry in sorted(config.VOICES_DIR.iterdir()):
        mp = entry / "meta.json"
        if mp.exists():
            try:
                voices.append(json.loads(mp.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return voices


async def delete_voice(voice_id: str) -> None:
    async with _lock_for(voice_id):
        if not exists(voice_id):
            raise FileNotFoundError(f"Voz '{voice_id}' não encontrada")
        shutil.rmtree(voice_dir(voice_id), ignore_errors=True)
    _locks.pop(voice_id, None)


def append_manifest_entries(voice_id: str, entries: list[dict[str, Any]]) -> None:
    path = manifest_path(voice_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_manifest(voice_id: str) -> list[dict[str, Any]]:
    path = manifest_path(voice_id)
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
