"""Gerente de adapters LoRA por voz.

Mantém um único modelo base carregado em VRAM (`tts_model.llm`) e envolve-o
uma única vez com `peft`, registrando um adapter LoRA nomeado por
`voice_id`. Trocar de voz em runtime é apenas `set_adapter(voice_id)` — não
recarrega pesos do disco a cada chamada (o adapter fica residente depois do
primeiro uso; adapters LoRA são pequenos, poucos MB cada).

Toda mutação de estado aqui (wrap, add_adapter, set_adapter, load_adapter)
deve acontecer com o semáforo de GPU do `app.engine` já adquirido pelo
chamador — este módulo não faz locking próprio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app import config

logger = logging.getLogger(__name__)


class VoiceAdapterManager:
    def __init__(self, model):
        self.model = model
        self._peft_ready = False
        self._loaded: set[str] = set()
        self._active: Optional[str] = None

    # -- setup -----------------------------------------------------------

    def _ensure_peft(self) -> None:
        if self._peft_ready:
            return
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "O pacote 'peft' é necessário para treinar/usar vozes com "
                "fine-tuning LoRA. Instale com `pip install peft`."
            ) from exc

        lora_config = LoraConfig(task_type=None, **config.LORA_DEFAULTS)
        self.model.llm = get_peft_model(
            self.model.llm, lora_config, adapter_name=config.BASE_ADAPTER_NAME
        )
        # O adapter base nunca é treinado: fica com B=0 (inicialização padrão
        # de LoRA), ou seja, equivale exatamente ao modelo original.
        for p in self.model.llm.parameters():
            p.requires_grad_(False)
        self._peft_ready = True
        self._loaded.add(config.BASE_ADAPTER_NAME)
        self._active = config.BASE_ADAPTER_NAME
        logger.info("Modelo LLM envolvido com PEFT/LoRA (adapter base '%s')", config.BASE_ADAPTER_NAME)

    @property
    def active(self) -> Optional[str]:
        return self._active

    def is_loaded(self, voice_id: str) -> bool:
        return voice_id in self._loaded

    # -- treino ------------------------------------------------------------

    def prepare_for_training(self, voice_id: str, lora_overrides: Optional[dict] = None) -> list:
        """Garante que `voice_id` existe como adapter treinável (novo, do
        zero) e retorna a lista de parâmetros treináveis (só desse adapter).
        """
        from peft import LoraConfig

        self._ensure_peft()

        cfg = dict(config.LORA_DEFAULTS)
        if lora_overrides:
            cfg.update({k: v for k, v in lora_overrides.items() if v is not None})

        if voice_id in self.model.llm.peft_config:
            self.model.llm.delete_adapter(voice_id)
            self._loaded.discard(voice_id)

        self.model.llm.add_adapter(voice_id, LoraConfig(task_type=None, **cfg))
        self.model.llm.set_adapter(voice_id)
        self._active = voice_id
        self._loaded.add(voice_id)

        trainable = []
        for name, param in self.model.llm.named_parameters():
            is_this_adapter = f".{voice_id}." in name or name.endswith(f".{voice_id}")
            param.requires_grad_(is_this_adapter and "lora_" in name)
            if param.requires_grad:
                trainable.append(param)

        if not trainable:
            raise RuntimeError(
                f"Nenhum parâmetro treinável encontrado para o adapter '{voice_id}' "
                "— verifique se os target_modules do LoRA existem no modelo."
            )
        return trainable

    def save_adapter(self, voice_id: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.llm.save_pretrained(str(output_dir), selected_adapters=[voice_id])

    # -- inferência --------------------------------------------------------

    def load_adapter_from_disk(self, voice_id: str, adapter_path: Path) -> None:
        self._ensure_peft()
        if voice_id in self._loaded:
            return
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter da voz '{voice_id}' não encontrado em {adapter_path}")
        self.model.llm.load_adapter(str(adapter_path), adapter_name=voice_id)
        self._loaded.add(voice_id)

    def activate(self, voice_id: str, adapter_path: Optional[Path] = None) -> None:
        self._ensure_peft()
        if voice_id not in self._loaded:
            if adapter_path is None:
                raise FileNotFoundError(f"Adapter da voz '{voice_id}' não está carregado e nenhum caminho foi informado")
            self.load_adapter_from_disk(voice_id, adapter_path)
        self.model.llm.set_adapter(voice_id)
        self._active = voice_id

    def deactivate(self) -> None:
        """Volta ao modelo base (zero-shot / voice design), sem nenhum LoRA ativo."""
        if not self._peft_ready:
            return
        self.model.llm.set_adapter(config.BASE_ADAPTER_NAME)
        self._active = config.BASE_ADAPTER_NAME

    def unload(self, voice_id: str) -> None:
        if not self._peft_ready or voice_id not in self.model.llm.peft_config:
            self._loaded.discard(voice_id)
            return
        if self._active == voice_id:
            self.deactivate()
        self.model.llm.delete_adapter(voice_id)
        self._loaded.discard(voice_id)
