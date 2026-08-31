from typing import Optional

from pydantic import BaseModel, Field


class VoiceCreateRequest(BaseModel):
    voice_id: str = Field(..., description="Identificador único da voz (ex.: 'joao'). Usado como nome de diretório.")
    name: Optional[str] = None
    language: Optional[str] = None
    description: Optional[str] = None


class TrainRequest(BaseModel):
    steps: Optional[int] = Field(None, description="Total de passos de treino. Se omitido, é derivado do tamanho do dataset.")
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    eval_every: Optional[int] = Field(None, description="A cada quantos passos avaliar e salvar checkpoint")
    r: Optional[int] = Field(None, description="Rank do LoRA")
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None


class EvaluateRequest(BaseModel):
    sample_texts: Optional[list[str]] = Field(
        None, description="Frases customizadas para avaliar. Se omitido, usa frases do próprio dataset."
    )
    num_samples: int = 3


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = Field(None, description="Usa uma voz treinada via fine-tuning (LoRA)")
    ref_audio_url: Optional[str] = None
    ref_text: Optional[str] = ""
    instruct: str = "female, portuguese accent"
    preset: str = Field("balanced", description="'fast' | 'balanced' | 'quality'")
    guidance: Optional[float] = Field(None, description="Sobrescreve o guidance_scale do preset")
    steps: Optional[int] = Field(None, description="Sobrescreve o num_step do preset")
    speed: float = 1.0
    duration: Optional[float] = None
