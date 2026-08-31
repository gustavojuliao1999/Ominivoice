"""Loop de fine-tuning LoRA para uma única voz.

Não usa o `omnivoice.training.trainer.OmniTrainer` (feito para pré-treino
distribuído com Accelerate/DeepSpeed/WebDataset) — em vez disso, um loop
simples de PyTorch, adequado a um único GPU de consumo (ex.: RTX 3060) e a
datasets pequenos (30min a poucas horas de áudio).

O semáforo de GPU é adquirido por passo (não pelo job inteiro), para que
`/tts` e `/transcribe` continuem respondendo intercalados com o treino.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from app import config, dataset_builder, evaluation, voice_store
from app.engine import adapter_manager, get_gpu_semaphore, tts_model

logger = logging.getLogger(__name__)

_training_lock = asyncio.Lock()


def is_training_busy() -> bool:
    return _training_lock.locked()


class _SampleListDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self._processor(self.samples[idx])

    def bind_processor(self, processor) -> None:
        self._processor = processor


def _default_total_steps(num_clips: int) -> int:
    d = config.TRAIN_DEFAULTS
    steps = num_clips * d["steps_per_clip"]
    return max(d["min_steps"], min(d["max_steps"], steps))


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.inference_mode()
def generate_eval_sample(text: str) -> np.ndarray:
    outputs = tts_model.generate(
        text=text,
        instruct=None,
        ref_audio=None,
        num_step=16,
        guidance_scale=3.0,
    )
    audio = outputs[0] if isinstance(outputs, list) else outputs
    if hasattr(audio, "cpu"):
        audio = audio.detach().cpu().numpy()
    return np.squeeze(audio)


async def run_training_job(voice_id: str, train_cfg: dict[str, Any]) -> None:
    if _training_lock.locked():
        raise RuntimeError("Já existe um treinamento em andamento neste servidor")

    async with _training_lock:
        try:
            await _run_training_job_inner(voice_id, train_cfg)
        except Exception as exc:
            logger.exception("Treinamento da voz '%s' falhou", voice_id)

            def mark_failed(meta):
                meta["status"] = voice_store.STATUS_FAILED
                meta["error"] = str(exc)

            try:
                await voice_store.update_meta(voice_id, mark_failed)
            except FileNotFoundError:
                logger.warning("Voz '%s' foi removida durante o treinamento; status não atualizado", voice_id)
            raise


async def _run_training_job_inner(voice_id: str, train_cfg: dict[str, Any]) -> None:
    from omnivoice.data.collator import PaddingDataCollator
    from omnivoice.data.processor import OmniVoiceSampleProcessor
    from omnivoice.training.config import TrainingConfig

    semaphore = get_gpu_semaphore()

    def mark_status(meta):
        meta["status"] = voice_store.STATUS_TRAINING
        meta["error"] = None

    await voice_store.update_meta(voice_id, mark_status)

    logger.info("Preparando dataset de tokens para a voz '%s'...", voice_id)
    async with semaphore:
        samples = await asyncio.to_thread(dataset_builder.build_or_load_samples, voice_id, tts_model)

    num_clips = len(samples)
    total_steps = int(train_cfg.get("steps") or _default_total_steps(num_clips))
    batch_size = int(train_cfg.get("batch_size", config.TRAIN_DEFAULTS["batch_size"]))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", config.TRAIN_DEFAULTS["gradient_accumulation_steps"]))
    learning_rate = float(train_cfg.get("learning_rate", config.TRAIN_DEFAULTS["learning_rate"]))
    eval_every = int(train_cfg.get("eval_every", config.TRAIN_DEFAULTS["eval_every"]))
    max_grad_norm = float(config.TRAIN_DEFAULTS["max_grad_norm"])

    defaults = TrainingConfig()
    processor = OmniVoiceSampleProcessor(
        text_tokenizer=tts_model.text_tokenizer,
        num_channels=tts_model.config.num_audio_codebook,
        audio_mask_id=tts_model.config.audio_mask_id,
        prompt_ratio_range=defaults.prompt_ratio_range,
        mask_ratio_range=defaults.mask_ratio_range,
        drop_cond_ratio=defaults.drop_cond_ratio,
        language_ratio=0.0,  # dataset não anota idioma por amostra
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,  # dataset não anota instruct; foco é o timbre puro
        only_instruct_ratio=0.0,
    )
    collator = PaddingDataCollator(processor, batch_tokens=defaults.batch_tokens)

    dataset = _SampleListDataset(samples)
    dataset.bind_processor(processor)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, num_clips)),
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )

    lora_overrides = {
        k: train_cfg.get(k)
        for k in ("r", "lora_alpha", "lora_dropout")
        if train_cfg.get(k) is not None
    }

    async with semaphore:
        trainable_params = await asyncio.to_thread(
            adapter_manager.prepare_for_training, voice_id, lora_overrides or None
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    warmup_steps = max(10, int(0.03 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = tts_model.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info(
        "Iniciando treino de '%s': %d clipes, %d passos, batch=%d, accum=%d, lr=%g",
        voice_id, num_clips, total_steps, batch_size, grad_accum, learning_rate,
    )

    reference_audio, reference_sr = load_reference_clip(voice_id)
    eval_text = samples[0]["label"]["text"]

    best_similarity = -1.0
    best_step = 0
    start_time = time.time()
    step = 0
    micro_step = 0
    loader_iter = iter(loader)

    def make_progress(extra: Optional[dict] = None):
        elapsed = time.time() - start_time
        rate = step / elapsed if elapsed > 0 else 0.0
        eta = (total_steps - step) / rate if rate > 0 else None
        progress = {
            "step": step,
            "total_steps": total_steps,
            "elapsed_sec": round(elapsed, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "best_step": best_step,
            "best_similarity": None if best_similarity < 0 else round(best_similarity, 4),
        }
        if extra:
            progress.update(extra)
        return progress

    async def persist_progress(extra: Optional[dict] = None):
        def mutate(meta):
            meta.setdefault("training", {})
            meta["training"]["progress"] = make_progress(extra)
            meta["training"]["lora"] = dict(config.LORA_DEFAULTS, **(lora_overrides or {}))
            meta["training"]["config"] = {
                "learning_rate": learning_rate,
                "total_steps": total_steps,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
            }

        await voice_store.update_meta(voice_id, mutate)

    optimizer.zero_grad()
    while step < total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        async with semaphore:

            def train_micro_step():
                tts_model.train()
                device = tts_model.device
                dev_batch = _to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = tts_model(**dev_batch)
                    loss = outputs.loss / grad_accum
                scaler.scale(loss).backward()
                return float(loss.detach().item()) * grad_accum

            loss_value = await asyncio.to_thread(train_micro_step)
            micro_step += 1

            did_step = micro_step % grad_accum == 0
            if did_step:
                def optimizer_step():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    tts_model.eval()

                await asyncio.to_thread(optimizer_step)
                step += 1
            else:
                tts_model.eval()

        # A avaliação/checkpoint só roda logo após um passo de otimizador de
        # fato acontecer (não a cada micro-batch de acumulação de gradiente).
        if not did_step:
            continue

        if step % max(1, eval_every) == 0 or step == total_steps:
            similarity = None
            try:
                async with semaphore:
                    similarity = await asyncio.to_thread(
                        _evaluate_checkpoint, voice_id, eval_text, reference_audio, reference_sr
                    )
            except Exception:
                logger.exception("Avaliação intermediária falhou para '%s' no passo %d", voice_id, step)

            if similarity is not None:
                async with semaphore:
                    await asyncio.to_thread(
                        adapter_manager.save_adapter, voice_id, voice_store.adapter_dir(voice_id, "last")
                    )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_step = step
                    async with semaphore:
                        await asyncio.to_thread(
                            adapter_manager.save_adapter, voice_id, voice_store.adapter_dir(voice_id, "best")
                        )

            await persist_progress({"loss": round(loss_value, 4), "similarity": similarity})
        elif step % 5 == 0:
            await persist_progress({"loss": round(loss_value, 4)})

    async with semaphore:
        await asyncio.to_thread(
            adapter_manager.save_adapter, voice_id, voice_store.adapter_dir(voice_id, "last")
        )
        tts_model.eval()

    active_checkpoint = "best" if best_similarity >= 0 else "last"

    if active_checkpoint == "best":
        # O adapter residente em memória tem os pesos do último passo, que
        # podem não ser os do melhor checkpoint salvo em disco. Descarrega-o
        # para forçar um `load_adapter` a partir de "best" na próxima
        # ativação (evita servir pesos "last" quando o meta diz "best").
        async with semaphore:
            await asyncio.to_thread(adapter_manager.unload, voice_id)

    def finalize(meta):
        meta["status"] = voice_store.STATUS_TRAINED
        meta.setdefault("training", {})
        meta["training"]["progress"] = make_progress()
        meta["training"]["active_checkpoint"] = active_checkpoint
        meta["error"] = None

    await voice_store.update_meta(voice_id, finalize)
    logger.info("Treino da voz '%s' concluído em %d passos (melhor passo=%d)", voice_id, total_steps, best_step)


def load_reference_clip(voice_id: str) -> tuple[np.ndarray, int]:
    manifest = voice_store.read_manifest(voice_id)
    entry = manifest[0]
    path = voice_store.dataset_dir(voice_id) / entry["audio"]
    data, sr = sf.read(str(path), dtype="float32")
    return data, sr


def _evaluate_checkpoint(voice_id: str, text: str, ref_audio: np.ndarray, ref_sr: int) -> Optional[float]:
    tts_model.eval()
    audio = generate_eval_sample(text)
    tts_model.eval()
    try:
        return evaluation.speaker_similarity(audio, tts_model.sampling_rate, ref_audio, ref_sr)
    except Exception:
        logger.exception("Falha ao calcular similaridade de locutor para '%s'", voice_id)
        return None
