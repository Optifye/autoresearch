"""Training loop for the attentive classifier."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import constants
from src.pipeline import checkpoint_utils


class ClassifierTrainer:
    def __init__(
        self,
        encoder,
        classifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger: logging.Logger,
        checkpoint_path: Path,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.criterion = nn.CrossEntropyLoss()
        self.use_amp = self._should_use_bf16()
        self.autocast_dtype = torch.bfloat16 if self.use_amp else torch.float32
        trainable_params = [p for p in classifier.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("Classifier has no trainable parameters; check freeze configuration.")
        total_params = sum(p.numel() for p in classifier.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=constants.LEARNING_RATE,
            weight_decay=constants.WEIGHT_DECAY,
        )
        self.logger.info(
            "Precision config -> autocast=%s dtype=%s",
            self.use_amp,
            self.autocast_dtype,
        )
        self.logger.info(
            "Trainable classifier parameters: %d / %d",
            trainable_count,
            total_params,
        )

    def _should_use_bf16(self) -> bool:
        if not constants.USE_BFLOAT16:
            return False
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self.logger.warning("bfloat16 requested but CUDA is unavailable; using fp32")
            return False
        supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(supported):
            if not supported():
                self.logger.warning("GPU does not support bfloat16 tensor cores; using fp32")
                return False
        else:
            # Older PyTorch: assume amp works on Ampere+, warn user.
            capability = torch.cuda.get_device_capability(self.device)
            if capability[0] < 8:
                self.logger.warning(
                    "GPU capability %s lacks native bfloat16 support; using fp32",
                    capability,
                )
                return False
        return True

    def train(self):
        best_val_acc = 0.0
        start_time = time.time()
        for epoch in range(1, constants.NUM_EPOCHS + 1):
            train_stats = self._run_epoch(epoch, training=True)
            val_stats = self._run_epoch(epoch, training=False)

            self.logger.info(
                "Epoch %d summary | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%%",
                epoch,
                train_stats["loss"],
                train_stats["accuracy"],
                val_stats["loss"],
                val_stats["accuracy"],
            )

            if val_stats["accuracy"] > best_val_acc:
                best_val_acc = val_stats["accuracy"]
                ckpt_path = checkpoint_utils.save_classifier_checkpoint(
                    classifier_state=self.classifier.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    epoch=epoch,
                    best_metric=best_val_acc,
                    output_path=self.checkpoint_path,
                )
                self.logger.info(
                    "New best accuracy %.2f%% at epoch %d -> saved %s",
                    best_val_acc,
                    epoch,
                    ckpt_path,
                )

        elapsed = time.time() - start_time
        self.logger.info("Training complete in %.2f minutes", elapsed / 60.0)

    def _forward_encoder(self, videos: torch.Tensor) -> torch.Tensor:
        clips = [[videos]]
        outputs = self.encoder(clips, clip_indices=None)
        logits = sum(self.classifier(o) for o in outputs) / len(outputs)
        return logits

    def _run_epoch(self, epoch: int, training: bool) -> Dict[str, float]:
        loader = self.train_loader if training else self.val_loader
        self.classifier.train(mode=training)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        log_every = max(1, len(loader) // 5)

        grad_ctx = nullcontext() if training else torch.inference_mode()

        with grad_ctx:
            for step, (videos, labels) in enumerate(loader, start=1):
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast(dtype=self.autocast_dtype, enabled=self.use_amp):
                    logits = self._forward_encoder(videos)
                    loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += batch_size

                if step % log_every == 0 or step == len(loader):
                    self.logger.info(
                        "%s Epoch %d | step %d/%d | loss %.4f | acc %.2f%%",
                        "Train" if training else "Val",
                        epoch,
                        step,
                        len(loader),
                        total_loss / total_samples,
                        100.0 * total_correct / total_samples,
                    )

        avg_loss = total_loss / max(1, total_samples)
        avg_acc = 100.0 * total_correct / max(1, total_samples)
        return {"loss": avg_loss, "accuracy": avg_acc}
