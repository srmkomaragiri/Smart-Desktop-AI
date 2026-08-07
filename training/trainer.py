"""
trainer.py — Multi-task training loop
======================================
Orchestrates training with:
  - Multi-task weighted loss
  - Cosine annealing with warm-up
  - Gradient accumulation for small-batch training
  - Mixed precision (torch.cuda.amp)
  - Checkpointing (best model per metric)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


class Trainer:
    """
    Multi-task trainer with gradient accumulation and mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        gradient_accumulation: int = 4,
        mixed_precision: bool = True,
        checkpoint_dir: str = "./training/checkpoints",
        save_best_only: bool = True,
        save_every_n_epochs: int = 10,
        device: Optional[str] = None,
    ) -> None:
        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.gradient_accumulation = gradient_accumulation
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler: warmup + cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=epochs - warmup_epochs,
            T_mult=1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.mixed_precision else None

        # Tracking
        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        logger.info(
            "Trainer initialized — device=%s, epochs=%d, accum=%d, AMP=%s",
            self.device, epochs, gradient_accumulation, self.mixed_precision,
        )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move tensors to device
            batch = self._to_device(batch)

            # Forward pass with optional AMP
            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(batch)
                    loss = outputs["loss"] / self.gradient_accumulation

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(batch)
                loss = outputs["loss"] / self.gradient_accumulation
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += outputs["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation. Returns average validation loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)

            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(batch)
            else:
                outputs = self.model(batch)

            total_loss += outputs["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

    def train(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns
        -------
        Training history dict with per-epoch metrics
        """
        logger.info("Starting training for %d epochs...", self.epochs)
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss = self.validate()
            self.history["val_loss"].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)
            self.scheduler.step()

            epoch_time = time.time() - epoch_start

            msg = (
                f"Epoch {epoch:3d}/{self.epochs} — "
                f"train_loss: {train_loss:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"lr: {current_lr:.6f}, "
                f"time: {epoch_time:.1f}s"
            )
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
                logger.info("  ↳ New best model! val_loss=%.4f", val_loss)

            if not self.save_best_only and epoch % self.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)

        total_time = time.time() - start_time
        logger.info(
            "Training complete — %d epochs in %.1f min, best val_loss=%.4f",
            self.epochs, total_time / 60, self.best_val_loss,
        )

        return self.history

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        """Save model checkpoint."""
        filename = "best.pt" if is_best else f"epoch_{epoch:03d}.pt"
        path = self.checkpoint_dir / filename

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, path)

        logger.debug("Saved checkpoint: %s", path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint. Returns the epoch number.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)

        epoch = checkpoint.get("epoch", 0)
        logger.info("Loaded checkpoint from epoch %d (val_loss=%.4f)", epoch, checkpoint.get("val_loss", 0))
        return epoch

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively move batch tensors to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                result[k] = self._to_device(v)
            else:
                result[k] = v
        return result
